# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ZMQ client for communicating with Isaac-Gr00t inference server.

This module is adapted from Isaac-Gr00t's server_client.py to be self-contained
without requiring the full Isaac-Gr00t package as a dependency.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Callable

import msgpack
import numpy as np
import zmq


@dataclass
class ModalityConfig:
    """Configuration for a modality defining how data should be sampled and loaded.

    This is a simplified version of the Isaac-Gr00t ModalityConfig for client-side use.
    """

    delta_indices: list[int]
    """Delta indices to sample relative to the current index."""

    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""

    sin_cos_embedding_keys: list[str] | None = None
    """Optional list of keys to apply sin/cos encoding."""

    mean_std_embedding_keys: list[str] | None = None
    """Optional list of keys to apply mean/std normalization."""


def _to_json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, ModalityConfig):
        return {
            "delta_indices": obj.delta_indices,
            "modality_keys": obj.modality_keys,
            "sin_cos_embedding_keys": obj.sin_cos_embedding_keys,
            "mean_std_embedding_keys": obj.mean_std_embedding_keys,
        }
    return obj


class MsgSerializer:
    """Serializer for ZMQ messages using msgpack with numpy array support."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        """Serialize data to bytes using msgpack."""
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        """Deserialize bytes to data using msgpack."""
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj: Any) -> Any:
        """Decode custom classes from msgpack format."""
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            return ModalityConfig(**obj["as_json"])
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj: Any) -> Any:
        """Encode custom classes to msgpack format."""
        if isinstance(obj, ModalityConfig):
            return {"__ModalityConfig_class__": True, "as_json": _to_json_serializable(obj)}
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


@dataclass
class EndpointHandler:
    """Handler for a server endpoint."""

    handler: Callable
    requires_input: bool = True


class PolicyClient:
    """ZMQ client for communicating with Isaac-Gr00t PolicyServer.

    This client connects to a running Isaac-Gr00t inference server and provides
    methods for getting actions from the policy.

    Example:
        >>> client = PolicyClient(host="localhost", port=5555)
        >>> if client.ping():
        ...     action, info = client.get_action(observation)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ):
        """Initialize the PolicyClient.

        Args:
            host: Server hostname or IP address.
            port: Server port number.
            timeout_ms: Request timeout in milliseconds.
            api_token: Optional API token for authentication.
        """
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self) -> None:
        """Initialize or reinitialize the socket with current settings."""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        """Check if the server is running.

        Returns:
            True if server responds, False otherwise.
        """
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    def kill_server(self) -> None:
        """Send kill signal to the server."""
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self,
        endpoint: str,
        data: dict | None = None,
        requires_input: bool = True,
    ) -> Any:
        """Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.

        Returns:
            The response from the server.

        Raises:
            RuntimeError: If the server returns an error.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure the correct policy server is running.")
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get action from the policy server.

        Args:
            observation: Observation dictionary with the following structure:
                - "video": dict mapping camera names to numpy arrays of shape (B, T, H, W, 3)
                - "state": dict mapping state names to numpy arrays of shape (B, T, D)
                - "language": dict mapping language keys to list[list[str]]
            options: Optional dictionary of additional options.

        Returns:
            Tuple of (action_dict, info_dict) where:
                - action_dict: dict mapping action names to numpy arrays of shape (B, T_action, D)
                - info_dict: Additional information from the policy.
        """
        response = self.call_endpoint("get_action", {"observation": observation, "options": options})
        return tuple(response)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy state.

        Args:
            options: Optional dictionary of reset options.

        Returns:
            Reset information dictionary.
        """
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Get the modality configuration from the server.

        Returns:
            Dictionary mapping modality names to ModalityConfig objects.
        """
        return self.call_endpoint("get_modality_config", requires_input=False)

    def close(self) -> None:
        """Close the client connection and clean up resources."""
        self.socket.close()
        self.context.term()

    def __del__(self) -> None:
        """Cleanup resources on destruction."""
        try:
            self.socket.close()
            self.context.term()
        except Exception:
            pass
