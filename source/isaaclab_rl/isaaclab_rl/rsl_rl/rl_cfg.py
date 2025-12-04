# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    actor_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the actor network."""

    critic_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the critic network."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers."""

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCriticRecurrent."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""


@configclass
class RslRlTd3ActorCriticCfg:
    """Configuration for the TD3 actor-critic networks."""

    class_name: str = "TD3ActorCritic"
    """The policy class name. Default is TD3ActorCritic."""

    actor_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the actor network."""

    critic_obs_normalization: bool = MISSING
    """Whether to normalize the observation for the critic network."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlActorCriticHistoryCfg(RslRlPpoActorCriticCfg):
    """Configuration for actor-critic networks with observation history."""

    class_name: str = "ActorCriticHistory"
    """The policy class name. Default is ActorCriticHistory."""


@configclass
class RslRlActorCriticDepthCNNCfg:
    """Configuration for actor-critic networks with depth camera observations using generic ActorCriticCNN."""

    class_name: str = "ActorCriticCNN"
    """The policy class name. Uses generic ActorCriticCNN."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_obs_normalization: bool = False
    """Whether to normalize 1D actor observations. Default is False."""

    critic_obs_normalization: bool = False
    """Whether to normalize 1D critic observations. Default is False."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor MLP network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic MLP network."""

    activation: str = MISSING
    """The activation function for networks."""

    actor_cnn_cfg: dict | None = None
    """Configuration for actor CNN. If None, uses default depth CNN config."""

    critic_cnn_cfg: dict | None = None
    """Configuration for critic CNN. If None, uses default depth CNN config."""

    def __post_init__(self):
        """Set default CNN configs if not provided."""
        # Default depth CNN configuration matching our DepthOnlyFCBackbone
        if self.actor_cnn_cfg is None:
            self.actor_cnn_cfg = {
                "output_channels": [16, 32],  # 2 conv layers
                "kernel_size": [5, 3],        # Kernel sizes
                "stride": 1,                   # Stride 1
                "padding": "none",             # No padding (reduces dims)
                "activation": self.activation if hasattr(self, 'activation') and self.activation != MISSING else "elu",
                "max_pool": [True, True],      # Max pool after each conv
                "flatten": True,               # Must flatten for MLP
            }
        if self.critic_cnn_cfg is None:
            self.critic_cnn_cfg = {
                "output_channels": [16, 32],
                "kernel_size": [5, 3],
                "stride": 1,
                "padding": "none",
                "activation": self.activation if hasattr(self, 'activation') and self.activation != MISSING else "elu",
                "max_pool": [True, True],
                "flatten": True,
            }


@configclass
class RslRlActorCriticDepthCNNRecurrentCfg:
    """Configuration for actor-critic networks with depth camera and recurrent layers.

    Note: Currently uses custom ActorCriticDepthCNNRecurrent implementation.
    TODO: Migrate to generic ActorCriticCNNRecurrent when available in rsl_rl.
    """

    class_name: str = "ActorCriticDepthCNNRecurrent"
    """The policy class name. Uses custom ActorCriticDepthCNNRecurrent."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for networks."""

    num_actor_obs_prop: int = 13
    """Number of proprioceptive observations (without depth). Default is 13 for navigation."""

    obs_depth_shape: tuple[int, int] = (53, 30)
    """Shape of depth observations (height, width). Default is (53, 30)."""

    rnn_type: str = "lstm"
    """The type of RNN to use. Either 'lstm' or 'gru'. Default is 'lstm'."""

    rnn_input_size: int = 256
    """The input size for the RNN. Default is 256."""

    rnn_hidden_size: int = 256
    """The hidden size for the RNN. Default is 256."""

    rnn_num_layers: int = 1
    """The number of RNN layers. Default is 1."""

    num_critic_obs_prop: int = 13
    """Number of critic proprioceptive observations (without depth). Default is 13."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """

    rnd_cfg: RslRlRndCfg | None = None
    """The RND configuration. Default is None, in which case RND is not used."""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""


@configclass
class RslRlTd3AlgorithmCfg:
    """Configuration for the TD3 algorithm."""

    class_name: str = "TD3"
    """The algorithm class name. Default is TD3."""

    learning_rate_actor: float = MISSING
    """The learning rate for the actor network."""

    learning_rate_critic: float = MISSING
    """The learning rate for the critic networks."""

    gamma: float = MISSING
    """The discount factor."""

    tau: float = MISSING
    """The soft update coefficient for target networks."""

    policy_noise: float = MISSING
    """Standard deviation of Gaussian noise added to target policy during critic update."""

    noise_clip: float = MISSING
    """Range to clip target policy noise."""

    policy_delay: int = MISSING
    """Frequency of delayed policy updates."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    replay_buffer_size: int = MISSING
    """The size of the replay buffer."""

    batch_size: int = MISSING
    """The batch size for training."""

    exploration_noise: float = MISSING
    """Standard deviation of exploration noise added to actions during data collection."""


@configclass
class RslRlFastTd3AlgorithmCfg(RslRlTd3AlgorithmCfg):
    """Configuration for the FastTD3 algorithm.

    FastTD3 is a variant of TD3 with optimized update frequency and reduced policy delay
    for faster learning in certain environments.
    """

    class_name: str = "FastTD3"
    """The algorithm class name. Default is FastTD3."""

    policy_delay: int = 1
    """Frequency of delayed policy updates. Default is 1 for FastTD3 (faster than standard TD3)."""

    num_critic_updates: int = MISSING
    """Number of critic updates per environment step."""


#########################
# Runner configurations #
#########################


@configclass
class RslRlBaseRunnerCfg:
    """Base configuration of the runner."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool | None = None
    """This parameter is deprecated and will be removed in the future.

    Use `actor_obs_normalization` and `critic_obs_normalization` instead.
    """

    obs_groups: dict[str, list[str]] = MISSING
    """A mapping from observation groups to observation sets.

    The keys of the dictionary are predefined observation sets used by the underlying algorithm
    and values are lists of observation groups provided by the environment.

    For instance, if the environment provides a dictionary of observations with groups "policy", "images",
    and "privileged", these can be mapped to algorithmic observation sets as follows:

    .. code-block:: python

        obs_groups = {
            "policy": ["policy", "images"],
            "critic": ["policy", "privileged"],
        }

    This way, the policy will receive the "policy" and "images" observations, and the critic will
    receive the "policy" and "privileged" observations.

    For more details, please check ``vec_env.py`` in the rsl_rl library.
    """

    clip_actions: float | None = None
    """The clipping value for actions. If None, then no clipping is done. Defaults to None.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    resume: bool = False
    """Whether to resume a previous training. Default is False.

    This flag will be ignored for distillation.
    """

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """


@configclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "OnPolicyRunner"
    """The runner class name. Default is OnPolicyRunner."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""


@configclass
class RslRlOnPolicyRunnerHistoryCfg(RslRlOnPolicyRunnerCfg):
    """Configuration of the runner for on-policy algorithms with observation history."""

    class_name: str = "OnPolicyRunnerHistory"
    """The runner class name. Default is OnPolicyRunnerHistory."""

    policy: RslRlActorCriticHistoryCfg = MISSING
    """The policy configuration."""


@configclass
class RslRlOffPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration of the runner for off-policy algorithms."""

    class_name: str = "OffPolicyRunner"
    """The runner class name. Default is OffPolicyRunner."""

    policy: RslRlTd3ActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlTd3AlgorithmCfg = MISSING
    """The algorithm configuration."""

    random_steps: int = MISSING
    """Number of random exploration steps before using policy."""

    gradient_steps: int = MISSING
    """Number of gradient steps per environment step."""
