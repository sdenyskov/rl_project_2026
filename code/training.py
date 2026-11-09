import sys
import torch
import warnings
from tqdm import tqdm
from datetime import datetime
import importlib.util

from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

from torchrl.collectors import Collector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# warning suppression and parameter setting

warnings.filterwarnings("ignore", category=FutureWarning)
torch.manual_seed(0)
date_time = datetime.now().strftime("tr-%Y%m%d-%H%M%S")

if __name__ == "__main__":
    model_name = str(sys.argv[1])

module_name = "config_" + model_name
module_path = "configs/config_" + model_name + ".py"
spec = importlib.util.spec_from_file_location(module_name, module_path)
config = importlib.util.module_from_spec(spec)
sys.modules[module_name] = config
spec.loader.exec_module(config)

# defining hyperparameters

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
vmas_device = device

frames_per_batch = config.frames_per_batch  # team frames collected per training iteration
n_iters = config.n_iters  # sampling and training iterations
total_frames = frames_per_batch * n_iters

num_epochs = 10  # optimization steps per training iteration
minibatch_size = 400  # size of the mini-batches in each optimization step
lr = 5e-5  # learning rate of Adam optimizer
max_grad_norm = 1.0

clip_epsilon = 0.2  # clip value for PPO loss
gamma = 0.99  # future reward discount factor
lmbda = 0.9  # lambda for generalised advantage estimation (GAE)
entropy_eps = 5e-4  # coefficient of the entropy term in the PPO loss

set_composite_lp_aggregate(False).set()

# setting up environment

max_steps = 500  # episode steps before done
num_vmas_envs = (frames_per_batch // max_steps)

env = VmasEnv(
    scenario="flocking_project",
    num_envs=num_vmas_envs, 
    continuous_actions=True,
    max_steps=max_steps,
    device=vmas_device,
    # scenario kwargs
    n_agents=config.n_agents,
    n_obstacles=config.n_obstacles,
    spawn_agents_together=config.spawn_agents_together,
    boundary_type=config.boundary_type,
    obstacles_file=None,
    agents_file=None,
    destination_file=None,
    collision_reward=config.collision_reward,
    dist_shaping_factor=config.dist_shaping_factor,
    phi_reward_scale=config.phi_reward_scale,
    phi_window=config.phi_window,
    communication_range_dist=config.communication_range_dist,
    communication_range_phi=config.communication_range_phi,
    localise_phi=config.localise_phi, 
    max_range_lidar=config.max_range_lidar
)

env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)

# small test rollout

n_rollout_steps = 5
rollout = env.rollout(n_rollout_steps)

# policy

share_parameters_policy = True

policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=2 * env.full_action_spec[env.action_key].shape[-1],
        n_agents=env.n_agents,
        centralised=False,
        share_params=share_parameters_policy,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    ),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    policy_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "loc"), ("agents", "scale")],
)

policy = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec_unbatched,
    in_keys=[("agents", "loc"), ("agents", "scale")],
    out_keys=[env.action_key],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.full_action_spec_unbatched[env.action_key].space.low,
        "high": env.full_action_spec_unbatched[env.action_key].space.high,
    },
    return_log_prob=True,
)

# critic network

share_parameters_critic = True
mappo = True  # IPPO if False

critic_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    n_agent_outputs=1,
    n_agents=env.n_agents,
    centralised=mappo,
    share_params=share_parameters_critic,
    device=device,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)

critic = TensorDictModule(
    module=critic_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "state_value")],
)

# data collector

collector = Collector(
    env,
    policy,
    device=vmas_device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

# replay buffer

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=device
    ),
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,
)

# loss function

loss_module = ClipPPOLoss(
    actor_network=policy,
    critic_network=critic,
    clip_epsilon=clip_epsilon,
    entropy_coeff=entropy_eps,
    normalize_advantage=False,
)

loss_module.set_keys(
    reward=env.reward_key,
    action=env.action_key,
    value=("agents", "state_value"),
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)

loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)
GAE = loss_module.value_estimator

optim = torch.optim.Adam(loss_module.parameters(), lr)

# (!!!) training loop

pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

episode_reward_mean_list = []
for tensordict_data in collector:
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )
    tensordict_data.set(
        ("next", "agents", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )

    with torch.no_grad():
        GAE(
            tensordict_data,
            params=loss_module.critic_network_params,
            target_params=loss_module.target_critic_network_params,
        )

    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view)

    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )

            optim.step()
            optim.zero_grad()

    collector.update_policy_weights_()

    # logging
    done = tensordict_data.get(("next", "agents", "done"))
    episode_reward_mean = (
        tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list.append(episode_reward_mean)
    pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
    pbar.update()

# saving the model

torch.save(policy.state_dict(), f"models/{model_name}.pt")
print(f"Model saved at models/{model_name}.pt")
