import sys
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import evaluation_functions as efun
import importlib.util
import json

from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import step_mdp
from tensordict.nn import TensorDictModule
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal, NormalParamExtractor

warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    model_name = str(sys.argv[1])

    obstacles_name = str(sys.argv[2])
    obstacles_file = None if obstacles_name == "" else "envs/" + obstacles_name + ".pt"
    agents_name = str(sys.argv[3])
    agents_file = None if agents_name == "" else "envs/" + agents_name + ".pt"
    destination_name = str(sys.argv[4])
    destination_file = None if destination_name == "" else "envs/" + destination_name + ".pt"

    message = str(sys.argv[5])
    record_gif = True if str(sys.argv[6]) == "gif" else False
    record_plot = True if str(sys.argv[7]) == "plot" else False


# config

date_time = datetime.now().strftime("%Y%m%d-%H%M%S")

module_name = "config_" + model_name
module_path = "configs/config_" + model_name + ".py"
spec = importlib.util.spec_from_file_location(module_name, module_path)
config = importlib.util.module_from_spec(spec)
sys.modules[module_name] = config
spec.loader.exec_module(config)

max_steps = 500
gif_duration = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = f"models/{model_name}.pt"

# environment

env = VmasEnv(
    scenario="flocking_project",
    num_envs=1,  # single env for clean evaluation
    continuous_actions=True,
    max_steps=max_steps,
    device=device,
    n_agents=config.n_agents,
    n_obstacles=config.n_obstacles,
    spawn_agents_together=config.spawn_agents_together,
    boundary_type=config.boundary_type,
    obstacles_file=obstacles_file,
    agents_file=agents_file,
    destination_file=destination_file,
    collision_reward=config.collision_reward,
    dist_shaping_factor=config.dist_shaping_factor,
    phi_reward_scale=config.phi_reward_scale,
    phi_window=config.phi_window,
    communication_range_dist=config.communication_range_dist,
    communication_range_phi=config.communication_range_phi,
    localise_phi=config.localise_phi,
    max_range_lidar=config.max_range_lidar
)

# policy (matches training exactly)

policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=2 * env.full_action_spec[env.action_key].shape[-1],
        n_agents=env.n_agents,
        centralised=False,
        share_params=True,
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
    return_log_prob=False,
)

policy.to(device)
policy.eval()

# loading trained weights

print(f"\nLoading model from {checkpoint_path}")
policy.load_state_dict(torch.load(checkpoint_path, map_location=device))

# evaluation loop

frames = []

episode_alignment = []
episode_cohesion = []
episode_diameter = []
episode_collisions = []
episode_connectivity = []
state_history = []

td = env.reset()

for t in range(max_steps):

    with torch.no_grad():
        td = policy(td)

    td = env.step(td)
    td = step_mdp(td)

    if record_gif:
        frame = env.render(mode="rgb_array")
        frames.append((frame, t))

    positions = td.get(("agents", "info", "position")).squeeze(0)
    if True:
        t_position = td.get(("agents", "info", "t_position")).squeeze(0)
        positions = torch.cat([positions, t_position[:1]], dim=0)
    velocities = td.get(("agents", "info", "speed")).squeeze(0)
    if True:
        t_velocity = td.get(("agents", "info", "t_speed")).squeeze(0)
        velocities = torch.cat([velocities, t_velocity[:1]], dim=0)
    collisions = td.get(("agents", "info", "collision_counter")).squeeze(0)
    
    episode_alignment.append(efun.dir_alignment(velocities))
    episode_cohesion.append(efun.avg_pwise_dist(positions))
    episode_diameter.append(efun.max_pwise_dist(positions))
    episode_collisions.append(efun.total_epis_collisions(collisions))
    episode_connectivity.append(efun.comp_number(positions, radius=0.5))
    state_history.append(torch.cat([positions, velocities], dim=-1).cpu())
    
states = torch.stack(state_history)

window = 50
phi_history = efun.phi_ts(states, w=window)

results = {
    'obstacles': obstacles_name, 
    'agents': agents_name, 
    'avg_alignm_after_100': np.mean(episode_alignment[100:]),
    'avg_cohesi_after_100': np.mean(episode_cohesion[100:]),
    'avg_diamet_after_100': np.mean(episode_diameter[100:]),
    'avg_collis_after_100': np.mean(episode_collisions[100:]),
    'avg_connec_after_100': np.mean(episode_connectivity[100:]),
    'avg_phipxy_after_100': np.mean(phi_history[50:]),
    'all_phipxy': efun.phi_proxy(states)
}
print(json.dumps(results))

if record_plot: 

    steps = range(len(episode_alignment))
    fig, ax = plt.subplots(3, 2, figsize=(12, 10))

    ax[0,0].plot(steps, episode_alignment)
    ax[0,0].set_ylabel("Alignment")
    ax[0,0].grid()
    ax[0,0].set_title(f"Avg after step 100 = {np.mean(episode_alignment[100:]):.4f}")

    ax[0,1].plot(steps, episode_cohesion)
    ax[0,1].set_ylabel("Cohesion (avg pairwise distance)")
    ax[0,1].grid()
    ax[0,1].set_title(f"Avg after step 100 = {np.mean(episode_cohesion[100:]):.4f}")

    ax[1,0].plot(steps, episode_diameter)
    ax[1,0].set_ylabel("Diameter (max pairwise distance)")
    ax[1,0].grid()
    ax[1,0].set_title(f"Avg after step 100 = {np.mean(episode_diameter[100:]):.4f}")

    ax[1,1].plot(steps, episode_collisions)
    ax[1,1].set_ylabel("Collision count")
    ax[1,1].grid()
    ax[1,1].set_title(f"Avg after step 100 = {np.mean(episode_collisions[100:]):.4f}")

    ax[2,0].plot(steps, episode_connectivity)
    ax[2,0].set_xlabel("Step")
    ax[2,0].set_ylabel("Connectivity (num of components)")
    ax[2,0].grid()
    ax[2,0].set_title(f"Avg after step 100 = {np.mean(episode_connectivity[100:]):.4f}")

    ax[2,1].plot(range(window, len(episode_alignment)), phi_history)
    ax[2,1].set_xlabel("Step")
    ax[2,1].set_ylabel("Integrated information proxy")
    ax[2,1].grid()
    ax[2,1].set_title(f"Avg after step 100 = {np.mean(phi_history[100:]):.4f}")

    plt.savefig(f"media/plots/{message}_{model_name}_{obstacles_name}_{agents_name}.png")

if record_gif:

    gif_frames = []

    for frame, step in frames:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.rectangle([5, 5, 180, 45], fill="white")
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
        draw.text((10, 10), f"Step: {step}", fill="black", font=font)
        gif_frames.append(img)

    gif_frames[0].save(
        f"media/gifs/{message}_{model_name}_{obstacles_name}_{agents_name}.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=gif_duration,
        loop=0,
    )
    print("GIF saved")
