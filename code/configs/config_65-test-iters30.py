n_agents = 6
n_obstacles = 5

spawn_agents_together = True
boundary_type = "none" # "periodic", "clip", "walls"

collision_reward = -0.02
dist_shaping_factor = 1.0
phi_reward_scale = 0.05

phi_window = 30
communication_range_dist = 1000.0
communication_range_phi = 1000.0
localise_phi = False
max_range_lidar = 0.2

frames_per_batch = 18000
n_iters = 30
