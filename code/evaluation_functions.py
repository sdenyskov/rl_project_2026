import torch

def total_epis_collisions(collision_counter):
    return collision_counter.sum().item() / 2

def avg_pwise_dist(positions):
    n_agents = positions.shape[0]
    distances = torch.cdist(positions, positions, p=2)
    mask = ~torch.eye(n_agents, dtype=torch.bool, device=positions.device) # to remove self-distances
    return distances[mask].mean().item()

def max_pwise_dist(positions):
    distances = torch.cdist(positions, positions, p=2)
    return distances.max().item()

def dir_alignment(velocities): # 0 - random dirs, 1 - perfect alignment
    velocity_norms = torch.linalg.vector_norm(velocities, dim=-1, keepdim=True)
    directions = velocities / (velocity_norms + 1e-8) # avoid div by 0
    mean_direction = directions.mean(dim=0)
    mean_direction = mean_direction / (torch.linalg.vector_norm(mean_direction) + 1e-8)
    alignment = (directions @ mean_direction).mean()
    return alignment.item()

def comp_number(positions, radius):
    n_agents = positions.shape[0]
    distances = torch.cdist(positions, positions, p=2)
    adjacency = distances < radius
    visited = torch.zeros(n_agents, dtype=torch.bool)
    components = 0

    for i in range(n_agents):
        if visited[i]:
            continue
        components += 1
        stack = [i]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            neighbours = torch.where(adjacency[node])[0]
            stack.extend(neighbours.tolist())

    return components

def gaus_entr(x):
    x = x - x.mean(dim=0)
    cov = (x.T @ x / (x.shape[0] - 1))
    cov += (1e-6 * torch.eye(cov.shape[0]))
    _, logdet = torch.slogdet(cov)
    entropy = 0.5 * logdet
    return entropy.item()

def phi_proxy(states):
    t, n, features = states.shape
    joint_entr = gaus_entr(states.reshape(t, n * features))
    indiv_entr = 0
    for i in range(n):
        agent_state = states[:, i, :]
        indiv_entr += gaus_entr(agent_state)
    return indiv_entr - joint_entr

def phi_ts(states, w=50):
    phi_vals=[]
    for t in range(w, states.shape[0]):
        w_states = states[t-w:t]
        phi_vals.append(phi_proxy(w_states))
    return phi_vals
