import matplotlib.pyplot as plt
import torch
import sys

if __name__ == "__main__":
    name = str(sys.argv[1])

data = torch.load(f"envs/{name}.pt")

agents = data["agents"][0]
plt.scatter(agents[:,0], agents[:,1], label="agents")

if data["obstacles"].shape[1] > 0:
    obstacles = data["obstacles"][0]
    plt.scatter(obstacles[:,0], obstacles[:,1], marker="x", label="obstacles")

target = data["target"][0]
plt.scatter(target[0], target[1], marker="*", s=200, label="target")

plt.axis("equal")
plt.legend()
plt.savefig(f"envs/{name}.png")
