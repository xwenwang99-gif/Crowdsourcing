import numpy as np
import matplotlib.pyplot as plt

rho = np.array([0.00, 0.25, 0.50, 0.75])

means = {
    "LiSER": [0.8935, 0.9120, 0.8740, 0.7815],
    "Dawid--Skene": [0.82005, 0.7720, 0.7720, 0.7055],
    "CBCC": [0.80035, 0.7485, 0.7890, 0.6685],
    "Majority Vote": [0.50160, 0.4885, 0.4500, 0.4045],
}

sds = {
    "LiSER": [0.094314, 0.053396, 0.085693, 0.151677],
    "Dawid--Skene": [0.187263, 0.169676, 0.145778, 0.215180],
    "CBCC": [0.219039, 0.212577, 0.203016, 0.265853],
    "Majority Vote": [0.088680, 0.061012, 0.057106, 0.051448],
}

markers = ["o", "s", "^", "D"]

plt.figure(figsize=(7, 4.8))

for (method, values), marker in zip(means.items(), markers):
    plt.errorbar(
        rho, values, yerr=sds[method],
        marker=marker, linewidth=2, markersize=6,
        capsize=4, label=method
    )

plt.xlabel(r"Latent-overlap parameter $\rho$")
plt.ylabel("Label accuracy")
plt.xticks(rho)
plt.ylim(0, 1)
plt.legend(frameon=False)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig("rho_sensitivity_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()