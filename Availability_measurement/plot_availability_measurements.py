import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

availability = [97.81, 98.43, 99.39, 96.41, 98.81,
                  99.40, 96.09, 98.19, 95.22, 98.79 ]

error_rate = [2.18, 1.56, 0.60, 3.58, 1.18, 
              0.59, 3.90, 1.80, 4.77, 1.20 ]

latency_p95 = [112.99, 209.77, 138.87, 68.85, 45.78,  
               34.64, 59.69, 36.32, 43.53, 32.16 ]

df = pd.DataFrame({
    "Availability (%)": availability,
    "Request Error Rate (%)": error_rate,
    "Request Latency p95 (ms)": latency_p95
})

# Style

sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300
})

# Figure

fig, axes = plt.subplots(1, 3, figsize=(8, 3.2))

metrics = [
    "Availability (%)",
    "Request Error Rate (%)",
    "Request Latency p95 (ms)"
]

# Plot

for ax, metric in zip(axes, metrics):

    sns.boxplot(
        y=df[metric],
        ax=ax,
        width=0.35,
        color="lightgray",
        linewidth=1,
        fliersize=0
    )

    sns.stripplot(
        y=df[metric],
        ax=ax,
        color="black",
        size=4,
        jitter=0.08
    )

    # Mean (diamond)
    mean_val = np.mean(df[metric])

    print("MEAN VALUES: ")
    print(mean_val)

    ax.scatter(
        0,
        mean_val,
        marker="D",
        s=35,
        color="black",
        zorder=5
    )

    ax.set_title(metric)
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.grid(True, linestyle=":", linewidth=0.4)

fig.suptitle(
    "Availability, Error Rate, and p95 Latency Across ReactiTwin Runs",
    fontsize=10,
    y=1.05
)

legend_elements = [
    Line2D([0], [0], marker='s', color='gray', label='Boxplot (IQR 25–75%)',
           markerfacecolor='lightgray', markersize=8, linestyle='None'),
    
    Line2D([0], [0], marker='o', color='black', label='Observations',
           linestyle='None', markersize=4),

    Line2D([0], [0], marker='D', color='black', label='Mean value',
           linestyle='None', markersize=6)
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=3,
    frameon=True,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.02)
)

# Layout

plt.tight_layout(pad=1.0)
plt.subplots_adjust(bottom=0.22)

# Save

plt.savefig(
    "service_performance_metrics_boxplot.png",
    bbox_inches="tight"
)

plt.show()