#Created on We Mai 29 15:56:16 2026
#@author: kambana

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.ticker import MultipleLocator, FuncFormatter

# GLOBAL VARIABLE
CENSORED_VALUE = 700
OBSERVATION_UNCENSORED = 0
TOTAL_OF_OBSERVATION = 0

# PREPROCESSING

def build_survival_data(arr, censor_value=700):
    arr = np.array(arr, dtype=float)

    event = (arr != censor_value).astype(int)
    duration = arr.copy()   

    return duration, event

# KAPLAN-MEIER plotting curve
def plot_km(dur_a, evt_a, dur_b, evt_b,
            label_a, label_b,
            title,
            filename="km_plot.png"):

    # Initialize Kaplan-Meier models
    kmf_a = KaplanMeierFitter()
    kmf_b = KaplanMeierFitter()

    # Create figure
    plt.figure(figsize=(8, 5))

    # Fit and plot first group
    kmf_a.fit(
        dur_a,
        event_observed=evt_a,
        label=label_a
    )

    ax = kmf_a.plot_survival_function(
        ci_show=True,
        ci_alpha=0.15,
        linewidth=2
    )

    # Fit and plot second group
    kmf_b.fit(
        dur_b,
        event_observed=evt_b,
        label=label_b
    )

    kmf_b.plot_survival_function(
        ax=ax,
        ci_show=True,
        ci_alpha=0.15,
        linewidth=2
    )

    # Tick spacing (fine grid)
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # Show labels only every 0.1 on X (clean & readable)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x:.1f}" if abs(x*10 - round(x*10)) < 1e-6 else "")
    )

    # Better readability
    ax.tick_params(axis='both', labelsize=9)
    

    # Extract line colors from plotted curves
    color_a = ax.lines[0].get_color()
    color_b = ax.lines[1].get_color()

    # Create confidence interval patches
    ci_patch_a = mpatches.Patch(
        color=color_a,
        alpha=0.15,
        label=f"{label_a} - 95% CI"
    )

    ci_patch_b = mpatches.Patch(
        color=color_b,
        alpha=0.15,
        label=f"{label_b} - 95% CI"
    )

    # Plot styling
    plt.title(title)
    plt.xlim(0.2, 0.8)
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.grid(True, alpha=0.3)

    # Get line handles already created by lifelines
    line_handles, line_labels = ax.get_legend_handles_labels()

    # Create complete legend
    plt.legend(
        handles=line_handles + [ci_patch_a, ci_patch_b],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False
    )

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    # Show figure
    plt.show()


# TTR ON (PROCESSING)
ttr_on_rawed = np.array([
    0.29,0.29,0.29,0.30,0.32,0.29,0.31,0.28,0.29,0.28,
    0.30,0.33,0.29,0.29,0.30,0.28,0.28,0.27,0.31,0.31,
    0.28,0.30,0.29,0.31,0.28,0.29,0.33,0.30,0.30,700,
    0.33,0.31,0.31,0.32,700,0.29,0.32,0.30,0.29,0.29,
    700,0.30,700,0.26,0.32,700,0.29,700,0.33,0.28
])

ttr_on_timedistrib, ttr_on_event = build_survival_data(ttr_on_rawed)


# TTR OFF (BASELINE)
ttr_off_raw = np.array([
    700, 700, 700, 700, 700, 700, 700, 700, 700, 700,
    700, 700, 700, 700, 700, 700, 700, 700, 700, 700,
    700, 700, 700, 700, 700, 700, 700, 700, 700, 700,
    700, 700, 700, 700, 700, 700, 700, 700, 700, 700,
    700, 700, 700, 700, 700, 700, 700, 700, 700, 700
])

ttr_off_timedistrib, ttr_off_event = build_survival_data(ttr_off_raw)


# TTD ON (PROCESSING)
ttd_on_rawed = np.array([
    0.51,0.51,0.57,0.51,0.53,0.53,0.60,0.55,0.57,0.54,
    0.50,0.58,0.51,0.53,0.54,0.52,0.52,0.52,0.50,0.53,
    0.53,0.53,0.53,0.57,0.53,0.58,0.52,0.52,0.53,700,
    0.57,0.52,0.54,0.54,700,0.54,0.52,0.51,0.53,0.57,
    700,0.51,700,0.50,0.55,700,0.53,700,0.57,0.53
])

ttd_on_timedistrib, ttd_on_event = build_survival_data(ttd_on_rawed)

# TTD OFF (BASELINE)
ttd_off_rawed = np.array([
    700, 700, 700, 700, 700, 700, 700, 700, 700, 700,
    700, 700, 700, 700, 700, 700, 700, 700, 700, 700,
    700, 700, 700, 700, 700, 700, 700, 700, 700, 700,
    700, 700, 700, 700, 700, 700, 700, 700, 700, 700,
    700, 700, 700, 700, 700, 700, 700, 700, 700, 700
])

ttd_off_timedistrib, ttd_off_event = build_survival_data(ttd_off_rawed)

# IC 95% of WILSON
OBSERVATION_UNCENSORED = np.count_nonzero(ttd_on_rawed != 400)
TOTAL_OF_OBSERVATION = len(ttd_on_rawed)
low, high = proportion_confint(
    count= OBSERVATION_UNCENSORED,
    nobs= TOTAL_OF_OBSERVATION,
    alpha=0.05,
    method="wilson"
)

print(f"Proportion : {OBSERVATION_UNCENSORED/TOTAL_OF_OBSERVATION:.2%}")
print(f"IC 95 % de Wilson : [{low:.2%}, {high:.2%}]")

# PLOTS KM + IC 95%

plot_km(
    ttd_on_timedistrib, ttd_on_event,
    ttd_off_timedistrib, ttd_off_event,
    "ReactiTwin Enabled", "ReactiTwin Disabled",
    "Kaplan-Meier curves of Time-To-Detect (TTD)",
    filename="ttd_km.png"
)

plot_km(
    ttr_on_timedistrib, ttr_on_event,
    ttr_off_timedistrib, ttr_off_event,
    "ReactiTwin Enabled", "ReactiTwin Disabled",
    "Kaplan-Meier curves of Time-To-Remediation (TTR)",
    filename="ttr_km.png"
)

# LOG-RANK TEST

result_ttd = logrank_test(
    ttd_on_timedistrib,
    ttd_off_timedistrib,
    event_observed_A=ttd_on_event,
    event_observed_B=ttd_off_event
)

print("\n TTD: Log-rank test ")
print("Chi-square statistic:", result_ttd.test_statistic)
print("p-value:", result_ttd.p_value)


result_ttr = logrank_test(
    ttr_on_timedistrib,
    ttr_off_timedistrib,
    event_observed_A=ttr_on_event,
    event_observed_B=ttr_off_event
)

print("\n TTR: Log-rank test ")
print("Chi-square statistic:", result_ttr.test_statistic)
print("p-value:", result_ttr.p_value)

# Print median and IQR of raw TTD and TTR distributions

def print_median_iqr(data, label):
    
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    print(f"\n{label}")
    print(f"Median: {median:.3f} s")
    print(f"Q1: {q1:.3f} s")
    print(f"Q3: {q3:.3f} s")
    print(f"IQR: {iqr:.3f} s")

    return median, q1, q3, iqr

# TTD
print("TTD: Median and IQR")

ttd_on_median, ttd_on_q1, ttd_on_q3, ttd_on_iqr = print_median_iqr(
    ttd_on_rawed,
    "TTD - ReactiTwin Enabled"
)

# TTR
print("TTR: Median and IQR")

ttr_on_median, ttr_on_q1, ttr_on_q3, ttr_on_iqr = print_median_iqr(
    ttr_on_rawed,
    "TTR - ReactiTwin Enabled"
)

#show TTD and TTR as a discret line plot with markers

# show 10^-3 value
def fmt_3dec(x, pos):
    if np.isinf(x):
        return "∞"
    return f"{x:.3f}"

def replace_infinite(arr):
    arr = arr.astype(float)
    arr[arr == CENSORED_VALUE] = np.inf
    return arr

ttr_on = replace_infinite(ttr_on_rawed)
ttr_off = replace_infinite(ttr_off_raw)
ttd_on = replace_infinite(ttd_on_rawed)
ttd_off = replace_infinite(ttd_off_rawed)

x = np.arange(len(ttr_on))

# TTD
plt.figure(figsize=(10, 5))
ttd_on_plot = np.where(np.isinf(ttd_on), 1.2, ttd_on)
ttd_off_plot = np.where(np.isinf(ttd_off), 1.2, ttd_off)

plt.plot(
    x,
    ttd_on_plot,
    marker='o',
    linestyle='-',
    color='#1f77b4',
    label='ReactiTwin Enabled'
)

plt.plot(
    x,
    ttd_off_plot,
    marker='s',
    linestyle='--',
    color='#ff7f0e',
    label='ReactiTwin Disabled'
)

plt.gca().yaxis.set_major_formatter(FuncFormatter(fmt_3dec))

plt.xticks(np.arange(0, len(x), 2))
yticks = list(np.arange(0.2, 1.1, 0.1))
yticks += [1.1, 1.2]
yticklabels = [f"{v:.3f}" for v in yticks[:-2]]
yticklabels += ["", "∞"]
plt.yticks(yticks, yticklabels)

plt.scatter(
    x[np.isinf(ttd_off)],
    np.ones_like(x[np.isinf(ttd_off)]) * 1.2,
    marker='s',
    color='#ff7f0e'
)

plt.title("Time-To-Detection (TTD) in ReactiTwin under ON/OFF Conditions",
          fontsize=14, pad=15)

plt.xlabel("Experimental Run")
plt.ylabel("Time (s)")
plt.ylim(0, 1.3)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("TTD_ON_OFF.png", dpi=300, bbox_inches='tight')
plt.show()


# TTR
plt.figure(figsize=(10, 5))

ttr_on_plot = np.where(np.isinf(ttr_on), 0.60, ttr_on)
ttr_off_plot = np.where(np.isinf(ttr_off), 0.60, ttr_off)

plt.plot(
    x,
    ttr_on_plot,
    marker='o',
    linestyle='-',
    color='#1f77b4',
    label='ReactiTwin Enabled'
)

plt.plot(
    x,
    ttr_off_plot,
    marker='s',
    linestyle='--',
    color='#ff7f0e',
    label='ReactiTwin Disabled'
)

plt.gca().yaxis.set_major_formatter(FuncFormatter(fmt_3dec))

plt.xticks(np.arange(0, len(x), 2))


yticks = list(np.arange(0, 0.57, 0.03))
yticks += [0.57, 0.60]

yticklabels = [f"{v:.3f}" for v in yticks[:-2]]
yticklabels += ["", "∞"]     

plt.yticks(yticks, yticklabels)

plt.scatter(
    x[np.isinf(ttr_off)],
    np.ones_like(x[np.isinf(ttr_off)]) * 0.60,
    marker='s',
    color='#ff7f0e'
)

plt.title("Time-To-Remediation (TTR) in ReactiTwin under ON/OFF Conditions",
          fontsize=14, pad=15)

plt.xlabel("Experimental Run")
plt.ylabel("Time (s)")
plt.ylim(0.2, 0.620)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("TTR_ON_OFF.png", dpi=300, bbox_inches='tight')
plt.show()
