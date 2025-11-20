import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import numpy as np
import os

# Configure data file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
runs516_path = os.path.join(DATA_DIR, '516 strategies with results.csv')
ga_path = os.path.join(DATA_DIR, 'RegrasAEeGA_npv-wei.csv')

# Load data
runs516 = pd.read_csv(runs516_path)
ga_data = pd.read_csv(ga_path)

wells = ['PRK014', 'PRK028', 'PRK045', 'PRK052', 'PRK060', 'PRK061', 'PRK083', 'PRK084', 'PRK085']

# Configure style
plt.rcParams["figure.figsize"] = (9, 7)
fig = plt.figure()
gs = fig.add_gridspec(3, 3, wspace=0.08, hspace=0.45)
axs = gs.subplots(sharey=True)

for well in wells:
    # Get row and column in 3x3 grid
    line = wells.index(well) // 3
    column = wells.index(well) % 3

    ax = axs[line, column]

    # Filter Strategy A data for this well (TVALUE == 1 and MAX == 1)
    filtered_data = runs516[(runs516[well + '_TVALUE'] == 1) & (runs516[well + '_MAX'] == 1)]

    # Strategy A data (new execution - red)
    strategy_a_weif_values = filtered_data[well].values

    # GA data (blue)
    ga_weif_values = ga_data['WEIF[' + well + ']'].values

    # Create DataFrame for histogram
    strategy_a_df = pd.DataFrame({
        'WEIF': strategy_a_weif_values,
        'Execution': 'Proposed method'
    })

    ga_df = pd.DataFrame({
        'WEIF': ga_weif_values,
        'Execution': 'GA'
    })

    merged = pd.concat([ga_df, strategy_a_df]).reset_index(drop=True)

    # Create histogram
    palette = ["#457b9d", "#e63946"]  # Blue and red

    plot = sns.histplot(
        merged, ax=ax, x='WEIF', hue="Execution", stat='percent', fill=True,
        common_norm=False, palette=palette
    )

    # Apply hatch to red part (Strategy A)
    if len(plot.containers) > 1:
        for patch in plot.containers[1].patches:
            patch.set_hatch('//')

    # Format X axis
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.2f}'.format(x / 1e+9)))

    # Configure title
    ax.set_title(well, y=1.0)
    plot.set(xlabel='', ylabel='')

    # Log scale on Y axis
    plot.set(yscale="log")

    # Configure Y axis ticks
    ax.set_yticks([0.1, 1, 10, 50, 100])
    ax.set_yticklabels(['0.1%', '1%', '10%', '50%', '100%'])

    # Remove minor ticks to avoid unwanted small lines
    ax.tick_params(axis='y', which='minor', left=False)
    ax.set_yticks([0.1, 1, 10, 50, 100], minor=False)

    # Remove individual legend
    plot.legend(handles=[], frameon=False)

# Main title
fig.suptitle(
    "WEIF distribution comparison: Proposed method vs GA (Genetic Algorithm)",
    fontsize=14
)

# Labels
fig.text(
    0.5, 0.05,
    'WEIF obtained in the simulation (billion USD)',
    ha='center', fontsize=13
)
fig.text(
    0.01, 0.5,
    'WEIF percentage (%)',
    va='center', rotation='vertical', fontsize=13
)

# Legenda global
from matplotlib import patches
fig.legend(
    title="",
    handles=[
        patches.Patch(
            facecolor="#457b9d", alpha=0.7,
            label='GA (Genetic Algorithm)',
            linewidth=1, edgecolor='black'
        ),
        patches.Patch(
            facecolor="#e63946", alpha=0.7,
            label='Proposed method',
            linewidth=1, edgecolor='black', hatch='//'
        ),
    ],
    ncol=2,
    bbox_to_anchor=(0.5, 0),
    loc='lower center',
    borderaxespad=0.,
    frameon=False,
)

plt.subplots_adjust(top=0.89, bottom=0.13, left=0.1, right=0.98, hspace=0.38, wspace=0.15)

# Save figure in project's outputs folder
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file_png = os.path.join(OUTPUT_DIR, 'strategy_a_vs_ga_comparison.png')
output_file_pdf = os.path.join(OUTPUT_DIR, 'strategy_a_vs_ga_comparison.pdf')
plt.savefig(output_file_png, dpi=300)
plt.savefig(output_file_pdf, dpi=300)
print(f"Chart saved: {output_file_png}")
print(f"Chart saved: {output_file_pdf}")

# Show statistics
print("\nData statistics:")
print("=" * 60)
for well in wells:
    filtered_data = runs516[(runs516[well + '_TVALUE'] == 1) & (runs516[well + '_MAX'] == 1)]
    strategy_a_data = filtered_data[well]
    ga_data_well = ga_data['WEIF[' + well + ']']

    print(f"\n{well}:")
    print(f"  Strategy A - Count: {len(strategy_a_data)}")
    if len(strategy_a_data) > 0:
        print(f"    Minimum: {strategy_a_data.min():.2e}")
        print(f"    Maximum: {strategy_a_data.max():.2e}")
        print(f"    Mean: {strategy_a_data.mean():.2e}")
        print(f"    Median: {strategy_a_data.median():.2e}")

    print(f"  GA - Count: {len(ga_data_well)}")
    if len(ga_data_well) > 0:
        print(f"    Minimum: {ga_data_well.min():.2e}")
        print(f"    Maximum: {ga_data_well.max():.2e}")
        print(f"    Mean: {ga_data_well.mean():.2e}")
        print(f"    Median: {ga_data_well.median():.2e}")

plt.clf()
