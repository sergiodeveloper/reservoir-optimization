
from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from  matplotlib.ticker import FuncFormatter
from scipy.interpolate import make_interp_spline

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_simulated_data(well_name: str) -> pd.DataFrame:
    """
    Load simulated data for a specific well from CSV files.
    Returns a DataFrame with iterations and corresponding WEIF values.
    """
    csv_file = f"simulated_rule_adjustments/results_Rules-G2_{well_name}-evolution.csv"
    df = pd.read_csv(csv_file)

    # Extract the WEIF column for the specific well
    weif_column = f"WEIF[{well_name}]"

    # Group by SAMPLE and get the evolution for each sample
    # Each sample represents one optimization run (like the iterations in the original code)
    result_data = []

    for sample in df['SAMPLE'].unique():
        sample_data = df[df['SAMPLE'] == sample].sort_values('ITERATION')
        weif_values = sample_data[weif_column].values

        # Create iteration-indexed data for this sample
        for iteration, weif in enumerate(weif_values):
            result_data.append({
                'sample': sample,
                'iteration': iteration,
                'weif': weif
            })

    return pd.DataFrame(result_data)


wells = ['PRK014', 'PRK028', 'PRK045', 'PRK052', 'PRK060', 'PRK061', 'PRK083', 'PRK084', 'PRK085']

def plot_weif_percentage_variation(
  ax: plt.Axes, weif_evolution, y_max: float, well_name: str, bottom_line=False,
):
  x = weif_evolution["iteration"]
  y = weif_evolution["weif"].pct_change()
  y[0] = 0

  X_Y_Spline = make_interp_spline(x, y)
  X_ = np.linspace(x.min(), x.max(), len(x) * 20)
  Y_ = X_Y_Spline(X_)

  ax.plot(X_, Y_, color="black", alpha=0.5)
  ax.spines['bottom'].set_position('center')
  ax.xaxis.set_ticks_position('bottom')
  ax.xaxis.set_ticks(np.arange(0, x.max(), 2))
  ax.fill_between(X_, (Y_ > 0) * Y_, step="pre", alpha=1, color="#b4e3af")
  ax.fill_between(X_, (Y_ < 0) * Y_, step="pre", alpha=1, color="#e3afaf")
  ax.set_ylim(ymin=-y_max, ymax=y_max)

  if bottom_line:
    ax.axhline(y=-y_max, color='black')

  ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y).replace('.', ',')))
  ax.set_title(well_name, y=0.85)

  nbins = 6 # len(ax1.get_xticklabels())
  ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins, prune='upper'))


def plot_rule_adjustment_in_all_wells(
  title: str, subtitle: str, image_filename: str, percentage_variation_title: str,
  percentage_variation_image_filename: str,
):
  evolution_per_well = {} # Format: { 'PRK014': [[1.345, 1.445, ...], [2.31, 1.75, ...]] }

  for well in wells:
    print(f"Loading simulated data for {well}")

    # Load the actual simulated data for this well
    simulated_data = load_simulated_data(well)

    evolution_per_well[well] = []

    # Group by sample (each sample represents one optimization run)
    for sample in simulated_data['sample'].unique():
      sample_data = simulated_data[simulated_data['sample'] == sample]
      # Extract WEIF values for this optimization run
      weif_evolution = sample_data['weif'].tolist()
      evolution_per_well[well].append(weif_evolution)

  # Prepare plot

  rows = [] # Format: [well, iteration, weif]
  for well in wells:
    for weif_evolution in evolution_per_well[well]:
      for iteration, weif in enumerate(weif_evolution):
        rows.append([well, iteration, weif])

  evolution_history = pd.DataFrame(rows, columns=['well', 'iteration', 'weif'])

  # WEIF variation figure

  plt.rcParams["figure.figsize"] = (8,6)
  ax = sns.lineplot(data=evolution_history, x="iteration", y="weif", hue="well", style="well", errorbar='sd')

  ax.yaxis.set_major_locator(ticker.MultipleLocator(1e+8))
  ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

  plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0f} mi'.format(x / 1e+6))) # format Y ticks
  plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x))) # Show X ticks as integers
  ax.set(xticks=evolution_history['iteration'].unique()) # Show all X ticks

  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0) # Legend to the right
  plt.suptitle(title, x=0.5)
  plt.title(subtitle, fontsize=10, x=0.55)
  # ax.set(xlabel="Iteração", ylabel='WEIF previsto pela Proxy (US$)')
  ax.set(xlabel="Iteration", ylabel='Simulated WEIF (USD)')
  plt.tight_layout()

  # plt.show()
  plt.savefig(image_filename)
  plt.clf()

  # Percentage variation figure

  plt.rcParams["figure.figsize"] = (8,6)
  fig = plt.figure()
  gs = fig.add_gridspec(3, 3, hspace=0, wspace=0)
  axs = gs.subplots(sharex=True, sharey=True)
  fig.suptitle(percentage_variation_title)
  # Hide some y labels
  axs[0, 1].get_yaxis().set_visible(False)
  axs[1, 1].get_yaxis().set_visible(False)
  axs[2, 1].get_yaxis().set_visible(False)
  # Labels
  fig.text(
    0.5, 0.04,
    # 'Iteração',
    'Iteration',
    ha='center'
  )
  fig.text(
    0.04, 0.5,
    # 'Variação do WEIF médio previsto',
    'Simulated WEIF mean variation',
    va='center', rotation='vertical'
  )

  ylimit_per_well = []
  for well in wells:
    print(well)
    print(evolution_history[evolution_history['well'] == well] \
          .groupby('iteration', as_index=False)['weif'].mean()['weif'].pct_change())
    ylimit_per_well.append(
      np.absolute(
        evolution_history[evolution_history['well'] == well] \
          .groupby('iteration', as_index=False)['weif'].mean()['weif'].pct_change()
      ).max()
    )

  print(ylimit_per_well)
  print(np.array(ylimit_per_well).max())

  for line in range(3):
    for col in range(3):
      well = wells[line * 3 + col]

      plot_weif_percentage_variation(
        ax=axs[line, col],
        y_max=np.array(ylimit_per_well).max() * 1.2,
        weif_evolution=evolution_history[evolution_history['well'] == well] \
          .groupby('iteration', as_index=False)['weif'].mean(),
        well_name=well,
        bottom_line=line == 2,
      )

  plt.tight_layout(rect=[0.05, 0.05, 1, 0.99])

  # plt.show()
  plt.savefig(percentage_variation_image_filename)
  plt.clf()


print()
print('Plotting WEIF evolution using real simulated data...')

plot_rule_adjustment_in_all_wells(
  title="WEIF evolution during rule optimization process",
  subtitle="Using actual simulated data from reservoir optimization runs",
  image_filename="WEIF_evolution_simulated_data.pdf",
  percentage_variation_title='WEIF percentage variation during optimization with simulated data',
  percentage_variation_image_filename="WEIF_percentage_evolution_simulated_data.pdf",
)
