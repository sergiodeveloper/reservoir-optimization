
import os
from typing import List
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from  matplotlib.ticker import FuncFormatter
from scipy.interpolate import make_interp_spline

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from target_rules import RULES

proxy_files_path = '04_output_best_proxy'
output_image_filename = '06_output_adjust_closest_factor.png'
output_variation_image_filename = '06_output_adjust_closest_factor_variation.png'

def adjust_rule(current_rule: List[int], target_rule: List[str]):
  if len(target_rule) != len(current_rule):
    raise Exception('Rule sizes must match')

  adjusted_rule = current_rule.copy()

  factors_diff = np.absolute(np.array(current_rule) - np.array(target_rule))

  if len(factors_diff[factors_diff > 0]) == 0: # If all factors have converged
    return adjusted_rule

  closest_factor = factors_diff[factors_diff > 0].min()

  closest_factor_indices = np.where(factors_diff == closest_factor)[0]

  for factor_idx in closest_factor_indices:
    adjusted_rule[factor_idx] = target_rule[factor_idx]

  return adjusted_rule

def adjust_rule_verbose(well: str, input_rule: List[int], target_rule: List[str]):
  print('Target rule from well ' + well + ':')
  print(target_rule)

  print('Input rule:')
  print(input_rule)

  print('Adjusted rule:')
  adjusted = adjust_rule(input_rule, target_rule)
  print(adjusted)
  print()

  # # ADJUSTING RULE REPEATEDLY
  # adjusted = adjust_rule(adjusted, target_rule)
  # print(adjusted)
  # adjusted = adjust_rule(adjusted, target_rule)
  # print(adjusted)
  # adjusted = adjust_rule(adjusted, target_rule)
  # print(adjusted)
  # adjusted = adjust_rule(adjusted, target_rule)
  # print(adjusted)
  # adjusted = adjust_rule(adjusted, target_rule)
  # print(adjusted)
  # adjusted = adjust_rule(adjusted, target_rule)
  # print(adjusted)
  # adjusted = adjust_rule(adjusted, target_rule)
  # print(adjusted)
  # adjusted = adjust_rule(adjusted, target_rule)
  # print(adjusted)
  # adjusted = adjust_rule(adjusted, target_rule)
  # print(adjusted)
  # adjusted = adjust_rule(adjusted, target_rule)
  # print(adjusted)
  # adjusted = adjust_rule(adjusted, target_rule)
  # print(adjusted)
  # adjusted = adjust_rule(adjusted, target_rule)
  # print(adjusted)
  # adjusted = adjust_rule(adjusted, target_rule)
  # print(adjusted)

  return adjusted

print('Testing rule adjuster...')

adjust_rule_verbose(
  well='PRK014',
  input_rule=[1900,  1900,  2100,  2100,  3700,    1300,  1300,  1300,  1300,  1300,    1700,  1700,  1700,  1700,  1700],
  target_rule=RULES['simplified_dataset_rules']['PRK014'],
)

testing_adjuster_prk028 = adjust_rule_verbose(
  well='PRK028',
  input_rule=[1900,  1900,  2100,  2100,  3700,    1300,  1300,  1300,  1300,  1300,    1700,  1700,  1700,  1700,  1700],
  target_rule=RULES['simplified_dataset_rules']['PRK028'],
)
adjust_rule_verbose(
  well='PRK028',
  input_rule=testing_adjuster_prk028,
  target_rule=RULES['simplified_dataset_rules']['PRK028'],
)


wells = ['PRK014', 'PRK028', 'PRK045', 'PRK052', 'PRK060', 'PRK061', 'PRK083', 'PRK084', 'PRK085']
possible_rule_values = [
  0, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500, 2700, 2900, 3100, 3300, 3500, 3700
]

# # ADJUSTING A RANDOM STARTING RULE
# print('RANDOM STARTING RULE::')
# adjust_rule_verbose(
#   well='PRK014',
#   input_rule=[random.choice(possible_rule_values) for _ in range(15)],
#   target_rule=RULES['simplified_dataset_rules']['PRK014'],
# )

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
  proxy_files_path, target_rule_per_well, should_simplify_rule: bool,
  title: str, subtitle: str, image_filename: str, percentage_variation_title: str,
  percentage_variation_image_filename: str,
):
  evolution_per_well = {} # Format: { 'PRK014': [[1.345, 1.445, ...], [2.31, 1.75, ...]] }

  for well in wells:
    random.seed(0)
    print(well)

    if not os.path.exists(proxy_files_path + '/' + well):
      print('Well ' + well + ' not found. Skipping...')
      continue

    evolution_per_well[well] = []

    with open(proxy_files_path + '/' + well + '/proxy.pkl', 'rb') as file:
      regression_model = pickle.load(file)

    with open(proxy_files_path + '/' + well + '/y_scaler.pkl', 'rb') as file:
      y_scaler = pickle.load(file)

    target_rule = target_rule_per_well[well]

    for iteration in range(0, 100):
      current_rule = [random.choice(possible_rule_values) for _ in range(15)]

      if should_simplify_rule:
        # Simplify current random rule
        V = current_rule
        for z in [0, 5, 10]: # Zone index
          if V[z] > V[z+1] or V[z] > V[z+2] or V[z] > V[z+3] or V[z] > V[z+4]:
            V[z] = 0
          if V[z+1] > V[z+2] or V[z+1] > V[z+3] or V[z+1] > V[z+4]:
            V[z+1] = 0
          if V[z+2] > V[z+3] or V[z+2] > V[z+4]:
            V[z+2] = 0
          if V[z+3] > V[z+4]:
            V[z+3] = 0

      scaled_prediction = regression_model.predict(np.array(current_rule).reshape(1, -1)).reshape(-1, 1)
      prediction = y_scaler.inverse_transform(scaled_prediction)[0][0]
      # print('Current rule: ', current_rule, 'Predicted WEIF:', prediction)

      weif_evolution = [prediction]

      while current_rule != target_rule:
        current_rule = adjust_rule(current_rule, target_rule)
        scaled_prediction = regression_model.predict(np.array(current_rule).reshape(1, -1)).reshape(-1, 1)
        prediction = y_scaler.inverse_transform(scaled_prediction)[0][0]
        # print('Current rule: ', current_rule, 'Predicted WEIF:', prediction)
        weif_evolution.append(prediction)

      evolution_per_well[well].append(weif_evolution)

  # Prepare plot

  rows = [] # Format: [well, iteration, weif]
  for well in wells:
    if well not in evolution_per_well:
      continue
    for weif_evolution in evolution_per_well[well]:
      for iteration, weif in enumerate(weif_evolution):
        rows.append([well, iteration, weif])

  evolution_history = pd.DataFrame(rows, columns=['well', 'iteration', 'weif'])

  # WEIF variation figure

  plt.rcParams["figure.figsize"] = (8,6)
  ax = sns.lineplot(data=evolution_history, x="iteration", y="weif", hue="well", style="well", ci='sd')

  ax.yaxis.set_major_locator(ticker.MultipleLocator(1e+8))
  ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

  plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0f} mi'.format(x / 1e+6))) # format Y ticks
  plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x))) # Show X ticks as integers
  ax.set(xticks=evolution_history['iteration'].unique()) # Show all X ticks

  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0) # Legend to the right
  plt.suptitle(title, x=0.5)
  plt.title(subtitle, fontsize=10, x=0.55)
  ax.set(xlabel="Itera????o", ylabel='WEIF previsto pela Proxy (US$)')
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
  fig.text(0.5, 0.04, 'Itera????o', ha='center')
  fig.text(0.04, 0.5, 'Varia????o do WEIF m??dio previsto', va='center', rotation='vertical')

  ylimit_per_well = []
  for well in wells:
    if well not in evolution_per_well:
      continue
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

      if well not in evolution_per_well:
        continue

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
print('Testing rule adjustment against regressor models...')

plot_rule_adjustment_in_all_wells(
  proxy_files_path=proxy_files_path,
  target_rule_per_well=RULES['simplified_dataset_rules'],
  should_simplify_rule=True,
  title="Evolu????o do WEIF ajustando apenas os valores mais pr??ximos da regra alvo por itera????o",
  subtitle="Usando regras e fun????es proxy geradas pelo conjunto de dados simplificado",
  image_filename=output_image_filename,
  percentage_variation_title='Varia????o do WEIF ajustando o valor mais pr??ximo da regra com dados simplificados',
  percentage_variation_image_filename=output_variation_image_filename,
)
