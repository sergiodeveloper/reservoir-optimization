import os
import numpy as np
import pandas as pd
from typing import List

from ideal_rules import RULES

# ITERATION: 8
# MODEL: run0756
# TEMPLATE: U2DBO_IM80_E0_WAG_IW.tpl
# NPVM: 2480466116
PIVOT_RULES = {
  "PRK014": "0	0	0	2100	3700	300	500	2100	2500	3500	0	0	0	0	1300",
  "PRK028": "300	0	0	0	900	0	0	0	700	2300	700	2100	0	0	3300",
  "PRK045": "300	0	0	0	900	2100	2900	0	0	3100	500	0	0	2300	3700",
  "PRK052": "0	0	0	700	900	0	0	0	500	1700	1500	2100	0	0	2900",
  "PRK060": "0	1500	0	2300	3500	500	0	0	0	2300	0	0	2500	0	3500",
  "PRK061": "0	0	0	300	700	0	0	0	0	300	300	0	500	3100	3700",
  "PRK083": "0	0	0	0	500	0	0	0	0	500	700	3100	0	0	3500",
  "PRK084": "0	300	2100	2300	3700	500	0	0	0	1100	700	0	0	900	1500",
  "PRK085": "0	0	700	2100	3300	900	0	0	1100	1500	0	0	0	1700	3500"
}

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

def adjust_rule_repeatedly(well: str, input_rule: List[int], target_rule: List[str], max_iterations: int = 15):
  current_rule = input_rule.copy()
  rule_evolution = []

  for iteration in range(max_iterations):
    target_rule_int = [int(x) for x in target_rule]
    differences = [abs(current_rule[i] - target_rule_int[i]) for i in range(len(current_rule))]
    total_diff = sum(differences)

    rule_evolution.append(current_rule.copy())

    if current_rule == target_rule_int:
      break

    adjusted_rule = adjust_rule(current_rule, target_rule)

    if adjusted_rule == current_rule:
      break

    current_rule = adjusted_rule

  return rule_evolution

def main():
  dataset = pd.read_csv('./simplified_dataset.csv')

  wells = ['PRK014', 'PRK028', 'PRK045', 'PRK052', 'PRK060', 'PRK061', 'PRK083', 'PRK084', 'PRK085']

  output_dir = './01.2_output_repeated_adjustment'
  os.makedirs(output_dir, exist_ok=True)

  # Create all possible column names
  all_columns = []
  zones = ["Z1", "Z2", "Z3"]
  stages = ["S1", "S2", "S3", "S4", "S5"]

  for well in wells:
    for zone in zones:
      for stage in stages:
        all_columns.append(f'CLOSE_GOR_{well}_{zone}_{stage}')

  all_evolutions = []

  for well in wells:
    if well in RULES['simplified_dataset_rules']:
      target_rule = RULES['simplified_dataset_rules'][well]
    else:
      continue

    gor_columns = [col for col in dataset.columns if col.startswith(f'CLOSE_GOR_{well}_')]
    gor_columns.sort()

    # Parse PIVOT_RULES for this well
    pivot_rule_str = PIVOT_RULES[well]
    pivot_rule = [int(x) for x in pivot_rule_str.split()]

    for sample_idx in range(len(dataset)):
      sample_num = sample_idx + 1

      starting_rule = dataset.iloc[sample_idx][gor_columns].tolist()

      rule_evolution = adjust_rule_repeatedly(
        well=well,
        input_rule=starting_rule,
        target_rule=target_rule,
        max_iterations=15
      )

      for i, rule in enumerate(rule_evolution):
        evolution_entry = {
          'SAMPLE': sample_num,
          'ITERATION': i + 1,
          'WELL': well,
          'converged': (i == len(rule_evolution) - 1) and (rule == [int(x) for x in target_rule])
        }

        # Initialize all columns with PIVOT_RULES values
        for col in all_columns:
          well_name = col.split('_')[2]  # Extract well name from column
          zone_stage = '_'.join(col.split('_')[3:])  # Extract zone_stage part

          if well_name == well:
            # For the current well, use the evolution values
            zone = zone_stage.split('_')[0]
            stage = zone_stage.split('_')[1]
            zone_idx = zones.index(zone)
            stage_idx = stages.index(stage)
            rule_idx = zone_idx * len(stages) + stage_idx
            evolution_entry[col] = rule[rule_idx]
          else:
            # For other wells, use PIVOT_RULES values
            pivot_well_rule = [int(x) for x in PIVOT_RULES[well_name].split()]
            zone = zone_stage.split('_')[0]
            stage = zone_stage.split('_')[1]
            zone_idx = zones.index(zone)
            stage_idx = stages.index(stage)
            rule_idx = zone_idx * len(stages) + stage_idx
            evolution_entry[col] = pivot_well_rule[rule_idx]

        all_evolutions.append(evolution_entry)

    results_df = pd.DataFrame(all_evolutions)
  results_df.to_csv(f'{output_dir}/all_wells_evolution.csv', index=False)


if __name__ == "__main__":
  main()
