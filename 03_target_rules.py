
import os
import decimal
import itertools
from typing import Any, List
import numpy as np
import pandas as pd

import graphviz
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

dataset_path = './simplified_dataset.csv'
output_directory = './03_output_target_rules'
target_rules_output_file = './target_rules.py'

translations = {
  'Variable': 'Variável',
  'Coefficient': 'Coeficiente',
  'T-Value': 'T-Value',
}

def df_as_latex_table(df: pd.DataFrame):
  df = df.copy()
  df['Variable'] = [('Coeficiente linear' if cell == 'Intercept' else cell) for cell in df['Variable']]

  labels = [translations[col] for col in df.columns]
  values = df.values

  text = '\\begin{tabular}{ccc}\n\\toprule\n'
  for i, label in enumerate(labels):
    text = text + label
    text = text + (' & ' if i < len(labels) - 1 else ' \\\\\n')

  text = text + '\\midrule\n'

  for row in values:
    for i, cell in enumerate(row):
      text = text + str(cell).replace('.', ',').replace('_', '\\_')
      text = text + (' & ' if i < len(row) - 1 else ' \\\\\n')

  text = text + '\\bottomrule\n\\end{tabular}\n'

  return text


def scientific_notation(x):
  return '{:.10e}'.format(x)

def format_number(num):
  try:
    dec = decimal.Decimal(num)
  except:
    return 'Invalid'
  tup = dec.as_tuple()
  delta = len(tup.digits) + tup.exponent
  digits = ''.join(str(d) for d in tup.digits)
  if delta <= 0:
    zeros = abs(tup.exponent) - len(tup.digits)
    val = '0.' + ('0'*zeros) + digits
  else:
    val = digits[:delta] + ('0'*tup.exponent) + '.' + digits[delta:]
  val = val.rstrip('0')
  if val[-1] == '.':
    val = val[:-1]
  if tup.sign:
    return '-' + val
  return val



def tree_to_dictionary(tree: DecisionTreeRegressor, feature_names: List[str], parent_reference=True):
  tree_ = tree.tree_
  feature_name = [
      feature_names[i] if i != sklearn.tree._tree.TREE_UNDEFINED else "undefined!"
      for i in tree_.feature
  ]

  def recurse(node_id: int, parent_side: str, parent_node):
    branch = {
      "node_id": node_id,
      "samples": tree_.n_node_samples[node_id],
      "prediction": float(tree_.value[node_id][0][0]),
      "variable": None,
      "threshold": None,
      "parent_side": parent_side,
      "parent": None if not parent_reference else parent_node,
      "left": None,
      "right": None,
    }

    if tree_.feature[node_id] != sklearn.tree._tree.TREE_UNDEFINED: # Has children
      branch['variable'] = feature_name[node_id]
      branch['threshold'] = tree_.threshold[node_id]
      branch['left'] = recurse(
        node_id=tree_.children_left[node_id],
        parent_side='left',
        parent_node=branch,
      )
      branch['right'] = recurse(
        node_id=tree_.children_right[node_id],
        parent_side='right',
        parent_node=branch,
      )
    return branch

  return recurse(node_id=0, parent_node=None, parent_side=None)


def get_branch_with_best_prediction(tree_root, minimum_samples=45):
  def recurse(branch) -> List[Any]:
    leaves = []

    if branch['left'] is not None and branch['right'] is not None:
      leaves += recurse(branch=branch['left'])
      leaves += recurse(branch=branch['right'])
    else:
      leaves.append(branch)
    return leaves

  leaves = recurse(tree_root)

  best_leaf = None

  for leaf in leaves:
    if leaf['samples'] < minimum_samples:
      continue

    if best_leaf is None or leaf['prediction'] > best_leaf['prediction'] \
      or (leaf['prediction'] == best_leaf['prediction'] and leaf['samples'] > best_leaf['samples']):
      best_leaf = leaf

  if best_leaf is None:
    return None

  # Build conditions in the format ["Z1_S1 > 300", "Z2_S4 <= 200"]
  branch_conditions = []

  current_node = best_leaf
  while current_node is not None:
    parent = current_node['parent']
    if parent is None:
      break

    condition = parent['variable']
    condition += ' <= ' if current_node['parent_side'] == 'left' else ' > '
    condition += format_number(parent['threshold'])

    branch_conditions = [condition] + branch_conditions
    current_node = current_node['parent']

  return ' & '.join(branch_conditions)


def tree_to_graphviz(tree_root):
  def recurse(branch):
    code = ''

    if branch['left'] is not None and branch['right'] is not None:
      code += str(branch['node_id']) + ' [label=<'
      code +=  '<FONT POINT-SIZE="8">Amostras:</FONT><BR />' + str(branch['samples'])
      code += '>] ;\n'

      code += recurse(branch=branch['left'])
      code += str(branch['node_id']) + ' -> ' + str(branch['left']['node_id'])
      code += ' [label="' + branch['variable'] + ' ≤ ' + format_number(branch['threshold']) + '"] ;\n'

      code += recurse(branch=branch['right'])
      code += str(branch['node_id']) + ' -> ' + str(branch['right']['node_id'])
      code += ' [label="' + branch['variable'] + ' > ' + format_number(branch['threshold']) + '"] ;\n'
    else:
      code += str(branch['node_id']) + ' [label=<'
      code +=  '<FONT POINT-SIZE="8">Amostras:</FONT><BR />' + str(branch['samples'])
      code += '<BR /><FONT POINT-SIZE="8">Valor:</FONT><BR />'
      code += str(round(branch['prediction'], 3)).replace('.', ',')
      code += '>] ;\n'
    return code

  code = '''digraph Tree {
node [shape=box, style="rounded", color="black", fontname="helvetica"] ;
edge [fontname="helvetica", fontsize=8] ;
'''
  code += recurse(branch=tree_root)
  code += '}\n'

  return code

def calculateTvalue(
  linear_regression: LinearRegression, X: np.ndarray, y: np.ndarray, feature_names: List[str]
):
  predictions = linear_regression.predict(X)
  # X with a column of ones
  newX = pd.DataFrame({ "Constant": np.ones(len(X)) }).join(pd.DataFrame(X))

  MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))
  tvalues = np.append(linear_regression.intercept_, linear_regression.coef_) \
    / np.sqrt(MSE * np.linalg.inv(np.dot(newX.T, newX)).diagonal())
  stats = pd.DataFrame()

  for index, tvalue in enumerate(tvalues):
    if index == 0: continue
    stats = pd.concat([stats, pd.DataFrame({
      'Variable': feature_names[index - 1],
      'Coefficient': scientific_notation(linear_regression.coef_[0][index - 1]),
      'T-Value': scientific_notation(tvalue),
    }, index=[0])], ignore_index=True)

  stats = pd.concat([stats, pd.DataFrame({
    'Variable': 'Intercept',
    'Coefficient': scientific_notation(linear_regression.intercept_[0]),
    'T-Value': scientific_notation(tvalues[0]),
  }, index=[0])], ignore_index=True)
  return stats



factor_possible_values = [
  0, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500, 2700, 2900, 3100, 3300, 3500, 3700
]

minimum_factor_value = 0
maximum_factor_value = 3700

factor_names = [
  'Z1_S1', 'Z1_S2', 'Z1_S3', 'Z1_S4', 'Z1_S5',
  'Z2_S1', 'Z2_S2', 'Z2_S3', 'Z2_S4', 'Z2_S5',
  'Z3_S1', 'Z3_S2', 'Z3_S3', 'Z3_S4', 'Z3_S5',
]

def previous_factor_value(value: int):
  combined_list = factor_possible_values.copy()
  combined_list.append(value)
  combined_list = list(set(combined_list))
  combined_list.sort()

  index = combined_list.index(value)
  if index - 1 >= 0:
    return combined_list[index - 1]
  print('Warning: cannot get previous factor from value', value)
  return value

def next_factor_value(value: int):
  combined_list = factor_possible_values.copy()
  combined_list.append(value)
  combined_list = list(set(combined_list))
  combined_list.sort()

  index = combined_list.index(value)
  if index + 1 <= len(combined_list) - 1:
    return combined_list[index + 1]
  print('Warning: cannot get next factor from value', value)
  return value


def values_for_branch(branch: str):
  branch_conditions = [a.strip().split() for a in branch.split('&')] # Format: [['Z1_S5', '>', '1200'], ...]

  tree_values = {} # Format: { 'Z1_S5': 1400, ... }

  for condition in branch_conditions:
    variable_name = condition[0]
    comparator = condition[1]
    condition_value = int(condition[2].split('.')[0])

    recommended_value = condition_value

    if comparator == '>':
      recommended_value = next_factor_value(condition_value)
    if comparator == '>=' and recommended_value not in factor_possible_values:
      recommended_value = next_factor_value(condition_value)
    if comparator == '<':
      recommended_value = previous_factor_value(condition_value)
    if comparator == '<=' and recommended_value not in factor_possible_values:
      recommended_value = previous_factor_value(condition_value)


    if not variable_name in tree_values:
      tree_values[variable_name] = recommended_value
    else: # Variable was mentioned more than once in the branch, check if current value satisfies this condition
      current_value = tree_values[variable_name]

      if comparator == '>' and not current_value > condition_value:
        tree_values[variable_name] = recommended_value
      if comparator == '>=' and not current_value >= condition_value:
        tree_values[variable_name] = recommended_value
      if comparator == '<' and not current_value < condition_value:
        tree_values[variable_name] = recommended_value
      if comparator == '<=' and not current_value <= condition_value:
        tree_values[variable_name] = recommended_value

  return tree_values


def coefficientsWithGoodTvalue(coefficients):
  return [
    coefficient for coefficient in coefficients \
      if coefficient['tvalue'] >= 1.96 or coefficient['tvalue'] <= -1.96
  ]

def negativeCoefficientNames(coefficients):
  return [
    coefficient['variable_name'] for coefficient in coefficients \
      if coefficient['coefficient'] < 0
  ]

def build_rule(branch: str, linear_regression_info: pd.DataFrame, use_tvalue: bool, maximize: bool=True):
  rule = values_for_branch(branch)

  coefficients = [
    {
      'variable_name': line['Variable'],
      'coefficient': float(line['Coefficient']),
      'tvalue': float(line['T-Value']),
    } for index, line in linear_regression_info.iterrows() if 'Intercept' not in line['Variable']
  ]

  if use_tvalue:
    coefficients = coefficientsWithGoodTvalue(coefficients)

  negative_coefficients = negativeCoefficientNames(coefficients)

  for zone in [1, 2, 3]:
    # Triggers from this zone
    trigger_1 = 'Z' + str(zone) + '_S1'
    trigger_2 = 'Z' + str(zone) + '_S2'
    trigger_3 = 'Z' + str(zone) + '_S3'
    trigger_4 = 'Z' + str(zone) + '_S4'
    trigger_5 = 'Z' + str(zone) + '_S5'

    if trigger_5 not in rule: # Not defined by tree yet
      if trigger_5 in negative_coefficients:
        rule[trigger_5] = minimum_factor_value
      else:
        rule[trigger_5] = maximum_factor_value

    if trigger_4 not in rule: # Not defined by tree yet
      if trigger_4 not in negative_coefficients:
        triggers_in_front = [i for i in [rule[trigger_5]] if i != 0]
        rule[trigger_4] = maximum_factor_value if len(triggers_in_front) == 0 else triggers_in_front[0]
      else:
        rule[trigger_4] = minimum_factor_value

    if trigger_3 not in rule: # Not defined by tree yet
      if trigger_3 not in negative_coefficients:
        triggers_in_front = [i for i in [rule[trigger_4], rule[trigger_5]] if i != 0]
        rule[trigger_3] = maximum_factor_value if len(triggers_in_front) == 0 else triggers_in_front[0]
      else:
        rule[trigger_3] = minimum_factor_value

    if trigger_2 not in rule: # Not defined by tree yet
      if trigger_2 not in negative_coefficients:
        triggers_in_front = [
          i for i in [rule[trigger_3], rule[trigger_4], rule[trigger_5]] if i != 0
        ]
        rule[trigger_2] = maximum_factor_value if len(triggers_in_front) == 0 else triggers_in_front[0]
      else:
        rule[trigger_2] = minimum_factor_value

    if trigger_1 not in rule: # Not defined by tree yet
      if trigger_1 not in negative_coefficients:
        triggers_in_front = [
          i for i in [rule[trigger_2], rule[trigger_3], rule[trigger_4], rule[trigger_5]] if i != 0
        ]
        rule[trigger_1] = maximum_factor_value if len(triggers_in_front) == 0 else triggers_in_front[0]
      else:
        rule[trigger_1] = minimum_factor_value

  rule = dict(sorted(rule.items()))
  return rule



def process_well(well: str, original_dataset: pd.DataFrame, output_path: str):
  print(output_path, well, flush=True)
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  features = [('CLOSE_GOR_' + well + '_Z'+v[0] + '_S' + v[1]) for v in list(itertools.product(['1','2','3'],['1','2','3','4','5']))]
  weif_column = 'WEIF[' + well + ']'

  # Copy features (rules) and add WEIF at the end
  dataset = original_dataset[features].copy()
  dataset[weif_column] = original_dataset[weif_column].div(1e+8)

  dataset = dataset.to_numpy()

  X = dataset[:, :-1]
  y = dataset[:, -1].reshape(-1, 1)

  feature_names = [feature.replace('CLOSE_GOR_' + well + '_', '') for feature in features]

  linear_regression = LinearRegression().fit(X, y)
  linear_stats = calculateTvalue(linear_regression=linear_regression, X=X, y=y, feature_names=feature_names)

  print(df_as_latex_table(linear_stats))

  # Excluding root
  MAX_DEPTH_PER_WELL = {
    'PRK014': 5,
    'PRK028': 6,
    'PRK045': 3,
    'PRK052': 5,
    'PRK060': 4,
    'PRK061': 3,
    'PRK083': 3,
    'PRK084': 4,
    'PRK085': 3,
  }
  max_depth = MAX_DEPTH_PER_WELL[well]

  tree_regressor = DecisionTreeRegressor(random_state=0, max_depth=max_depth).fit(X, y)

  tree_root = tree_to_dictionary(tree=tree_regressor, feature_names=feature_names, parent_reference=True)

  image = graphviz.Source(tree_to_graphviz(tree_root))
  image.render(directory=output_path, filename=well + '_tree', cleanup=True)

  branch_with_best_prediction = get_branch_with_best_prediction(tree_root=tree_root, minimum_samples=45)

  rule = build_rule(
    branch=branch_with_best_prediction,
    linear_regression_info=linear_stats,
    use_tvalue=True,
  )

  return rule



def main():
  # Simplified dataset
  simplified_dataset_rules = {}
  for well in ['PRK014', 'PRK028', 'PRK045', 'PRK052', 'PRK060', 'PRK061', 'PRK083', 'PRK084', 'PRK085']:
    rule = process_well(
      well,
      original_dataset=pd.read_csv(dataset_path),
      output_path=output_directory,
    )
    simplified_dataset_rules[well] = list(rule.values())


  with open(target_rules_output_file, 'w') as file:
    string = str({
      'simplified_dataset_rules': simplified_dataset_rules,
    }).replace('{', '{\n  ').replace('},', '},\n  ').replace('],', '],\n')

    print('# AUTOMATICALLY GENERATED - DO NOT EDIT', file=file)
    print(file=file)
    print('RULES = ' + string, file=file)

main()
