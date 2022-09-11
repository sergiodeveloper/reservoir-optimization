
import pandas as pd

dataset_path = 'sample_original_dataset.csv'
output_path = 'simplified_dataset.csv'

original_dataset = pd.read_csv(dataset_path)

simplified_dataset = pd.DataFrame()

# simplified_dataset = pd.concat([original_dataset[['ITERATION', 'MODEL']]], axis=1)

for well in ['PRK014', 'PRK028', 'PRK045', 'PRK052', 'PRK060', 'PRK061', 'PRK083', 'PRK084', 'PRK085']:
  weif_column = 'WEIF[' + well + ']'

  for zone in ['Z1', 'Z2', 'Z3']:
    columns = [('CLOSE_GOR_' + well + '_' + zone + '_' + stage) for stage in ['S1', 'S2', 'S3', 'S4', 'S5']]

    zone_features = original_dataset[columns].copy()

    # Simplify rules
    for V in zone_features.iloc:
      if V[0] > V[1] or V[0] > V[2] or V[0] > V[3] or V[0] > V[4]:
        V[0] = 0
      if V[1] > V[2] or V[1] > V[3] or V[1] > V[4]:
        V[1] = 0
      if V[2] > V[3] or V[2] > V[4]:
        V[2] = 0
      if V[3] > V[4]:
        V[3] = 0

    simplified_dataset = pd.concat([simplified_dataset, zone_features], axis=1)

  simplified_dataset = pd.concat([simplified_dataset, original_dataset[weif_column]], axis=1)


# simplified_dataset = pd.concat([simplified_dataset, original_dataset['NPVM']], axis=1)
print(simplified_dataset)

simplified_dataset.to_csv(output_path, index=False)





