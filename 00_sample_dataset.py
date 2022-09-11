
import pandas as pd
import numpy as np

sample_size = 500

sample_dataset = pd.DataFrame()

for i in range(sample_size):
  row = pd.DataFrame()

  for well in ['PRK014', 'PRK028', 'PRK045', 'PRK052', 'PRK060', 'PRK061', 'PRK083', 'PRK084', 'PRK085']:
    for zone in ['Z1', 'Z2', 'Z3']:
      columns = [('CLOSE_GOR_' + well + '_' + zone + '_' + stage) for stage in ['S1', 'S2', 'S3', 'S4', 'S5']]
      gor_values = pd.DataFrame(np.random.randint(100, 3700, size=(1, len(columns))), columns=columns)
      row = pd.concat([row, gor_values], axis=1)

    weif_column = 'WEIF[' + well + ']'
    weif_value = pd.DataFrame(np.random.randint(10**8, 10**10, size=(1, 1)), columns=[weif_column])
    row = pd.concat([row, weif_value], axis=1)

  sample_dataset = pd.concat([sample_dataset, row], axis=0)

sample_dataset.to_csv('sample_original_dataset.csv', index=False)
