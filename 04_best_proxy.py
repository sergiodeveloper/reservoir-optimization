
import re
import os
import shutil

wells = ['PRK014', 'PRK028', 'PRK045', 'PRK052', 'PRK060', 'PRK061', 'PRK083', 'PRK084', 'PRK085']

input_directory = '02_output_proxy_models/'
output_directory = '04_output_best_proxy/'

# Choose best GTB model

for well in wells:
  best_rmse = None
  best_trial = None

  for trial in [1,2,3,4,5]:
    trial_directory = input_directory + well + '/GTB/TRIAL_' + str(trial)

    if not os.path.exists(trial_directory):
      continue

    with open(trial_directory + '/output.txt') as file:
      lines = file.readlines()

    for line in lines:
      if 'Best RMSE' in line:
        rmse = float(re.search('[0-9.]+', line).group(0))
        if best_trial is None or rmse < best_rmse:
          best_rmse = rmse
          best_trial = trial

  if best_trial is None:
    continue

  trial_directory = input_directory + well + '/GTB/TRIAL_' + str(best_trial)

  print('Best RMSE for well ' + well + ': ' + str(best_rmse) + ' from trial ' + str(best_trial))

  proxy_final_directory = output_directory + well
  os.makedirs(proxy_final_directory, exist_ok=True)
  shutil.copyfile(trial_directory + '/proxy.pkl', proxy_final_directory + '/proxy.pkl')
  shutil.copyfile(trial_directory + '/y_scaler.pkl', proxy_final_directory + '/y_scaler.pkl')

