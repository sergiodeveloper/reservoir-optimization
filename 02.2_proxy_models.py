
import os
import sys
import pickle
import datetime
import itertools
import numpy as np
import pandas as pd
import shutil
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

single_processor = False

folds = 5

dataset_path = './otm_gor_icv5_clean.csv'

output_directory = './02.2_output_proxy_models'
cross_validations = 10

regressors = {
  'GTB': {
    'function': GradientBoostingRegressor,
    'params': {},
    'search_params': {
      'min_samples_split': [0.05, 0.1, 0.2, 0.3],
      'n_estimators': [50, 100, 150],
      'learning_rate': [0.01, 0.1, 0.5],
      'loss': ['ls', 'lad', 'huber']
    }
  },
  # 'KRR': {
  #   'function': KernelRidge,
  #   'params': {},
  #   'search_params': [
  #     {'kernel': ['poly'], 'degree': [2,3,4], 'alpha': [1e0, 0.1, 1e-2, 1e-3]},
  #     {'kernel': ['rbf'], 'gamma':  np.logspace(-3, 3, 7), 'alpha': [1e0, 0.1, 1e-2, 1e-3]}
  #   ]
  # },
  # 'GPR': {
  #   'function': GaussianProcessRegressor,
  #   'params': {},
  #   'search_params': [
  #     {'kernel': [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))], 'alpha': np.logspace(-2, 0, 3)},
  #     {'kernel': [1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)],'alpha': np.logspace(-2, 0, 3)},
  #   ]
  # },
  # 'MLP': {
  #   'function': MLPRegressor,
  #   'params': { 'max_iter': 400, 'verbose': 0 },
  #   'search_params': {
  #     'learning_rate': ["invscaling"],
  #     'learning_rate_init': [0.001, 0.01, 0.1],
  #     'hidden_layer_sizes': [(25,), (50), (100,), (150,), (50,25), (50,50), (100,50), (100, 100), (150, 100)],
  #     'activation': ["logistic", "relu", "tanh"]
  #   }
  # },
  # 'KNN': {
  #   'function': KNeighborsRegressor,
  #   'params': { 'n_jobs': 1 if single_processor else -1 },
  #   'search_params': {
  #     'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  #     'weights': ('uniform', 'distance')
  #   }
  # },
}

def r2_adj(observation, prediction):
  r2 = r2_score(observation, prediction)
  (n, p) = observation.shape
  return 1 - (1-r2) * (n-1) / (n-p-1)

def saveCsv(filename, prediction, original):
  prediction = pd.DataFrame(prediction, original)
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  prediction.to_csv(filename + '.csv', decimal='.', sep=';')


train_durations = []
test_durations = []

original_dataset = pd.read_csv(dataset_path)

shuffled_dataset = original_dataset.sample(frac=1).reset_index(drop=True)

# Calculate total iterations for progress tracking
total_wells = len(['PRK014', 'PRK028', 'PRK045', 'PRK052', 'PRK060', 'PRK061', 'PRK083', 'PRK084', 'PRK085'])
total_regressors = len(['GTB', 'KRR', 'GPR', 'MLP', 'KNN'])
total_iterations = total_wells * total_regressors
current_iteration = 0

# Time tracking for estimation
start_time = datetime.datetime.now()
iteration_times = []

for well in ['PRK014', 'PRK028', 'PRK045', 'PRK052', 'PRK060', 'PRK061', 'PRK083', 'PRK084', 'PRK085']:
  features = [('CLOSE_GOR_' + well + '_Z'+v[1] + '_S' + v[0]) for v in list(itertools.product(['1','2','3','4','5'],['1','2','3']))]
  weif_column = 'WEIF[' + well + ']'

  # Copy features (rules) and add WEIF at the end
  dataset = shuffled_dataset[features].copy()
  dataset[weif_column] = shuffled_dataset[weif_column]

  dataset[features] = dataset[features].div(3700) # Normalize y

  dataset = dataset.to_numpy()

  print(well, flush=True)

  # Track best regressor performance for this well
  best_regressor_name = None
  best_mean_rmse = float('inf')
  best_mean_r2 = -float('inf')
  best_regressor_data = {}

  for regressor_name in ['GTB', 'KRR', 'GPR', 'MLP', 'KNN']:
    current_iteration += 1
    progress = (current_iteration / total_iterations) * 100

    # Calculate and show overall estimated time remaining
    if current_iteration > 1:
      avg_iteration_time = np.mean(iteration_times)
      remaining_iterations = total_iterations - current_iteration + 1
      estimated_remaining_time = avg_iteration_time * remaining_iterations

      # Convert to hours, minutes, seconds
      hours = int(estimated_remaining_time // 3600)
      minutes = int((estimated_remaining_time % 3600) // 60)
      seconds = int(estimated_remaining_time % 60)

      if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds}s"
      elif minutes > 0:
        time_str = f"{minutes}m {seconds}s"
      else:
        time_str = f"{seconds}s"

      print(f"Progress: {progress:.1f}% - Processing {well} with {regressor_name}")
      print(f"⏰ OVERALL ESTIMATED TIME REMAINING: {time_str}")
    else:
      print(f"Progress: {progress:.1f}% - Processing {well} with {regressor_name}")

    # Show overall time elapsed
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    elapsed_hours = int(elapsed_time // 3600)
    elapsed_minutes = int((elapsed_time % 3600) // 60)
    elapsed_seconds = int(elapsed_time % 60)

    if elapsed_hours > 0:
      elapsed_str = f"{elapsed_hours}h {elapsed_minutes}m {elapsed_seconds}s"
    elif elapsed_minutes > 0:
      elapsed_str = f"{elapsed_minutes}m {elapsed_seconds}s"
    else:
      elapsed_str = f"{elapsed_seconds}s"

    print(f"⏱️  TIME ELAPSED: {elapsed_str}")

    RMSEs = []
    R2s = []

    k_fold = KFold(n_splits=folds, shuffle=True, random_state=42)

    for fold_idx, (train_indices, test_indices) in enumerate(k_fold.split(dataset)):
      current_fold = fold_idx + 1
      fold_start_time = datetime.datetime.now()

      output_path = output_directory + '/' + well + '/' + regressor_name + '/FOLD_' + str(current_fold) + '/'
      os.makedirs(os.path.dirname(output_path), exist_ok=True)

      with open(output_path + '/output.txt', 'w') as output_file:
        print('Command line arguments: ' + str(sys.argv), file=output_file, flush=True)

        print(file=output_file)
        print(regressor_name, file=output_file, flush=True)
        print(file=output_file)

        train_data = dataset[train_indices, :]
        test_data = dataset[test_indices, :]

        regressor_info = regressors[regressor_name]
        regressor_function = regressor_info['function'](**regressor_info['params'])

        # Train

        train_start = datetime.datetime.now()

        train_features = train_data[:, :-1]
        train_y = train_data[:, -1].reshape(-1, 1)
        y_scaler = MinMaxScaler().fit(train_y)
        train_y = y_scaler.transform(train_y)

        regressor = GridSearchCV(
          estimator=regressor_function,
          param_grid=regressor_info['search_params'],
          cv=cross_validations,
          n_jobs=1 if single_processor else -1,
          verbose=0
        )

        try:
          regressor.fit(train_features, train_y.ravel())
        except Exception as e:
          print('Fit failed', file=output_file, flush=True)
          print(e, file=output_file, flush=True)
          continue

        train_end = datetime.datetime.now()
        train_durations.append((train_end - train_start).total_seconds())

        # Test

        test_start = datetime.datetime.now()

        test_features = test_data[:, :-1]
        test_y = test_data[:, -1].reshape(-1, 1)
        test_y = y_scaler.transform(test_y)

        try:
          predicted_y = regressor.predict(test_features).reshape(-1, 1)
        except Exception as e:
          print('Predict failed', file=output_file, flush=True)
          print(e, file=output_file, flush=True)
          continue

        test_end = datetime.datetime.now()
        test_durations.append((test_end - test_start).total_seconds())

        RMSE = np.sqrt(mean_squared_error(test_y, predicted_y))
        R2 = r2_adj(test_y.reshape(-1, 1), predicted_y.reshape(-1, 1))

        RMSEs.append(RMSE)
        R2s.append(R2)

        print(f'Fold {current_fold} - RMSE: {RMSE:.6f} R2: {R2:.6f}', file=output_file, flush=True)

        # Save predictions for this fold
        denormalized_test_y = y_scaler.inverse_transform(test_y)
        denormalized_prediction = y_scaler.inverse_transform(predicted_y)

        saveCsv(
          filename=output_path + '/csv',
          prediction=denormalized_prediction,
          original=denormalized_test_y.ravel()
        )

        # Save model and scaler for this fold
        with open(output_path + '/proxy.pkl', 'wb') as f:
          pickle.dump(regressor, f)

        with open(output_path + '/y_scaler.pkl', 'wb') as f:
          pickle.dump(y_scaler, f)

        print(file=output_file)
        print("RMSE: %f" % RMSE, file=output_file)
        print("R2: %f" % R2, file=output_file)
        print(file=output_file)
        print("Regressor training seconds: %f" % ((train_end - train_start).total_seconds()), file=output_file, flush=True)
        print("Regressor testing seconds: %f" % ((test_end - test_start).total_seconds()), file=output_file, flush=True)
        print(file=output_file, flush=True)

        # Show fold completion time and estimate remaining folds
        fold_end_time = datetime.datetime.now()
        fold_duration = (fold_end_time - fold_start_time).total_seconds()
        remaining_folds = folds - current_fold

        # Always show fold completion
        print(f"      ✓ Fold {current_fold} completed in {fold_duration:.1f}s")

        # Only show remaining folds estimate if there are remaining folds
        if remaining_folds > 0:
          avg_fold_time = fold_duration  # Use current fold as estimate
          estimated_folds_time = avg_fold_time * remaining_folds

          if estimated_folds_time > 60:
            minutes = int(estimated_folds_time // 60)
            seconds = int(estimated_folds_time % 60)
            fold_time_str = f"{minutes}m {seconds}s"
          else:
            fold_time_str = f"{int(estimated_folds_time)}s"

          print(f"      ⏱️  Remaining folds: {fold_time_str}")

    # After all folds for this regressor, check if it's the best
    if RMSEs:  # Only if we have valid results
      mean_rmse = np.mean(RMSEs)
      mean_r2 = np.mean(R2s)

      if mean_rmse < best_mean_rmse:
        best_mean_rmse = mean_rmse
        best_mean_r2 = mean_r2
        best_regressor_name = regressor_name
        best_regressor_data = {
          'name': regressor_name,
          'mean_rmse': mean_rmse,
          'std_rmse': np.std(RMSEs),
          'mean_r2': mean_r2,
          'std_r2': np.std(R2s),
          'all_rmse': RMSEs,
          'all_r2': R2s,
          'train_duration': np.sum(train_durations),
          'test_duration': np.sum(test_durations)
        }

    # Track time for this iteration and estimate remaining time
    iteration_end_time = datetime.datetime.now()
    iteration_duration = (iteration_end_time - start_time).total_seconds()
    iteration_times.append(iteration_duration)

  # Create BEST_REGRESSOR folder with the best model and info
  if best_regressor_name:
    best_output_path = output_directory + '/' + well + '/BEST_REGRESSOR/'
    os.makedirs(best_output_path, exist_ok=True)

    # Copy the best model files from the winning regressor
    best_regressor_path = output_directory + '/' + well + '/' + best_regressor_name + '/'

    # Find the fold with the best RMSE for the best regressor
    best_fold_idx = np.argmin(best_regressor_data['all_rmse'])
    best_fold = best_fold_idx + 1

    # Copy files from the best fold
    source_fold_path = best_regressor_path + 'FOLD_' + str(best_fold) + '/'

    # Copy model files
    shutil.copy2(source_fold_path + 'proxy.pkl', best_output_path + 'proxy.pkl')
    shutil.copy2(source_fold_path + 'y_scaler.pkl', best_output_path + 'y_scaler.pkl')
    shutil.copy2(source_fold_path + 'csv.csv', best_output_path + 'predictions.csv')

    # Create INFO.txt with detailed model information
    with open(best_output_path + 'INFO.txt', 'w') as info_file:
      info_file.write(f"BEST REGRESSOR FOR WELL {well}\n")
      info_file.write("=" * 50 + "\n\n")
      info_file.write(f"Regressor: {best_regressor_name}\n")
      info_file.write(f"Selected Fold: {best_fold}\n\n")
      info_file.write("PERFORMANCE METRICS:\n")
      info_file.write("-" * 25 + "\n")
      info_file.write(f"Mean RMSE: {best_regressor_data['mean_rmse']:.6f}\n")
      info_file.write(f"RMSE Std: {best_regressor_data['std_rmse']:.6f}\n")
      info_file.write(f"Mean R²: {best_regressor_data['mean_r2']:.6f}\n")
      info_file.write(f"R² Std: {best_regressor_data['std_r2']:.6f}\n\n")
      info_file.write("FOLD-BY-FOLD RESULTS:\n")
      info_file.write("-" * 25 + "\n")
      for i, (rmse, r2) in enumerate(zip(best_regressor_data['all_rmse'], best_regressor_data['all_r2']), 1):
        info_file.write(f"Fold {i}: RMSE={rmse:.6f}, R²={r2:.6f}\n")
      info_file.write(f"\nTRAINING TIME: {best_regressor_data['train_duration']:.2f} seconds\n")
      info_file.write(f"TESTING TIME: {best_regressor_data['test_duration']:.2f} seconds\n")
      info_file.write(f"\nSELECTION CRITERION: Lowest mean RMSE across {folds} folds\n")
      info_file.write(f"SELECTION DATE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print("All wells processed!")
print(f"Total training time: {np.sum(train_durations):.2f} seconds")
print(f"Total testing time: {np.sum(test_durations):.2f} seconds")
