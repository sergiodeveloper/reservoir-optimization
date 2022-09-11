
import os
import sys
import pickle
import datetime
import itertools
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit

testing = True # Change to False to run all iterations

single_processor = False

if testing:
  print('TESTING')

trials = 5 if not testing else 3

dataset_path = './simplified_dataset.csv'

output_directory = './02_output_proxy_models'
cross_validations = 10 if not testing else 3
num_splits = 10 if not testing else 4

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
  'KRR': {
    'function': KernelRidge,
    'params': {},
    'search_params': [
      {'kernel': ['poly'], 'degree': [2,3,4], 'alpha': [1e0, 0.1, 1e-2, 1e-3]},
      {'kernel': ['rbf'], 'gamma':  np.logspace(-3, 3, 7), 'alpha': [1e0, 0.1, 1e-2, 1e-3]}
    ]
  },
  'GPR': {
    'function': GaussianProcessRegressor,
    'params': {},
    'search_params': [
      {'kernel': [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))], 'alpha': np.logspace(-2, 0, 3)},
      {'kernel': [1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)],'alpha': np.logspace(-2, 0, 3)},
    ]
  },
  'MLP': {
    'function': MLPRegressor,
    'params': { 'max_iter': 400, 'verbose': 0 },
    'search_params': {
      'learning_rate': ["invscaling"],
      'learning_rate_init': [0.001, 0.01, 0.1],
      'hidden_layer_sizes': [(25,), (50), (100,), (150,), (50,25), (50,50), (100,50), (100, 100), (150, 100)],
      'activation': ["logistic", "relu", "tanh"]
    }
  },
  'KNN': {
    'function': KNeighborsRegressor,
    'params': { 'n_jobs': 1 if single_processor else -1 },
    'search_params': {
      'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
      'weights': ('uniform', 'distance')
    }
  },
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

for well in ['PRK014', 'PRK028', 'PRK045', 'PRK052', 'PRK060', 'PRK061', 'PRK083', 'PRK084', 'PRK085']:
  features = [('CLOSE_GOR_' + well + '_Z'+v[1] + '_S' + v[0]) for v in list(itertools.product(['1','2','3','4','5'],['1','2','3']))]
  weif_column = 'WEIF[' + well + ']'

  # Copy features (rules) and add WEIF at the end
  dataset = original_dataset[features].copy()
  dataset[weif_column] = original_dataset[weif_column]

  dataset[features] = dataset[features].div(3700) # Normalize y

  dataset = dataset.to_numpy()

  print(well, flush=True)

  for regressor_name in ['GTB', 'KRR', 'GPR', 'MLP', 'KNN']:
    for trial in range(1, trials + 1):
      output_path = output_directory + '/' + well + '/' + regressor_name + '/TRIAL_' + str(trial) + '/'
      os.makedirs(os.path.dirname(output_path), exist_ok=True)

      with open(output_path + '/output.txt', 'w') as output_file:
        if testing:
          print('TESTING', file=output_file, flush=True)
          print(file=output_file)

        print('Command line arguments: ' + str(sys.argv), file=output_file, flush=True)

        print('Trial ' + str(trial), file=output_file, flush=True)

        print(file=output_file)
        print(regressor_name, flush=True)
        print(regressor_name, file=output_file, flush=True)
        print(file=output_file)

        RMSEs = []
        R2s = []

        best_regressor = None
        best_RMSE = None
        best_R2 = None
        best_y_scaler = None
        current_test_features = None
        current_test_y = None

        k_fold = ShuffleSplit(n_splits=num_splits, random_state=trial, test_size=0.1)

        for train_indices, test_indices in k_fold.split(dataset):
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

          if best_RMSE is None or RMSE < best_RMSE:
            print('- RMSE: ' + str(RMSE) + ' R2: ' + str(R2), file=output_file, flush=True)

            best_regressor = regressor
            best_RMSE = RMSE
            best_R2 = R2
            best_y_scaler = y_scaler
            current_test_features = test_features
            current_test_y = test_y

        print(file=output_file)
        print("Best RMSE %f" % (best_RMSE), file=output_file)
        print("Best R2 %f" % (best_R2), file=output_file)

        # Save predictions from best iteration

        prediction = best_regressor.predict(current_test_features).reshape(-1, 1)

        denormalized_test_y = best_y_scaler.inverse_transform(current_test_y)
        denormalized_prediction = best_y_scaler.inverse_transform(prediction)

        saveCsv(
          filename=output_path + '/csv',
          prediction=denormalized_prediction,
          original=denormalized_test_y.ravel()
        )

        with open(output_path + '/proxy.pkl', 'wb') as f:
          pickle.dump(best_regressor, f)

        with open(output_path + '/y_scaler.pkl', 'wb') as f:
          pickle.dump(best_y_scaler, f)

        print(file=output_file)
        print("RMSE mean %f std %f " % (np.mean(RMSEs), np.std(RMSEs)), file=output_file)
        print("R2 mean %f std %f " % (np.mean(R2s), np.std(R2s)), file=output_file)

        print(file=output_file)
        print("Regressor training seconds: %f" % (np.sum(train_durations)), file=output_file, flush=True)
        print("Regressor testing seconds: %f" % (np.sum(test_durations)), file=output_file, flush=True)
        # print("Genetic algorithm seconds: %f" %(genetic_duration), file=output_file, flush=True)
        print(file=output_file, flush=True)




