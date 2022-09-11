
# Reservoir optimization - Project structure

## `00_sample_dataset.py`

Script to generate a sample random dataset with the following columns:
```
CLOSE_GOR_{well}_{Z1 ... Z3}_{S1 ... S5} - value in the range [100, 3700]
WEIF[{well}] - value in the range [10^8, 10^10]
```

The file is saved in `sample_original_dataset.csv`

---

## `01_simplify_dataset.py`

Simplifies the dataset with zeroes and saves it in `simplified_dataset.csv`

---

## `02_proxy_models.py`

Runs several regression models for comparison. The script outputs the trained models in the following file structure:

```
02_output_proxy_models
├── {well_name}
│   ├── {GPR, GTB, KNN, KRR, MLP}
│   │   ├── TRIAL_{trial_number}
│   │   │   ├── csv.csv - Predictions for the test set with the model
│   │   │   ├── output.txt - Model RMSE, R2 and training and testing time
│   │   │   ├── rules.txt - Rules for the model
│   │   │   ├── proxy.pkl - Trained model
│   │   │   └── y_scaler.pkl - Scaler for the target variable
│   │  ...
└── ...
```

Set the variable `testing` to `False` if you want to run all iterations, and `True` if you are testing the code.

---

## `03_target_rules.py`

Regressão de árvore e linear, criação de regras ideais, imagens de árvores e imagens dos coeficientes de forma automatizada

Treina os modelos de regressão linear e árvore de regressão para cada poço, usando os dataset simplificado e original.

A partir desses modelos são geradas as imagens. Depois são geradas as regras ideais, baseado no algoritmo de maximização dos valores dos gatilhos e levando o t-value em consideração.

As regras ideais para os dois conjuntos de treinos são alvos no arquivo target_rules.py

It outputs the linear regression coefficients on standard output. The regression trees are saved in the following file structure:
```
03_output_target_rules
├── {well_name}_tree.pdf
...
```

The generated rules are saved in the file `target_rules.py`

---

## `04_best_proxy.py`

Gets the best GTB proxy model for each well and saves it in the following file structure:
```
04_output_best_proxy
├── {well_name}
│   ├── proxy.pkl - Trained model
│   └── y_scaler.pkl - Scaler for the target variable
└── ...
```

The best proxy model are used in scripts 5 and 6.

---

## `05_adjust_all_factors.py` and `06_adjust_closest_factor.py`

Given a set of rules (`target_rules.py`), the scripts simulate the rule adjustment process for each well saves images in the current directory.
