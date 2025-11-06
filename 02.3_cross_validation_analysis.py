import os
import sys
import pickle
import datetime
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
dataset_path = './otm_gor_icv5_clean.csv'
proxies_folder = './02.2_output_proxy_models_bkp'
output_directory = './02.3_cross_validation_analysis'
folds = 10

# Wells to analyze
wells = ['PRK014', 'PRK028', 'PRK045', 'PRK052', 'PRK060', 'PRK061', 'PRK083', 'PRK084', 'PRK085']

# GTB Configuration for training
gtb_config = {
    'min_samples_split': 0.1,
    'n_estimators': 100,
    'learning_rate': 0.1,
    'loss': 'squared_error',
    'random_state': 42
}

# Create output directory
os.makedirs(output_directory, exist_ok=True)

def r2_adj(observation, prediction, n_features):
    """Calculate adjusted R² score"""
    r2 = r2_score(observation, prediction)
    n = len(observation)
    p = n_features  # Number of features (corrected)
    return 1 - (1-r2) * (n-1) / (n-p-1)

def normalize_0_to_1(data):
    """Normalize data to range [0, 1]"""
    return (data - data.min()) / (data.max() - data.min())

def format_scientific_value(value, threshold=0.01):
    """Format a number using fixed-point notation if above threshold, scientific otherwise (LaTeX-style exponent)."""
    if abs(value) >= threshold:
        return f"{value:.3f}"
    exponent_fmt = f"{value:.2e}"
    base, exp = exponent_fmt.split('e')
    exp = int(exp)
    return f"{base}×10^{{{exp}}}"


def format_rmse_value(rmse):
    """Format RMSE using metric prefixes (×10³, ×10⁶, etc.) when appropriate."""
    abs_rmse = abs(rmse)
    if abs_rmse >= 1_000_000:
        return f"{rmse / 1_000_000:.2f}×10⁶"
    elif abs_rmse >= 1_000:
        return f"{rmse / 1_000:.2f}×10³"
    else:
        return f"{rmse:.2f}"


def load_and_prepare_data():
    """Load dataset and prepare features for each well"""
    print("Loading dataset...")
    original_dataset = pd.read_csv(dataset_path)

    # Shuffle dataset for better cross-validation
    shuffled_dataset = original_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    well_data = {}

    for well in wells:
        print(f"Preparing data for {well}...")

        # Extract GOR features for this well
        features = [('CLOSE_GOR_' + well + '_Z'+v[1] + '_S' + v[0])
                   for v in list(itertools.product(['1','2','3','4','5'],['1','2','3']))]

        # Extract WEIF column for this well
        weif_column = 'WEIF[' + well + ']'

        # Create dataset for this well
        well_dataset = shuffled_dataset[features + [weif_column]].copy()

        # Remove rows with NaN values
        well_dataset = well_dataset.dropna()

        # Normalize GOR features (divide by 3700 as in original script)
        well_dataset[features] = well_dataset[features].div(3700)

        well_data[well] = {
            'features': features,
            'weif_column': weif_column,
            'data': well_dataset,
            'X': well_dataset[features].values,
            'y': well_dataset[weif_column].values
        }

        print(f"  {well}: {len(well_dataset)} samples, {len(features)} features")

    return well_data



def perform_cross_validation(well_name, X, y, n_features):
    """Perform k-fold cross-validation with model training for each fold"""
    print(f"Performing cross-validation for {well_name}...")

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    results = {
        'fold': [],
        'train_size': [],
        'test_size': [],
        'rmse': [],
        'r2': [],
        'r2_adj': [],
        'y_true': [],
        'y_pred': [],
        'train_indices': [],
        'test_indices': [],
        'regressor_type': 'GTB (Cross-validation)'
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"  Fold {fold}: Training on {len(train_idx)} samples, testing on {len(test_idx)} samples")

        # Split data for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # Scale target variable
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Train GTB model for this fold
        model = GradientBoostingRegressor(**gtb_config)
        model.fit(X_train_scaled, y_train_scaled)

        # Make predictions
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        r2_adj_val = r2_adj(y_test, y_pred, n_features)

        # Store results
        results['fold'].append(fold)
        results['train_size'].append(len(train_idx))
        results['test_size'].append(len(test_idx))
        results['rmse'].append(rmse)
        results['r2'].append(r2)
        results['r2_adj'].append(r2_adj_val)
        results['y_true'].append(y_test)
        results['y_pred'].append(y_pred)
        results['train_indices'].append(train_idx)
        results['test_indices'].append(test_idx)

        print(f"    RMSE: {rmse:.6f}, R²: {r2:.6f}, R²_adj: {r2_adj_val:.6f}")

    return results

def create_scatter_plots(well_data, cv_results):
    """Create 3x3 scatter plot matrix showing original vs predicted WEIF for each well"""
    print("Creating scatter plot matrix...")

    # Set up the plot (30% smaller: 15 * 0.7 = 10.5)
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.5))
    fig.suptitle('Original vs Predicted WEIF Values', fontsize=16, y=0.95)

    # Add matrix-level axis labels
    fig.supxlabel('Original WEIF (Normalized)', fontsize=14, y=0.05)
    fig.supylabel('Predicted WEIF (Normalized)', fontsize=14, x=0.05)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    for idx, well in enumerate(wells):
        if idx >= 9:  # Safety check
            break

        ax = axes_flat[idx]

        # Get all true and predicted values for this well
        all_y_true = []
        all_y_pred = []

        for fold in range(folds):
            all_y_true.extend(cv_results[well]['y_true'][fold])
            all_y_pred.extend(cv_results[well]['y_pred'][fold])

        # Convert to numpy arrays
        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)

        # Normalize to 0-1 range
        y_true_norm = normalize_0_to_1(y_true)
        y_pred_norm = normalize_0_to_1(y_pred)

        # Create scatter plot
        ax.scatter(y_true_norm, y_pred_norm, alpha=0.6, s=20)

        # Add perfect prediction line
        min_val = min(y_true_norm.min(), y_pred_norm.min())
        max_val = max(y_true_norm.max(), y_pred_norm.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1)

        # Calculate overall metrics for this well
        overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        overall_r2 = r2_score(y_true, y_pred)

        # Set title with metrics and regressor type
        regressor_type = cv_results[well]['regressor_type']

        # Format values using helper functions
        r2_str = format_scientific_value(overall_r2)
        rmse_str = format_rmse_value(overall_rmse)

        ax.set_title(f'{well} ({regressor_type})\nR² = {r2_str}, RMSE = {rmse_str}',
                    fontsize=12, pad=10)

        # Set axis limits to 0-1
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add grid
        ax.grid(True, alpha=0.3)

        # Make axes square
        ax.set_aspect('equal', adjustable='box')

    # Remove any unused subplots
    for idx in range(len(wells), 9):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(pad=2.5)

    # Save the plot as PNG
    plot_path_png = os.path.join(output_directory, 'weif_scatter_matrix.png')
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    print(f"Scatter plot matrix saved to: {plot_path_png}")

    # Save the plot as PDF
    plot_path_pdf = os.path.join(output_directory, 'weif_scatter_matrix.pdf')
    plt.savefig(plot_path_pdf, format='pdf', bbox_inches='tight')
    print(f"Scatter plot matrix saved to: {plot_path_pdf}")

    # Close the plot to free memory (no display window)
    plt.close()



def main():
    """Main execution function"""
    print("=" * 60)
    print("CROSS-VALIDATION ANALYSIS WITH MODEL TRAINING")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Model type: GradientBoostingRegressor")
    print(f"Output directory: {output_directory}")
    print(f"Wells: {len(wells)}")
    print(f"Cross-validation folds: {folds}")
    print("=" * 60)

    start_time = datetime.datetime.now()

    try:
        # Load and prepare data
        well_data = load_and_prepare_data()

        # Perform cross-validation for each well
        cv_results = {}
        for well in wells:
            print(f"\n{'='*40}")
            print(f"Processing well: {well}")
            print(f"{'='*40}")

            X = well_data[well]['X']
            y = well_data[well]['y']
            n_features = len(well_data[well]['features'])

            cv_results[well] = perform_cross_validation(well, X, y, n_features)

        # Create scatter plot matrix
        print(f"\n{'='*40}")
        print("CREATING SCATTER PLOT MATRIX")
        print(f"{'='*40}")
        create_scatter_plots(well_data, cv_results)

        # Final summary
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\n{'='*60}")
        print("CROSS-VALIDATION TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Total execution time: {duration:.2f} seconds")
        print(f"Cross-validation folds: {folds}")
        print(f"Model: GradientBoostingRegressor with {gtb_config}")
        print(f"Scatter plot matrix (PNG): {output_directory}/weif_scatter_matrix.png")
        print(f"Scatter plot matrix (PDF): {output_directory}/weif_scatter_matrix.pdf")
        print(f"Results summary displayed above")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
