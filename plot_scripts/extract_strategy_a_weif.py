import pandas as pd
import os

# Configure data file paths
# Files are in the project's data/ folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
runs516_path = os.path.join(DATA_DIR, '516 strategies with results.csv')

# Load data
runs516 = pd.read_csv(runs516_path)
wells = ['PRK014', 'PRK028', 'PRK045', 'PRK052', 'PRK060', 'PRK061', 'PRK083', 'PRK084', 'PRK085']

# Extract WEIF values for each well (Strategy A: TVALUE == 1 and MAX == 1)
all_data = []
for well in wells:
    filtered = runs516[(runs516[well + '_TVALUE'] == 1) & (runs516[well + '_MAX'] == 1)]
    for _, row in filtered.iterrows():
        data = {'WELL': well, 'WEIF': row[well]}
        if 'RUN' in filtered.columns:
            data['RUN'] = row['RUN']
        all_data.append(data)

result_df = pd.DataFrame(all_data)
if 'RUN' in result_df.columns:
    result_df = result_df[['RUN', 'WELL', 'WEIF']]

# Save to CSV in project's outputs folder
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, 'strategy_a_new_execution_weif_values.csv')
result_df.to_csv(output_file, index=False)

print(f"File saved: {output_file}")
print(f"Total rows: {len(result_df)}")
print(f"\nSummary by well:")
for well in wells:
    well_data = result_df[result_df['WELL'] == well]
    print(f"\n{well}: {len(well_data)} values")
    print(f"  Minimum: {well_data['WEIF'].min():.2e}")
    print(f"  Maximum: {well_data['WEIF'].max():.2e}")
    print(f"  Mean: {well_data['WEIF'].mean():.2e}")
    print(f"  Median: {well_data['WEIF'].median():.2e}")

print(f"\nFirst 20 rows:")
print(result_df.head(20))
