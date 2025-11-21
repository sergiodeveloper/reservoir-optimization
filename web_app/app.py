import os
import sys
import json
import tempfile
import shutil
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import graphviz
import sklearn
from typing import List, Any
import decimal
import itertools

# Add parent directory to path to import from 03_target_rules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMP_FOLDER'] = 'temp'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

# Store session data (in production, use proper session management)
session_data = {}

# Import helper functions from 03_target_rules.py
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
            "samples": int(tree_.n_node_samples[node_id]),
            "prediction": float(tree_.value[node_id][0][0]),
            "variable": None,
            "threshold": None,
            "parent_side": parent_side,
            "parent": None if not parent_reference else parent_node,
            "left": None,
            "right": None,
        }

        if tree_.feature[node_id] != sklearn.tree._tree.TREE_UNDEFINED:
            branch['variable'] = feature_name[node_id]
            branch['threshold'] = float(tree_.threshold[node_id])
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
    skipped_nodes = set()  # Track nodes with too few samples

    for leaf in leaves:
        if leaf['samples'] < minimum_samples:
            skipped_nodes.add(leaf['node_id'])  # Mark as skipped
            continue
        if best_leaf is None or leaf['prediction'] > best_leaf['prediction'] \
            or (leaf['prediction'] == best_leaf['prediction'] and leaf['samples'] > best_leaf['samples']):
            best_leaf = leaf

    if best_leaf is None:
        return None, None, skipped_nodes

    branch_conditions = []
    path_nodes = set()
    path_edges = set()  # Store (parent_id, child_id, side) tuples

    current_node = best_leaf
    while current_node is not None:
        path_nodes.add(current_node['node_id'])
        parent = current_node['parent']
        if parent is None:
            break
        path_nodes.add(parent['node_id'])
        # Store edge: (parent_id, child_id, side)
        path_edges.add((parent['node_id'], current_node['node_id'], current_node['parent_side']))

        condition = parent['variable']
        condition += ' <= ' if current_node['parent_side'] == 'left' else ' > '
        condition += format_number(parent['threshold'])
        branch_conditions = [condition] + branch_conditions
        current_node = current_node['parent']

    return ' & '.join(branch_conditions), (path_nodes, path_edges), skipped_nodes

def tree_to_graphviz(tree_root, path_nodes=None, path_edges=None, skipped_nodes=None):
    if path_nodes is None:
        path_nodes = set()
    if path_edges is None:
        path_edges = set()
    if skipped_nodes is None:
        skipped_nodes = set()

    def recurse(branch):
        code = ''
        node_id = branch['node_id']
        is_highlighted = node_id in path_nodes
        is_skipped = node_id in skipped_nodes

        if branch['left'] is not None and branch['right'] is not None:
            # Node styling
            node_style = 'fillcolor="#e8f4f8", style="rounded,filled", color="#667eea", penwidth=2' if is_highlighted else 'color="black"'
            code += str(node_id) + ' [label=<'
            code += '<FONT POINT-SIZE="8">Samples:</FONT><BR />' + str(branch['samples'])
            code += '>, ' + node_style + '] ;\n'

            code += recurse(branch=branch['left'])

            # Left edge styling
            left_edge_highlighted = (node_id, branch['left']['node_id'], 'left') in path_edges
            edge_style = 'color="#667eea", penwidth=3' if left_edge_highlighted else 'color="black"'
            code += str(node_id) + ' -> ' + str(branch['left']['node_id'])
            code += ' [label="' + branch['variable'] + ' ≤ ' + format_number(branch['threshold']) + '", ' + edge_style + '] ;\n'

            code += recurse(branch=branch['right'])

            # Right edge styling
            right_edge_highlighted = (node_id, branch['right']['node_id'], 'right') in path_edges
            edge_style = 'color="#667eea", penwidth=3' if right_edge_highlighted else 'color="black"'
            code += str(node_id) + ' -> ' + str(branch['right']['node_id'])
            code += ' [label="' + branch['variable'] + ' > ' + format_number(branch['threshold']) + '", ' + edge_style + '] ;\n'
        else:
            # Leaf node styling
            if is_skipped:
                # Mark skipped nodes with X and red background
                node_style = 'fillcolor="#f8d7da", style="rounded,filled", color="#dc3545", penwidth=2'
                code += str(node_id) + ' [label=<'
                code += '<FONT POINT-SIZE="8">Samples:</FONT><BR />' + str(branch['samples'])
                code += '<BR /><FONT POINT-SIZE="8">Value:</FONT><BR />'
                code += str(round(branch['prediction'], 3))
                code += '<BR /><FONT POINT-SIZE="10" COLOR="red"><B>✗ &lt; 45 samples</B></FONT>'
                code += '>, ' + node_style + '] ;\n'
            elif is_highlighted:
                # Optimal leaf (green)
                node_style = 'fillcolor="#d4edda", style="rounded,filled", color="#28a745", penwidth=2'
                code += str(node_id) + ' [label=<'
                code += '<FONT POINT-SIZE="8">Samples:</FONT><BR />' + str(branch['samples'])
                code += '<BR /><FONT POINT-SIZE="8">Value:</FONT><BR />'
                code += str(round(branch['prediction'], 3))
                code += '>, ' + node_style + '] ;\n'
            else:
                # Regular leaf
                node_style = 'color="black"'
                code += str(node_id) + ' [label=<'
                code += '<FONT POINT-SIZE="8">Samples:</FONT><BR />' + str(branch['samples'])
                code += '<BR /><FONT POINT-SIZE="8">Value:</FONT><BR />'
                code += str(round(branch['prediction'], 3))
                code += '>, ' + node_style + '] ;\n'
        return code

    code = '''digraph Tree {
node [shape=box, style="rounded", color="black", fontname="helvetica"] ;
edge [fontname="helvetica", fontsize=8] ;
'''
    code += recurse(branch=tree_root)
    code += '}\n'
    return code

def calculateTvalue(linear_regression: LinearRegression, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    predictions = linear_regression.predict(X)
    newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))
    tvalues = np.append(linear_regression.intercept_, linear_regression.coef_) \
        / np.sqrt(MSE * np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    stats = pd.DataFrame()

    for index, tvalue in enumerate(tvalues):
        if index == 0:
            continue
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

def previous_factor_value(value: int):
    combined_list = factor_possible_values.copy()
    combined_list.append(value)
    combined_list = list(set(combined_list))
    combined_list.sort()
    index = combined_list.index(value)
    if index - 1 >= 0:
        return combined_list[index - 1]
    return value

def next_factor_value(value: int):
    combined_list = factor_possible_values.copy()
    combined_list.append(value)
    combined_list = list(set(combined_list))
    combined_list.sort()
    index = combined_list.index(value)
    if index + 1 <= len(combined_list) - 1:
        return combined_list[index + 1]
    return value

def values_for_branch(branch: str):
    branch_conditions = [a.strip().split() for a in branch.split('&')]
    tree_values = {}

    for condition in branch_conditions:
        variable_name = condition[0]
        comparator = condition[1]
        condition_value = int(float(condition[2].split('.')[0]))

        recommended_value = condition_value

        if comparator == '>':
            recommended_value = next_factor_value(condition_value)
        if comparator == '>=' and recommended_value not in factor_possible_values:
            recommended_value = next_factor_value(condition_value)
        if comparator == '<':
            recommended_value = previous_factor_value(condition_value)
        if comparator == '<=' and recommended_value not in factor_possible_values:
            recommended_value = previous_factor_value(condition_value)

        if variable_name not in tree_values:
            tree_values[variable_name] = recommended_value
        else:
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
        coefficient for coefficient in coefficients
        if abs(coefficient['tvalue']) >= 1.96
    ]

def negativeCoefficientNames(coefficients):
    return [
        coefficient['variable_name'] for coefficient in coefficients
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

    for coefficient in coefficients:
        variable_name = coefficient['variable_name']
        if variable_name not in rule:
            if maximize:
                if coefficient['coefficient'] > 0:
                    rule[variable_name] = 3700
                else:
                    rule[variable_name] = 0
            else:
                if coefficient['coefficient'] > 0:
                    rule[variable_name] = 0
                else:
                    rule[variable_name] = 3700

    rule = dict(sorted(rule.items()))
    return rule

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Server is running'}), 200

@app.route('/api/upload-rules', methods=['POST'])
def upload_rules_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'rules_' + filename)
        file.save(filepath)

        # Read CSV to get columns
        try:
            df = pd.read_csv(filepath, nrows=5)  # Just read first few rows to get columns
            columns = df.columns.tolist()

            session_data['rules_file'] = filepath
            session_data['rules_columns'] = columns

            return jsonify({
                'success': True,
                'filename': filename,
                'columns': columns,
                'row_count': len(pd.read_csv(filepath))
            })
        except Exception as e:
            return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400
    else:
        return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

@app.route('/api/upload-results', methods=['POST'])
def upload_results_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'results_' + filename)
        file.save(filepath)

        # Read CSV to get columns
        try:
            df = pd.read_csv(filepath, nrows=5)  # Just read first few rows to get columns
            columns = df.columns.tolist()

            session_data['results_file'] = filepath
            session_data['results_columns'] = columns

            return jsonify({
                'success': True,
                'filename': filename,
                'columns': columns,
                'row_count': len(pd.read_csv(filepath))
            })
        except Exception as e:
            return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400
    else:
        return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

@app.route('/api/configure', methods=['POST'])
def configure_parameters():
    data = request.json

    if 'rules_file' not in session_data or 'results_file' not in session_data:
        return jsonify({'error': 'Both rules and results files must be uploaded'}), 400

    # Store configuration
    session_data['num_rows'] = data.get('num_rows', 'all')
    session_data['rules_mappings'] = data.get('rules_mappings', {})  # {well_name: {gor_columns: [str]}}
    session_data['results_mappings'] = data.get('results_mappings', {})  # {well_name: {weif_column: str}}
    session_data['rules_id_column'] = data.get('rules_id_column', '')
    session_data['results_id_column'] = data.get('results_id_column', '')

    return jsonify({'success': True, 'message': 'Configuration saved'})

@app.route('/api/generate-rules', methods=['POST'])
def generate_rules():
    if 'rules_file' not in session_data or 'results_file' not in session_data:
        return jsonify({'error': 'Both rules and results files must be uploaded'}), 400

    if 'rules_mappings' not in session_data or 'results_mappings' not in session_data:
        return jsonify({'error': 'Parameters not configured'}), 400

    try:
        # Load both datasets
        rules_filepath = session_data['rules_file']
        results_filepath = session_data['results_file']

        df_rules = pd.read_csv(rules_filepath)
        df_results = pd.read_csv(results_filepath)

        # Limit rows if specified
        if session_data.get('num_rows', 'all') != 'all':
            num_rows = int(session_data['num_rows'])
            df_rules = df_rules.head(num_rows)
            df_results = df_results.head(num_rows)

        # Merge datasets - use user-selected ID columns or fall back to auto-detection
        rules_id_col = session_data.get('rules_id_column', '').strip()
        results_id_col = session_data.get('results_id_column', '').strip()

        # If user didn't select ID columns, try auto-detection
        if not rules_id_col:
            for col in df_rules.columns:
                if col.lower() in ['id', 'execution', 'iteration', 'model']:
                    rules_id_col = col
                    break

        if not results_id_col:
            for col in df_results.columns:
                if col.lower() in ['id', 'execution', 'iteration', 'model']:
                    results_id_col = col
                    break

        if rules_id_col and results_id_col:
            # Verify columns exist
            if rules_id_col not in df_rules.columns:
                return jsonify({'error': f'ID column "{rules_id_col}" not found in rules file'}), 400
            if results_id_col not in df_results.columns:
                return jsonify({'error': f'ID column "{results_id_col}" not found in results file'}), 400

            # Merge on ID columns
            df = pd.merge(df_rules, df_results, left_on=rules_id_col, right_on=results_id_col, how='inner')

            if len(df) == 0:
                return jsonify({'error': 'No matching rows found when merging on ID columns. Please verify the ID columns are correct.'}), 400
        else:
            # Assume same order
            if len(df_rules) != len(df_results):
                return jsonify({'error': f'Row count mismatch: Rules file has {len(df_rules)} rows, Results file has {len(df_results)} rows. Please select ID columns for proper matching.'}), 400
            df = pd.concat([df_rules.reset_index(drop=True), df_results.reset_index(drop=True)], axis=1)

        rules_mappings = session_data['rules_mappings']
        results_mappings = session_data['results_mappings']
        results = {}

        for well_name in rules_mappings.keys():
            if well_name not in results_mappings:
                return jsonify({'error': f'Well {well_name} missing in results mappings'}), 400

            gor_columns = rules_mappings[well_name]['gor_columns']
            weif_column = results_mappings[well_name]['weif_column']

            if len(gor_columns) != 15:
                return jsonify({'error': f'Well {well_name} must have exactly 15 GOR columns'}), 400

            # Prepare data
            X = df[gor_columns].values
            y = df[weif_column].values.reshape(-1, 1) / 1e8  # Normalize WEIF

            # Create feature names in format Z1_S1, Z1_S2, ..., Z3_S5
            # Map the 15 columns to zone/stage combinations
            feature_names = []
            for i in range(15):
                zone = (i // 5) + 1  # Zone 1, 2, or 3
                stage = (i % 5) + 1  # Stage 1, 2, 3, 4, or 5
                feature_names.append(f'Z{zone}_S{stage}')

            # Linear Regression
            linear_regression = LinearRegression().fit(X, y)
            linear_stats = calculateTvalue(linear_regression, X, y, feature_names)

            # Decision Tree (using default max_depth of 5, can be made configurable)
            max_depth = 5
            tree_regressor = DecisionTreeRegressor(random_state=0, max_depth=max_depth).fit(X, y)
            tree_root = tree_to_dictionary(tree_regressor, feature_names, parent_reference=True)

            # Get best branch
            result = get_branch_with_best_prediction(tree_root, minimum_samples=45)

            if result is None or result[0] is None:
                return jsonify({'error': f'Could not find suitable branch for well {well_name}'}), 400

            branch_with_best_prediction, path_info, skipped_nodes = result
            path_nodes, path_edges = path_info if path_info and len(path_info) == 2 else (set(), set())

            # Build rule and track which features came from where
            rule = build_rule(
                branch=branch_with_best_prediction,
                linear_regression_info=linear_stats,
                use_tvalue=True,
            )

            # Extract features from tree branch
            branch_features = set()
            for condition in branch_with_best_prediction.split(' & '):
                if condition.strip():
                    feature = condition.split()[0]  # Extract feature name (e.g., "Z1_S1" from "Z1_S1 <= 300")
                    branch_features.add(feature)

            # Identify significant regression features (t-value >= 1.96)
            significant_features = []
            for row in linear_stats.to_dict('records'):
                if 'Intercept' not in row['Variable']:
                    try:
                        t_value = abs(float(row['T-Value']))
                        if t_value >= 1.96:
                            significant_features.append({
                                'feature': row['Variable'],
                                'coefficient': float(row['Coefficient']),
                                't_value': float(row['T-Value'])
                            })
                    except:
                        pass

            # Generate tree visualization with highlighted path and skipped nodes
            tree_dot = tree_to_graphviz(tree_root, path_nodes=path_nodes, path_edges=path_edges, skipped_nodes=skipped_nodes)
            temp_dir = os.path.join(app.config['TEMP_FOLDER'], well_name)
            os.makedirs(temp_dir, exist_ok=True)

            try:
                image = graphviz.Source(tree_dot)
                tree_image_path = os.path.join(temp_dir, f'{well_name}_tree')
                image.render(directory=temp_dir, filename=f'{well_name}_tree', format='svg', cleanup=True)
                print(f"Tree image generated for {well_name}")
            except Exception as e:
                # If graphviz fails, we'll still return results but without tree image
                print(f"Warning: Could not generate tree image for {well_name}: {e}")
                import traceback
                traceback.print_exc()
                tree_image_path = None

            # Convert rule to ordered list (Z1_S1, Z1_S2, ..., Z3_S5)
            rule_list = []
            for i in range(15):
                zone = (i // 5) + 1
                stage = (i % 5) + 1
                feature_key = f'Z{zone}_S{stage}'
                rule_list.append(str(rule.get(feature_key, '0')))

            # Check if tree image was generated
            tree_image_exists = os.path.exists(
                os.path.join(app.config['TEMP_FOLDER'], well_name, f'{well_name}_tree.svg')
            )

            results[well_name] = {
                'linear_regression': linear_stats.to_dict('records'),
                'tree_image_path': f'{well_name}_tree.svg' if tree_image_exists else None,
                'target_rule': {k: str(v) for k, v in rule.items()},
                'rule_list': rule_list,
                'branch_conditions': branch_with_best_prediction,
                'branch_features': list(branch_features),
                'significant_features': significant_features
            }

        session_data['results'] = results
        return jsonify({'success': True, 'results': results})

    except Exception as e:
        import traceback
        return jsonify({'error': f'Error generating rules: {str(e)}\n{traceback.format_exc()}'}), 500

@app.route('/api/tree-image/<well_name>')
def get_tree_image(well_name):
    tree_image_path = os.path.join(app.config['TEMP_FOLDER'], well_name, f'{well_name}_tree.svg')
    if os.path.exists(tree_image_path):
        return send_file(tree_image_path, mimetype='image/svg+xml')
    # Return a placeholder or error message
    return jsonify({'error': 'Tree image not found. Make sure Graphviz is installed.'}), 404

@app.route('/api/download-rules', methods=['GET'])
def download_rules():
    if 'results' not in session_data:
        return jsonify({'error': 'No rules generated'}), 400

    results = session_data['results']
    rules_dict = {
        'simplified_dataset_rules': {
            well: result['rule_list']
            for well, result in results.items()
        }
    }

    # Create Python file content
    content = "# AUTOMATICALLY GENERATED - DO NOT EDIT\n\n"
    content += f"RULES = {json.dumps(rules_dict, indent=2)}\n"

    # Save to temp file
    temp_file = os.path.join(app.config['TEMP_FOLDER'], 'target_rules.py')
    with open(temp_file, 'w') as f:
        f.write(content)

    return send_file(temp_file, as_attachment=True, download_name='target_rules.py')

if __name__ == '__main__':
    import atexit
    import signal
    import subprocess
    import os

    def cleanup_port():
        """Clean up port 5003 before starting (only run once, not on reload)"""
        # Only clean up if we're the main process (not a reloader child)
        if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
            try:
                current_pid = os.getpid()
                # Kill processes on port 5003 (but not this one)
                result = subprocess.run(['lsof', '-ti:5003'], capture_output=True, text=True, timeout=2)
                if result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        try:
                            pid_int = int(pid)
                            # Don't kill our own process
                            if pid_int != current_pid:
                                subprocess.run(['kill', '-9', pid], check=False, timeout=1)
                        except (ValueError, subprocess.TimeoutExpired):
                            pass
            except (subprocess.TimeoutExpired, Exception):
                pass

    def cleanup():
        """Clean up on exit"""
        pass

    def signal_handler(sig, frame):
        """Handle signals gracefully"""
        cleanup()
        sys.exit(0)

    # Clean up port before starting (only in main process)
    cleanup_port()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup)

    # Use threaded mode for better concurrency
    # Disable reloader to prevent stopped processes from blocking the port
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        print("Flask server reloaded")
    else:
        print("Starting Flask server on http://127.0.0.1:5003")

    # Disable reloader to prevent port blocking issues
    # Set use_reloader=False to avoid stopped processes
    app.run(debug=True, host='127.0.0.1', port=5003, use_reloader=False, threaded=True)
