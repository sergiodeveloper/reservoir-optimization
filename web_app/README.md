# Target Rules Generator Web Application

A web-based interface for generating optimal GOR (Gas-Oil Ratio) threshold rules for reservoir well optimization.

## Features

- **Step 1: Upload Dataset** - Upload CSV files with well data
- **Step 2: Configure Parameters** - Map columns to WEIF and GOR features for each well
- **Step 3: Generate Target Rules** - Generate optimal rules with linear regression and decision tree analysis

## Installation

1. Install required Python packages:
```bash
pip install -r requirements.txt
```

2. Install Graphviz (required for decision tree visualization):
   - **macOS**: `brew install graphviz`
   - **Linux**: `sudo apt-get install graphviz` or `sudo yum install graphviz`
   - **Windows**: Download from [Graphviz website](https://graphviz.org/download/)

## Running the Application

1. Navigate to the web_app directory:
```bash
cd web_app
```

2. Run the Flask server:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Step 1: Upload Dataset
- Click or drag and drop a CSV file
- The file will be uploaded and column information will be displayed

### Step 2: Configure Parameters
- **Number of Rows**: Select how many rows to use for training (or "All rows")
- **NPV Column**: Optionally select the NPV column
- **Well Configurations**: For each of the 9 wells:
  - Select the WEIF (Water Equivalent Investment Factor) column
  - Select exactly 15 GOR feature columns (representing 3 zones × 5 stages)
- Click "Save Configuration" when done

### Step 3: Generate Target Rules
- Click "Generate Target Rules"
- View results for each well:
  - **Linear Regression Statistics**: Coefficients and T-values
  - **Decision Tree**: Visual representation of the decision tree
  - **Target Rule**: The generated optimal rule
- Download the rules as a Python file

## Output

The generated target rules are saved in Python format compatible with the `target_rules.py` file structure used in the main optimization pipeline.

## Notes

- The application uses temporary file storage for uploaded datasets
- Decision tree images are generated as SVG files
- All results are stored in session memory (for production, consider using proper session management)
