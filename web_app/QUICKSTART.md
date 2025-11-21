# Quick Start Guide

## Prerequisites

1. **Python 3.7+** installed
2. **Graphviz** installed:
   - macOS: `brew install graphviz`
   - Linux: `sudo apt-get install graphviz`
   - Windows: Download from https://graphviz.org/download/

## Setup

1. **Navigate to web_app directory:**
   ```bash
   cd web_app
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or if using the project's virtual environment:
   ```bash
   source ../venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

   Or use the provided script:
   ```bash
   ./run.sh
   ```

4. **Open in browser:**
   ```
   http://localhost:5000
   ```

## Usage Workflow

### Step 1: Upload Dataset
- Upload a CSV file containing your well data
- The file should have columns for:
  - GOR (Gas-Oil Ratio) features (15 per well: 3 zones × 5 stages)
  - WEIF (Water Equivalent Investment Factor) values for each well

### Step 2: Configure Parameters
- **Training Rows**: Select how many rows to use (or "All rows")
- **NPV Column**: Optionally select NPV column
- **For each well** (9 wells total):
  - Select the **WEIF column** for that well
  - Select exactly **15 GOR feature columns** (representing 3 zones × 5 stages)

### Step 3: Generate Rules
- Click "Generate Target Rules"
- View results:
  - Linear regression statistics
  - Decision tree visualization
  - Generated target rules
- Download rules as Python file

## Troubleshooting

### Graphviz not found
If you see errors about Graphviz:
- Install Graphviz system package (see Prerequisites)
- Restart the Flask server after installation

### Port already in use
If port 5000 is busy:
- Edit `app.py` and change `port=5000` to another port (e.g., `port=5001`)

### File upload errors
- Ensure CSV file is properly formatted
- Check file size (max 100MB)
- Verify CSV has valid column headers

## File Structure

```
web_app/
├── app.py                 # Flask backend server
├── requirements.txt        # Python dependencies
├── README.md              # Full documentation
├── run.sh                 # Startup script
├── templates/
│   └── index.html         # Main frontend page
├── static/
│   ├── css/
│   │   └── style.css      # Styling
│   └── js/
│       └── app.js         # Frontend logic
├── uploads/               # Uploaded CSV files (temporary)
└── temp/                  # Generated files (temporary)
```

## Output

The generated `target_rules.py` file follows the same format as the main project's target rules and can be used directly in the optimization pipeline.
