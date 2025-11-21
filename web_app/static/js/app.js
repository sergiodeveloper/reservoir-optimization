// Well names (can be made configurable)
const WELL_NAMES = ['PRK014', 'PRK028', 'PRK045', 'PRK052', 'PRK060', 'PRK061', 'PRK083', 'PRK084', 'PRK085'];

let rulesFile = null;
let resultsFile = null;
let rulesColumns = [];
let resultsColumns = [];
let rulesMappings = {};
let resultsMappings = {};
let configuration = {};
let rulesRowCount = 0;
let resultsRowCount = 0;
let rulesIdColumn = '';
let resultsIdColumn = '';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeRulesUpload();
    initializeResultsUpload();
    initializeParameters();
});

// Step 1: Upload Rules File
function initializeRulesUpload() {
    const rulesFileInput = document.getElementById('rulesFileInput');
    const uploadRulesBox = document.getElementById('uploadRulesBox');
    const uploadRulesArea = document.getElementById('uploadRulesArea');

    uploadRulesBox.addEventListener('click', () => rulesFileInput.click());
    uploadRulesArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadRulesBox.style.borderColor = '#764ba2';
    });
    uploadRulesArea.addEventListener('dragleave', () => {
        uploadRulesBox.style.borderColor = '#667eea';
    });
    uploadRulesArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadRulesBox.style.borderColor = '#667eea';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleRulesFileUpload(files[0]);
        }
    });

    rulesFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleRulesFileUpload(e.target.files[0]);
        }
    });
}

async function handleRulesFileUpload(file) {
    if (!file.name.endsWith('.csv')) {
        showStatus('uploadRulesStatus', 'Please upload a CSV file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    const uploadRulesBox = document.getElementById('uploadRulesBox');
    showStatus('uploadRulesStatus', 'Uploading rules file...', 'info');
    uploadRulesBox.style.pointerEvents = 'none';
    uploadRulesBox.style.opacity = '0.6';

    try {
        const response = await fetch('/api/upload-rules', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            rulesFile = file.name;
            rulesColumns = data.columns;
            rulesRowCount = data.row_count || 0;

            showStatus('uploadRulesStatus', 'Rules file uploaded successfully!', 'success');
            document.getElementById('rulesFileName').textContent = data.filename;
            document.getElementById('rulesRowCount').textContent = data.row_count;
            document.getElementById('rulesColumnCount').textContent = data.columns.length;
            document.getElementById('rulesFileInfo').style.display = 'block';

            // Populate ID column selector
            populateIdColumnSelector('rulesIdColumn', data.columns, 'rules');
            document.getElementById('rulesIdColumnSelection').style.display = 'block';

            // Show column selection
            populateRulesColumnSelects();
            document.getElementById('rulesColumnSelection').style.display = 'block';

            // Check if we can enable save button
            checkConfigurationReady();

            // Enable step 2 (Upload Results File)
            setTimeout(() => {
                document.getElementById('step2').classList.add('active');
            }, 500);
        } else {
            showStatus('uploadRulesStatus', data.error || 'Upload failed', 'error');
        }
    } catch (error) {
        showStatus('uploadRulesStatus', 'Error uploading file: ' + error.message, 'error');
    } finally {
        uploadRulesBox.style.pointerEvents = 'auto';
        uploadRulesBox.style.opacity = '1';
    }
}

function populateIdColumnSelector(selectId, columns, fileType) {
    const select = document.getElementById(selectId);
    if (!select) return;

    // Clear existing options except the first one
    while (select.options.length > 1) {
        select.remove(1);
    }

    // Add all columns as options
    columns.forEach(col => {
        const option = document.createElement('option');
        option.value = col;
        option.textContent = col;
        select.appendChild(option);
    });

    // Select first column by default, or common ID column names if found
    let selectedColumn = columns[0]; // Default to first column
    const commonIdNames = ['id', 'execution', 'iteration', 'model', 'index'];
    for (const col of columns) {
        if (commonIdNames.includes(col.toLowerCase())) {
            selectedColumn = col; // Prefer common ID names over first column
            break;
        }
    }

    select.value = selectedColumn;
    if (fileType === 'rules') {
        rulesIdColumn = selectedColumn;
    } else {
        resultsIdColumn = selectedColumn;
    }

    // Add event listener to track selection
    select.addEventListener('change', (e) => {
        if (fileType === 'rules') {
            rulesIdColumn = e.target.value || '';
        } else {
            resultsIdColumn = e.target.value || '';
        }
    });
}

function checkRowCountMatch() {
    const warningDiv = document.getElementById('rowCountWarning');
    if (!warningDiv || !rulesFile || !resultsFile) return;

    if (rulesRowCount > 0 && resultsRowCount > 0 && rulesRowCount !== resultsRowCount) {
        warningDiv.textContent = `⚠️ Warning: Row count mismatch! Rules file has ${rulesRowCount} rows, but Results file has ${resultsRowCount} rows. Make sure the ID columns are correctly selected for proper matching.`;
        warningDiv.className = 'status-message error';
        warningDiv.style.display = 'block';
    } else if (rulesRowCount > 0 && resultsRowCount > 0 && rulesRowCount === resultsRowCount) {
        warningDiv.textContent = `✓ Row counts match (${rulesRowCount} rows)`;
        warningDiv.className = 'status-message success';
        warningDiv.style.display = 'block';
    } else {
        warningDiv.style.display = 'none';
    }
}

function populateRulesColumnSelects() {
    const rulesWellsContainer = document.getElementById('rulesWellsContainer');
    rulesWellsContainer.innerHTML = '';

    WELL_NAMES.forEach(well => {
        const wellDiv = document.createElement('div');
        wellDiv.className = 'well-config';
        wellDiv.innerHTML = `
            <div class="well-header" style="display: flex; justify-content: space-between; align-items: center; cursor: pointer; padding: 10px; background: #f8f9fa; border-radius: 5px; margin-bottom: 10px;">
                <h4 style="margin: 0;">${well}</h4>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="selected-count" style="color: #667eea; font-weight: 600;">
                        Selected: <span id="count-${well}">0</span> / 15
                    </span>
                    <button class="minimize-btn" data-well="${well}" style="background: none; border: none; color: #667eea; cursor: pointer; font-size: 18px; padding: 5px 10px;">−</button>
                </div>
            </div>
            <div class="well-content" id="content-${well}">
                <div class="form-group">
                    <label>GOR Feature Columns (select exactly 15):</label>
                    <div class="multi-select" id="gor-select-${well}"></div>
                </div>
            </div>
            <div class="well-tags" id="tags-${well}" style="display: none; margin-top: 10px; flex-wrap: wrap; gap: 5px;"></div>
        `;
        rulesWellsContainer.appendChild(wellDiv);

        // Auto-detect columns matching the well pattern
        // Pattern: {well}_Z{1-3}_S{1-5}
        const zones = ['Z1', 'Z2', 'Z3'];
        const stages = ['S1', 'S2', 'S3', 'S4', 'S5'];
        const expectedColumns = [];

        zones.forEach(zone => {
            stages.forEach(stage => {
                expectedColumns.push(`${well}_${zone}_${stage}`);
            });
        });

        // Find matching columns in the actual file
        const matchingColumns = [];
        expectedColumns.forEach(expectedCol => {
            const found = rulesColumns.find(col => col === expectedCol || col.includes(expectedCol));
            if (found) {
                matchingColumns.push(found);
            }
        });

        // Populate GOR multi-select
        const gorSelect = wellDiv.querySelector(`#gor-select-${well}`);
        rulesColumns.forEach(col => {
            const label = document.createElement('label');
            const isAutoSelected = matchingColumns.includes(col);
            label.innerHTML = `
                <input type="checkbox" value="${col}" data-well="${well}" class="gor-checkbox" ${isAutoSelected ? 'checked' : ''}>
                <span>${col}</span>
            `;
            gorSelect.appendChild(label);
        });

        // Add checkbox listeners
        const checkboxes = wellDiv.querySelectorAll('.gor-checkbox');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                updateGORCount(well);
                updateWellTags(well);
            });
        });

        // Add minimize button listener
        const minimizeBtn = wellDiv.querySelector('.minimize-btn');
        const wellHeader = wellDiv.querySelector('.well-header');
        const wellContent = wellDiv.querySelector(`#content-${well}`);
        const wellTags = wellDiv.querySelector(`#tags-${well}`);

        // Set initial state to minimized
        wellContent.style.display = 'none';
        wellTags.style.display = 'flex';
        minimizeBtn.textContent = '+';

        wellHeader.addEventListener('click', () => {
            const isMinimized = wellContent.style.display === 'none';
            if (isMinimized) {
                wellContent.style.display = 'block';
                wellTags.style.display = 'none';
                minimizeBtn.textContent = '−';
            } else {
                wellContent.style.display = 'none';
                wellTags.style.display = 'flex';
                updateWellTags(well);
                minimizeBtn.textContent = '+';
            }
        });

        // Update count and tags after auto-selection
        updateGORCount(well);
        updateWellTags(well);
    });
}

function updateWellTags(well) {
    const checkboxes = document.querySelectorAll(`.gor-checkbox[data-well="${well}"]:checked`);
    const tagsContainer = document.getElementById(`tags-${well}`);

    if (!tagsContainer) return;

    tagsContainer.innerHTML = '';

    if (checkboxes.length === 0) {
        tagsContainer.innerHTML = '<span style="color: #999; font-style: italic;">No columns selected</span>';
        return;
    }

    checkboxes.forEach(checkbox => {
        const tag = document.createElement('span');
        tag.className = 'selected-tag';
        tag.textContent = checkbox.value;
        tag.style.cssText = 'display: inline-block; padding: 4px 8px; background: #e9ecef; color: #6c757d; border-radius: 3px; font-size: 12px; margin: 2px;';
        tagsContainer.appendChild(tag);
    });
}

function updateGORCount(well) {
    const checkboxes = document.querySelectorAll(`.gor-checkbox[data-well="${well}"]:checked`);
    const count = checkboxes.length;
    const countElement = document.getElementById(`count-${well}`);
    if (countElement) {
        countElement.textContent = count;

        if (count === 15) {
            countElement.style.color = '#28a745';
        } else if (count > 15) {
            countElement.style.color = '#dc3545';
        } else {
            countElement.style.color = '#667eea';
        }
    }

    // Update tags if well is minimized
    updateWellTags(well);

    // Check if configuration is ready
    checkConfigurationReady();
}

function checkConfigurationReady() {
    // Check if both files are uploaded
    if (!rulesFile || !resultsFile) {
        updateConfigurationSummary(0, 0, 0);
        return;
    }

    // Check if all wells have 15 GOR columns selected
    let allGORSelected = true;
    let totalGORSelected = 0;
    let wellsConfigured = 0;
    WELL_NAMES.forEach(well => {
        const checkboxes = document.querySelectorAll(`.gor-checkbox[data-well="${well}"]:checked`);
        if (checkboxes.length === 15) {
            wellsConfigured++;
            totalGORSelected += 15;
        } else {
            allGORSelected = false;
        }
    });

    // Check if all wells have WEIF columns selected
    let allWEIFSelected = true;
    let totalWEIFSelected = 0;
    WELL_NAMES.forEach(well => {
        const weifSelect = document.querySelector(`.weif-select[data-well="${well}"]`);
        if (weifSelect && weifSelect.value) {
            totalWEIFSelected++;
        } else {
            allWEIFSelected = false;
        }
    });

    // Update summary
    updateConfigurationSummary(wellsConfigured, totalGORSelected, totalWEIFSelected);

    // Enable generate button if everything is ready
    const generateBtn = document.getElementById('generateBtn');
    if (generateBtn) {
        generateBtn.disabled = !(allGORSelected && allWEIFSelected);
    }
}

function updateConfigurationSummary(wellsConfigured, totalGOR, totalWEIF) {
    const summaryDiv = document.getElementById('configSummary');
    const summaryRows = document.getElementById('summaryRows');
    const summaryWells = document.getElementById('summaryWells');
    const summaryGOR = document.getElementById('summaryGOR');
    const summaryWEIF = document.getElementById('summaryWEIF');
    const summaryAlert = document.getElementById('summaryAlert');

    if (!summaryDiv) return;

    // Show summary if at least one file is uploaded
    if (rulesFile || resultsFile) {
        summaryDiv.style.display = 'block';

        // Update rows
        const numRows = document.getElementById('numRows');
        const customRows = document.getElementById('customRows');
        if (numRows) {
            // Get available row count (use minimum of both files, or max if one is 0)
            const availableRows = Math.min(rulesRowCount || 0, resultsRowCount || 0) || Math.max(rulesRowCount || 0, resultsRowCount || 0);

            if (numRows.value === 'all') {
                summaryRows.textContent = availableRows > 0 ? `All rows (${availableRows} total)` : 'All rows';

                // Check if row counts differ between files
                if (rulesRowCount > 0 && resultsRowCount > 0 && rulesRowCount !== resultsRowCount) {
                    // Show warning if row counts don't match
                    summaryRows.style.color = '#e74c3c';
                    if (summaryAlert) {
                        summaryAlert.innerHTML = `<strong>⚠️ Warning:</strong> Row count mismatch! Rules file has ${rulesRowCount} rows, but Results file has ${resultsRowCount} rows. Make sure the ID columns are correctly selected for proper matching, or some rows may not be matched.`;
                        summaryAlert.className = 'summary-alert error';
                        summaryAlert.style.display = 'block';
                        summaryAlert.style.visibility = 'visible';
                        summaryAlert.style.opacity = '1';
                    }
                } else {
                    summaryRows.style.color = '';
                    // Hide alert when row counts match
                    if (summaryAlert) {
                        summaryAlert.style.display = 'none';
                    }
                }
            } else {
                let selectedRows = 0;
                if (numRows.value === 'custom') {
                    selectedRows = parseInt(customRows.value) || 0;
                } else {
                    selectedRows = parseInt(numRows.value) || 0;
                }

                if (selectedRows > 0) {
                    if (selectedRows > availableRows && availableRows > 0) {
                        // Selected more than available - show warning
                        summaryRows.textContent = `${selectedRows} rows (only ${availableRows} available)`;
                        summaryRows.style.color = '#e74c3c';
                        // Show alert in summary area
                        if (summaryAlert) {
                            summaryAlert.innerHTML = `<strong>⚠️ Warning:</strong> You selected ${selectedRows} rows, but only ${availableRows} rows are available. Only ${availableRows} rows will be used for training.`;
                            summaryAlert.className = 'summary-alert error';
                            summaryAlert.style.display = 'block';
                            summaryAlert.style.visibility = 'visible';
                            summaryAlert.style.opacity = '1';
                        }
                    } else {
                        summaryRows.textContent = `${selectedRows} rows`;
                        summaryRows.style.color = '';
                        // Hide alert when rows are compatible
                        if (summaryAlert) {
                            summaryAlert.style.display = 'none';
                        }
                    }
                } else {
                    summaryRows.textContent = numRows.value === 'custom' ? 'Custom' : `${numRows.value} rows`;
                    summaryRows.style.color = '';
                    // Hide alert when no valid selection
                    if (summaryAlert) {
                        summaryAlert.style.display = 'none';
                    }
                }
            }
        }

        // Update wells
        summaryWells.textContent = `${wellsConfigured} / 9`;

        // Update GOR columns
        summaryGOR.textContent = `${totalGOR} selected (${wellsConfigured} wells × 15 columns)`;

        // Update WEIF columns
        summaryWEIF.textContent = `${totalWEIF} selected`;
    } else {
        summaryDiv.style.display = 'none';
    }
}

// Step 2: Upload Results File
function initializeResultsUpload() {
    const resultsFileInput = document.getElementById('resultsFileInput');
    const uploadResultsBox = document.getElementById('uploadResultsBox');
    const uploadResultsArea = document.getElementById('uploadResultsArea');

    uploadResultsBox.addEventListener('click', () => resultsFileInput.click());
    uploadResultsArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadResultsBox.style.borderColor = '#764ba2';
    });
    uploadResultsArea.addEventListener('dragleave', () => {
        uploadResultsBox.style.borderColor = '#667eea';
    });
    uploadResultsArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadResultsBox.style.borderColor = '#667eea';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleResultsFileUpload(files[0]);
        }
    });

    resultsFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleResultsFileUpload(e.target.files[0]);
        }
    });
}

async function handleResultsFileUpload(file) {
    if (!file.name.endsWith('.csv')) {
        showStatus('uploadResultsStatus', 'Please upload a CSV file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    const uploadResultsBox = document.getElementById('uploadResultsBox');
    showStatus('uploadResultsStatus', 'Uploading results file...', 'info');
    uploadResultsBox.style.pointerEvents = 'none';
    uploadResultsBox.style.opacity = '0.6';

    try {
        const response = await fetch('/api/upload-results', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            resultsFile = file.name;
            resultsColumns = data.columns;
            resultsRowCount = data.row_count || 0;

            showStatus('uploadResultsStatus', 'Results file uploaded successfully!', 'success');
            document.getElementById('resultsFileName').textContent = data.filename;
            document.getElementById('resultsRowCount').textContent = data.row_count;
            document.getElementById('resultsColumnCount').textContent = data.columns.length;
            document.getElementById('resultsFileInfo').style.display = 'block';

            // Populate ID column selector
            populateIdColumnSelector('resultsIdColumn', data.columns, 'results');
            document.getElementById('resultsIdColumnSelection').style.display = 'block';

            // Check row count match and show warning if needed
            checkRowCountMatch();

            // Show column selection
            populateResultsColumnSelects();
            document.getElementById('resultsColumnSelection').style.display = 'block';

            // Check if we can enable save button
            checkConfigurationReady();

            // Enable step 3
            setTimeout(() => {
                document.getElementById('step3').classList.add('active');
            }, 500);
        } else {
            showStatus('uploadResultsStatus', data.error || 'Upload failed', 'error');
        }
    } catch (error) {
        showStatus('uploadResultsStatus', 'Error uploading file: ' + error.message, 'error');
    } finally {
        uploadResultsBox.style.pointerEvents = 'auto';
        uploadResultsBox.style.opacity = '1';
    }
}

function populateResultsColumnSelects() {
    const resultsWellsContainer = document.getElementById('resultsWellsContainer');
    resultsWellsContainer.innerHTML = '';

    WELL_NAMES.forEach(well => {
        const wellDiv = document.createElement('div');
        wellDiv.className = 'well-config-compact';
        wellDiv.innerHTML = `
            <div class="well-row" style="display: flex; align-items: center; gap: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px; margin-bottom: 8px;">
                <label style="min-width: 80px; font-weight: 600; color: #667eea; margin: 0;">${well}:</label>
                <select class="weif-select" data-well="${well}" style="flex: 1; padding: 8px; border: 2px solid #e0e0e0; border-radius: 6px; font-size: 0.95em;">
                    <option value="">-- Select WEIF Column --</option>
                </select>
            </div>
        `;
        resultsWellsContainer.appendChild(wellDiv);

        // Auto-detect WEIF column matching the well pattern
        // Pattern: WEIF[{well}] or WEIF[{well}-W] or similar variations
        const expectedPatterns = [
            `WEIF[${well}]`,
            `WEIF[${well}-W]`,
            `WEIF_${well}`,
            `WEIF ${well}`,
            well  // Sometimes the column might just be the well name
        ];

        // Find matching column
        let matchingColumn = null;
        for (const pattern of expectedPatterns) {
            matchingColumn = resultsColumns.find(col =>
                col === pattern ||
                col.includes(pattern) ||
                col.toLowerCase() === pattern.toLowerCase()
            );
            if (matchingColumn) break;
        }

        // Populate WEIF select
        const weifSelect = wellDiv.querySelector('.weif-select');
        resultsColumns.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            if (matchingColumn && col === matchingColumn) {
                option.selected = true;
            }
            weifSelect.appendChild(option);
        });

        // Add change listener to check configuration
        weifSelect.addEventListener('change', () => {
            checkConfigurationReady();
        });

        // Trigger check after auto-selection
        if (matchingColumn) {
            checkConfigurationReady();
        }
    });
}

// Step 3: Configure Parameters & Generate Target Rules
function initializeParameters() {
    const numRowsSelect = document.getElementById('numRows');
    const customRowsInput = document.getElementById('customRows');
    const generateBtn = document.getElementById('generateBtn');
    const downloadBtn = document.getElementById('downloadBtn');

    numRowsSelect.addEventListener('change', () => {
        if (numRowsSelect.value === 'custom') {
            customRowsInput.style.display = 'block';
        } else {
            customRowsInput.style.display = 'none';
        }
        checkConfigurationReady(); // Update summary when rows change
        // Also directly update summary to show alert immediately
        if (rulesFile && resultsFile) {
            const wellsConfigured = WELL_NAMES.filter(well => {
                const checkboxes = document.querySelectorAll(`.gor-checkbox[data-well="${well}"]:checked`);
                return checkboxes.length === 15;
            }).length;
            const totalGOR = wellsConfigured * 15;
            const totalWEIF = WELL_NAMES.filter(well => {
                const weifSelect = document.querySelector(`.weif-select[data-well="${well}"]`);
                return weifSelect && weifSelect.value;
            }).length;
            updateConfigurationSummary(wellsConfigured, totalGOR, totalWEIF);
        }
    });

    customRowsInput.addEventListener('input', () => {
        checkConfigurationReady(); // Update summary when custom rows change
        // Also directly update summary to show alert immediately
        if (rulesFile && resultsFile) {
            const wellsConfigured = WELL_NAMES.filter(well => {
                const checkboxes = document.querySelectorAll(`.gor-checkbox[data-well="${well}"]:checked`);
                return checkboxes.length === 15;
            }).length;
            const totalGOR = wellsConfigured * 15;
            const totalWEIF = WELL_NAMES.filter(well => {
                const weifSelect = document.querySelector(`.weif-select[data-well="${well}"]`);
                return weifSelect && weifSelect.value;
            }).length;
            updateConfigurationSummary(wellsConfigured, totalGOR, totalWEIF);
        }
    });

    generateBtn.addEventListener('click', generateRules);
    downloadBtn.addEventListener('click', downloadRules);
}


async function generateRules() {
    const generateBtn = document.getElementById('generateBtn');
    const generateStatus = document.getElementById('generateStatus');
    const resultsContainer = document.getElementById('resultsContainer');
    const numRows = document.getElementById('numRows');
    const customRows = document.getElementById('customRows');

    // First, save configuration
    // Collect rules mappings
    rulesMappings = {};
    let isValid = true;

    WELL_NAMES.forEach(well => {
        const gorCheckboxes = document.querySelectorAll(`.gor-checkbox[data-well="${well}"]:checked`);
        const gorColumns = Array.from(gorCheckboxes).map(cb => cb.value);

        if (gorColumns.length !== 15) {
            showStatus('configStatus', `${well} must have exactly 15 GOR columns selected (currently ${gorColumns.length})`, 'error');
            isValid = false;
            return;
        }

        rulesMappings[well] = {
            gor_columns: gorColumns
        };
    });

    // Collect results mappings
    resultsMappings = {};
    WELL_NAMES.forEach(well => {
        const weifSelect = document.querySelector(`.weif-select[data-well="${well}"]`);
        const weifColumn = weifSelect.value;

        if (!weifColumn) {
            showStatus('configStatus', `Please select WEIF column for ${well}`, 'error');
            isValid = false;
            return;
        }

        resultsMappings[well] = {
            weif_column: weifColumn
        };
    });

    if (!isValid) {
        return;
    }

    const configData = {
        num_rows: numRows.value === 'custom' ? customRows.value : numRows.value,
        rules_mappings: rulesMappings,
        results_mappings: resultsMappings,
        rules_id_column: rulesIdColumn || '',
        results_id_column: resultsIdColumn || ''
    };

    // Save configuration first
    try {
        const configResponse = await fetch('/api/configure', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(configData)
        });

        const configData_result = await configResponse.json();

        if (!configResponse.ok) {
            showStatus('configStatus', configData_result.error || 'Failed to save configuration', 'error');
            return;
        }

        configuration = configData;
    } catch (error) {
        showStatus('configStatus', 'Error saving configuration: ' + error.message, 'error');
        return;
    }

    // Now generate rules
    generateBtn.disabled = true;
    generateBtn.innerHTML = '<span class="loading"></span>Generating rules...';
    showStatus('generateStatus', 'Generating target rules... This may take a few moments.', 'info');

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout

        const response = await fetch('/api/generate-rules', {
            method: 'POST',
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();

        if (response.ok) {
            showStatus('generateStatus', 'Target rules generated successfully!', 'success');
            displayResults(data.results);
            resultsContainer.style.display = 'block';
        } else {
            showStatus('generateStatus', data.error || 'Failed to generate rules', 'error');
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            showStatus('generateStatus', 'Request timed out. The operation is taking longer than expected. Please check server logs.', 'error');
        } else {
            showStatus('generateStatus', 'Error generating rules: ' + error.message, 'error');
        }
        console.error('Generate rules error:', error);
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate Target Rules';
    }
}

function displayResults(results) {
    const resultsTabs = document.getElementById('resultsTabs');
    const resultsContent = document.getElementById('resultsContent');

    resultsTabs.innerHTML = '';
    resultsContent.innerHTML = '';

    // Create tabs
    Object.keys(results).forEach((well, index) => {
        const tab = document.createElement('button');
        tab.className = 'tab' + (index === 0 ? ' active' : '');
        tab.textContent = well;
        tab.addEventListener('click', () => switchTab(well));
        resultsTabs.appendChild(tab);
    });

    // Create content for each well
    Object.entries(results).forEach(([well, result], index) => {
        const wellResults = document.createElement('div');
        wellResults.className = 'well-results' + (index === 0 ? ' active' : '');
        wellResults.id = `results-${well}`;
        wellResults.setAttribute('data-well-name', well); // For print styling

        // Linear Regression Results
        wellResults.innerHTML += `
            <div class="results-section">
                <h3>Linear Regression Statistics</h3>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Variable</th>
                                <th>Coefficient</th>
                                <th>T-Value</th>
                                <th>Significant<br>(|t| ≥ 1.96)</th>
                            </tr>
                        </thead>
                        <tbody id="linear-table-${well}"></tbody>
                    </table>
                </div>
            </div>
        `;

        // Populate linear regression table
        const linearTable = wellResults.querySelector(`#linear-table-${well}`);
        result.linear_regression.forEach(row => {
            const tr = document.createElement('tr');
            // Check if significant (|t-value| >= 1.96)
            const tValue = parseFloat(row['T-Value']) || 0;
            const isSignificant = Math.abs(tValue) >= 1.96;
            const significanceClass = isSignificant ? 'significant-yes' : 'significant-no';
            const significanceText = isSignificant ? '✓ Yes' : '✗ No';

            tr.innerHTML = `
                <td>${row.Variable}</td>
                <td>${row.Coefficient}</td>
                <td>${row['T-Value']}</td>
                <td class="${significanceClass}">${significanceText}</td>
            `;
            linearTable.appendChild(tr);
        });

        // Decision Tree
        if (result.tree_image_path) {
            wellResults.innerHTML += `
                <div class="results-section">
                    <h3>Decision Tree</h3>
                    <div class="tree-image-container">
                        <img src="/api/tree-image/${well}" alt="Decision Tree for ${well}" />
                    </div>
                </div>
            `;
        } else {
            wellResults.innerHTML += `
                <div class="results-section">
                    <h3>Decision Tree</h3>
                    <div class="status-message info">
                        Tree visualization not available. Please ensure Graphviz is installed on the server.
                    </div>
                </div>
            `;
        }

        // Algorithm Explanation - Specific to this rule
        const branchConditions = result.branch_conditions || '';
        const branchFeatures = result.branch_features || [];
        const significantFeatures = result.significant_features || [];

        // Count features from tree vs regression
        const featuresFromTree = branchFeatures ? branchFeatures.length : 0;
        const featuresFromRegression = significantFeatures ? significantFeatures.filter(f => {
            const featureName = typeof f === 'object' ? f.feature : f;
            return !branchFeatures.includes(featureName);
        }).length : 0;

        // Get feature names for display
        const branchFeatureNames = branchFeatures.map(f => typeof f === 'string' ? f : f.feature || f).join(', ');
        const significantFeatureNames = significantFeatures.map(f => {
            const featureName = typeof f === 'object' ? f.feature : f;
            return featureName;
        }).join(', ');
        const regressionOnlyFeatures = significantFeatures.filter(f => {
            const featureName = typeof f === 'object' ? f.feature : f;
            return !branchFeatures.includes(featureName);
        }).map(f => {
            const featureName = typeof f === 'object' ? f.feature : f;
            return featureName;
        }).join(', ');

        wellResults.innerHTML += `
            <div class="results-section">
                <h3>Rule Generation Logic</h3>
                <div class="algorithm-explanation">
                    <div class="algorithm-step">
                        <h4>Decision Tree Path</h4>
                        <p><strong>Optimal branch:</strong> ${branchConditions || 'N/A'}</p>
                        <p>This path identified <strong>${featuresFromTree}</strong> feature${featuresFromTree !== 1 ? 's' : ''} with threshold values from the tree.</p>
                        ${featuresFromTree > 0 ? `<p class="feature-list"><strong>Features:</strong> ${branchFeatureNames}</p>` : ''}
                    </div>
                    <div class="algorithm-step">
                        <h4>Linear Regression</h4>
                        <p><strong>${significantFeatures ? significantFeatures.length : 0}</strong> significant feature${significantFeatures && significantFeatures.length !== 1 ? 's' : ''} (|t-value| ≥ 1.96).</p>
                        ${significantFeatures && significantFeatures.length > 0 ? `<p class="feature-list"><strong>Features:</strong> ${significantFeatureNames}</p>` : ''}
                        ${featuresFromRegression > 0 ? `<p><strong>${featuresFromRegression}</strong> feature${featuresFromRegression !== 1 ? 's' : ''} not in tree branch were set using regression coefficients.</p><p class="feature-list"><strong>Features:</strong> ${regressionOnlyFeatures}</p>` : '<p>All features were defined by the decision tree branch.</p>'}
                    </div>
                </div>
            </div>
        `;

        // Target Rule (all wells in Python format)
        wellResults.innerHTML += `
            <div class="results-section target-rule-section">
                <h3>Target Rule</h3>
                <div class="target-rule-display">
                    <div id="rule-display-${well}"></div>
                </div>
            </div>
        `;

        // Display all wells' rules in Python format with current well highlighted
        const ruleDisplay = wellResults.querySelector(`#rule-display-${well}`);
        ruleDisplay.innerHTML = buildPythonRulesFormat(results, well);

        resultsContent.appendChild(wellResults);
    });
}

function buildPythonRulesFormat(results, currentWell = null) {
    // Build a clean, readable format for display with HTML highlighting
    let displayHTML = "";

    // Sort wells for consistent output
    const sortedWells = Object.keys(results).sort();

    sortedWells.forEach((well) => {
        const result = results[well];
        const ruleList = result.rule_list || [];
        const ruleArray = ruleList.join(', ');
        const isCurrentWell = well === currentWell;
        if (isCurrentWell) {
            displayHTML += `<strong class="current-well-rule">${well}: [${ruleArray}]</strong>\n`;
        } else {
            displayHTML += `${well}: [${ruleArray}]\n`;
        }
    });

    return displayHTML.trim();
}

function switchTab(well) {
    // Update tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
        if (tab.textContent === well) {
            tab.classList.add('active');
        }
    });

    // Update content
    document.querySelectorAll('.well-results').forEach(results => {
        results.classList.remove('active');
        if (results.id === `results-${well}`) {
            results.classList.add('active');
        }
    });
}

async function downloadRules() {
    try {
        const response = await fetch('/api/download-rules');

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'target_rules.py';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } else {
            const data = await response.json();
            alert('Error downloading rules: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error downloading rules: ' + error.message);
    }
}

function showStatus(elementId, message, type) {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.className = 'status-message ' + type;
    element.style.display = 'block';
}
