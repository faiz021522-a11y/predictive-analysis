document.addEventListener('DOMContentLoaded', () => {
    // ----------------- STATE MANAGEMENT -----------------
    const state = {
        datasets: [],
        selectedDataset: '',
        datasetDetails: null,
        charts: {
            regScatter: null,
            regImportance: null,
            regResidual: null,
            tsForecast: null
        }
    };

    // ----------------- DOM ELEMENTS -----------------
    // Sidebar Controls
    const datasetSelect = document.getElementById('dataset-select');
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const taskSelect = document.getElementById('task-select');
    const scaleFeaturesCheckbox = document.getElementById('scale-features');
    const imputeStrategySelect = document.getElementById('impute-strategy');
    const trainBtn = document.getElementById('train-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');

    // Regression Settings
    const regressionConfigs = document.getElementById('regression-configs');
    const regModelSelect = document.getElementById('reg-model-select');
    const regTargetSelect = document.getElementById('reg-target-select');
    const regFeaturesList = document.getElementById('reg-features-list');
    const ridgeAlphaGroup = document.getElementById('ridge-alpha-group');
    const ridgeAlphaSlider = document.getElementById('ridge-alpha');
    const ridgeAlphaVal = document.getElementById('ridge-alpha-val');
    const rfEstimatorsGroup = document.getElementById('rf-estimators-group');
    const rfEstimatorsSlider = document.getElementById('rf-estimators');
    const rfEstimatorsVal = document.getElementById('rf-estimators-val');

    // Time Series Settings
    const timeseries_configs = document.getElementById('timeseries-configs');
    const tsModelSelect = document.getElementById('ts-model-select');
    const tsDateSelect = document.getElementById('ts-date-select');
    const tsTargetSelect = document.getElementById('ts-target-select');
    const tsForecastHorizonInput = document.getElementById('ts-forecast-horizon');
    const arimaParams = document.getElementById('arima-params');
    const hwParams = document.getElementById('hw-params');
    const hwTrendSelect = document.getElementById('hw-trend');
    const hwSeasonalSelect = document.getElementById('hw-seasonal');
    const hwSpGroup = document.getElementById('hw-sp-group');
    const hwSpInput = document.getElementById('hw-sp');
    const smaParams = document.getElementById('sma-params');
    const smaWindowInput = document.getElementById('sma-window');

    // Tabs & Views
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const tabRegBtn = document.getElementById('tab-reg-btn');
    const tabTsBtn = document.getElementById('tab-ts-btn');

    // Explorer Metrics & Tables
    const statRows = document.getElementById('stat-rows');
    const statCols = document.getElementById('stat-cols');
    const statMissing = document.getElementById('stat-missing');
    const schemaTable = document.getElementById('schema-table').querySelector('tbody');
    const previewTable = document.getElementById('preview-table');

    // Regression Results Panel
    const regBadgeModel = document.getElementById('reg-badge-model');
    const regBadgeDataset = document.getElementById('reg-badge-dataset');
    const metricR2 = document.getElementById('metric-r2');
    const metricR2Train = document.getElementById('metric-r2-train');
    const metricMae = document.getElementById('metric-mae');
    const metricMaeTrain = document.getElementById('metric-mae-train');
    const metricRmse = document.getElementById('metric-rmse');
    const metricRmseTrain = document.getElementById('metric-rmse-train');

    // Time Series Results Panel
    const tsBadgeModel = document.getElementById('ts-badge-model');
    const tsBadgeDataset = document.getElementById('ts-badge-dataset');
    const tsMetricMape = document.getElementById('ts-metric-mape');
    const tsMetricMae = document.getElementById('ts-metric-mae');
    const tsMetricRmse = document.getElementById('ts-metric-rmse');

    // ----------------- GLOBAL CHART.JS DEFAULTS -----------------
    Chart.defaults.color = '#94a3b8'; // Slate Muted
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
    Chart.defaults.font.family = "'Inter', sans-serif";

    // ----------------- INITIALIZATION -----------------
    fetchDatasets();

    // Setup slider label value sync
    ridgeAlphaSlider.addEventListener('input', (e) => ridgeAlphaVal.textContent = e.target.value);
    rfEstimatorsSlider.addEventListener('input', (e) => rfEstimatorsVal.textContent = e.target.value);

    // Tab Switcher
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            switchTab(targetTab);
        });
    });

    // ----------------- EVENT LISTENERS -----------------
    // Task Type Switch
    taskSelect.addEventListener('change', () => {
        const task = taskSelect.value;
        if (task === 'regression') {
            regressionConfigs.style.display = 'block';
            timeseries_configs.style.display = 'none';
        } else {
            regressionConfigs.style.display = 'none';
            timeseries_configs.style.display = 'block';
        }
        updateFormValidation();
    });

    // Regression Algorithm Switch
    regModelSelect.addEventListener('change', () => {
        const model = regModelSelect.value;
        ridgeAlphaGroup.style.display = (model === 'ridge') ? 'block' : 'none';
        rfEstimatorsGroup.style.display = (model === 'random_forest') ? 'block' : 'none';
    });

    // Time Series Algorithm Switch
    tsModelSelect.addEventListener('change', () => {
        const model = tsModelSelect.value;
        arimaParams.style.display = (model === 'arima') ? 'block' : 'none';
        hwParams.style.display = (model === 'exponential_smoothing') ? 'block' : 'none';
        smaParams.style.display = (model === 'moving_average') ? 'block' : 'none';
    });

    hwSeasonalSelect.addEventListener('change', () => {
        const seasonal = hwSeasonalSelect.value;
        hwSpGroup.style.display = (seasonal !== 'none') ? 'block' : 'none';
    });

    // Dataset selection change
    datasetSelect.addEventListener('change', () => {
        state.selectedDataset = datasetSelect.value;
        loadDatasetDetails(state.selectedDataset);
    });

    // File Upload Handlers
    uploadZone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            uploadFile(fileInput.files[0]);
        }
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    // Change target selectors validation
    regTargetSelect.addEventListener('change', () => {
        populateFeaturesList();
        updateFormValidation();
    });

    tsDateSelect.addEventListener('change', updateFormValidation);
    tsTargetSelect.addEventListener('change', updateFormValidation);

    // Model Training Trigger
    trainBtn.addEventListener('click', () => {
        if (taskSelect.value === 'regression') {
            trainRegressionModel();
        } else {
            trainTimeSeriesModel();
        }
    });


    // ----------------- FUNCTIONS -----------------

    function switchTab(tabId) {
        tabBtns.forEach(b => {
            if (b.getAttribute('data-tab') === tabId) b.classList.add('active');
            else b.classList.remove('active');
        });
        tabContents.forEach(c => {
            if (c.getAttribute('id') === tabId) c.classList.add('active');
            else c.classList.remove('active');
        });
    }

    // Fetch list of datasets
    async function fetchDatasets(selectedName = null) {
        try {
            const res = await fetch('/api/datasets');
            const data = await res.json();
            state.datasets = data.datasets;
            
            // Populate select options
            datasetSelect.innerHTML = '<option value="" disabled selected>Select a dataset...</option>';
            state.datasets.forEach(db => {
                const opt = document.createElement('option');
                opt.value = db;
                opt.textContent = db;
                datasetSelect.appendChild(opt);
            });

            if (selectedName && state.datasets.includes(selectedName)) {
                datasetSelect.value = selectedName;
                state.selectedDataset = selectedName;
                loadDatasetDetails(selectedName);
            } else if (state.datasets.length > 0) {
                // Default select the first dataset
                datasetSelect.value = state.datasets[0];
                state.selectedDataset = state.datasets[0];
                loadDatasetDetails(state.datasets[0]);
            }
        } catch (err) {
            console.error('Error fetching datasets:', err);
            datasetSelect.innerHTML = '<option value="" disabled>Failed to load datasets.</option>';
        }
    }

    // Upload new CSV
    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        showLoading('Uploading file...');
        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Upload failed');
            
            // Reload dataset list and select the newly uploaded file
            await fetchDatasets(data.filename);
        } catch (err) {
            alert(`Error: ${err.message}`);
        } finally {
            hideLoading();
        }
    }

    // Load details of selected dataset
    async function loadDatasetDetails(filename) {
        showLoading('Loading dataset schema...');
        try {
            const res = await fetch(`/api/datasets/${filename}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail);
            
            state.datasetDetails = data;
            
            // Update Data Summary stats cards
            statRows.textContent = data.shape[0].toLocaleString();
            statCols.textContent = data.shape[1].toLocaleString();
            
            const totalNulls = data.columns.reduce((acc, col) => acc + col.null_count, 0);
            statMissing.textContent = totalNulls.toLocaleString();

            // Populate Schema Table
            schemaTable.innerHTML = '';
            data.columns.forEach(col => {
                const row = document.createElement('tr');
                
                const hasStats = col.stats && Object.keys(col.stats).length > 0;
                const mean = hasStats && col.stats.mean !== null ? col.stats.mean.toFixed(2) : '-';
                const min = hasStats && col.stats.min !== null ? col.stats.min.toFixed(2) : '-';
                const max = hasStats && col.stats.max !== null ? col.stats.max.toFixed(2) : '-';

                row.innerHTML = `
                    <td><strong>${col.name}</strong></td>
                    <td><span class="badge secondary">${col.type}</span></td>
                    <td class="${col.null_count > 0 ? 'text-coral' : ''}">${col.null_count}</td>
                    <td>${col.unique_count}</td>
                    <td>${mean}</td>
                    <td>${min}</td>
                    <td>${max}</td>
                `;
                schemaTable.appendChild(row);
            });

            // Populate Preview Table
            renderPreviewTable(data.preview, data.columns.map(c => c.name));

            // Populate Select Dropdowns in sidebar
            populateSelectors(data.columns);

            updateFormValidation();
            switchTab('data-explorer');
        } catch (err) {
            alert(`Error loading dataset: ${err.message}`);
        } finally {
            hideLoading();
        }
    }

    // Render columns preview
    function renderPreviewTable(records, colNames) {
        previewTable.innerHTML = '';
        
        // Header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        colNames.forEach(name => {
            const th = document.createElement('th');
            th.textContent = name;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        previewTable.appendChild(thead);

        // Body
        const tbody = document.createElement('tbody');
        if (records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="${colNames.length}" class="text-center">No data available.</td></tr>`;
        } else {
            records.forEach(rec => {
                const tr = document.createElement('tr');
                colNames.forEach(name => {
                    const td = document.createElement('td');
                    td.textContent = rec[name] !== null ? rec[name] : 'NaN';
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
        }
        previewTable.appendChild(tbody);
    }

    // Populate variable selectors based on schema
    function populateSelectors(columns) {
        // Clear old choices
        regTargetSelect.innerHTML = '<option value="" disabled selected>Select target (y)...</option>';
        tsDateSelect.innerHTML = '<option value="" disabled selected>Select Date...</option>';
        tsTargetSelect.innerHTML = '<option value="" disabled selected>Select target...</option>';

        columns.forEach(col => {
            // Regression target: numerical variables are typical targets, but allow anything
            const optReg = document.createElement('option');
            optReg.value = col.name;
            optReg.textContent = col.name;
            regTargetSelect.appendChild(optReg);

            // TS Date selector: look for dates or strings
            const optDate = document.createElement('option');
            optDate.value = col.name;
            optDate.textContent = col.name;
            tsDateSelect.appendChild(optDate);

            // TS Value selector: numerical columns
            const optVal = document.createElement('option');
            optVal.value = col.name;
            optVal.textContent = col.name;
            tsTargetSelect.appendChild(optVal);
        });

        // Set smart defaults
        // For regression (House Price)
        const possibleRegTargets = ['PriceUSD', 'price', 'sales', 'Sales'];
        const matchedRegTarget = columns.find(c => possibleRegTargets.includes(c.name));
        if (matchedRegTarget) {
            regTargetSelect.value = matchedRegTarget.name;
        }

        // For TS (Sales)
        const possibleDates = ['Month', 'Date', 'date', 'month', 'YearMonth', 'timestamp'];
        const matchedDate = columns.find(c => possibleDates.includes(c.name));
        if (matchedDate) {
            tsDateSelect.value = matchedDate.name;
        }

        const possibleVal = ['Sales', 'sales', 'value', 'Value', 'PriceUSD', 'Price'];
        const matchedVal = columns.find(c => possibleVal.includes(c.name) && c.name !== tsDateSelect.value);
        if (matchedVal) {
            tsTargetSelect.value = matchedVal.name;
        }

        // Trigger features list creation
        populateFeaturesList();
    }

    // Features checkbox checklist
    function populateFeaturesList() {
        regFeaturesList.innerHTML = '';
        if (!state.datasetDetails) return;

        const target = regTargetSelect.value;
        
        state.datasetDetails.columns.forEach(col => {
            if (col.name === target) return; // exclude target
            
            const item = document.createElement('div');
            item.className = 'features-item';
            item.innerHTML = `
                <label class="checkbox-container">
                    <input type="checkbox" name="features" value="${col.name}" checked>
                    <span class="checkmark"></span>
                    ${col.name} <span style="font-size:0.75rem; color:#94a3b8; margin-left:4px;">(${col.type})</span>
                </label>
            `;
            regFeaturesList.appendChild(item);

            // Add event listeners to checklist checkboxes
            item.querySelector('input').addEventListener('change', updateFormValidation);
        });
    }

    // Enable/Disable Train button
    function updateFormValidation() {
        let isValid = false;
        
        if (state.selectedDataset) {
            const task = taskSelect.value;
            if (task === 'regression') {
                const target = regTargetSelect.value;
                const featuresChecked = getSelectedFeatures().length > 0;
                isValid = (target && featuresChecked);
            } else {
                const dateCol = tsDateSelect.value;
                const targetCol = tsTargetSelect.value;
                isValid = (dateCol && targetCol && dateCol !== targetCol);
            }
        }

        trainBtn.disabled = !isValid;
    }

    function getSelectedFeatures() {
        const checked = regFeaturesList.querySelectorAll('input[name="features"]:checked');
        return Array.from(checked).map(cb => cb.value);
    }

    // Show/Hide spinner loading screen
    function showLoading(text) {
        loadingText.textContent = text;
        loadingOverlay.classList.add('active');
    }

    function hideLoading() {
        loadingOverlay.classList.remove('active');
    }

    // ----------------- MODEL TRAINING - REGRESSION -----------------
    async function trainRegressionModel() {
        showLoading('Training regression models, generating performance analytics...');
        
        const target = regTargetSelect.value;
        const features = getSelectedFeatures();
        
        // Assemble imputation strategies
        const imputation_strategies = {};
        state.datasetDetails.columns.forEach(col => {
            imputation_strategies[col.name] = imputeStrategySelect.value;
        });

        // Assemble hyperparameters
        const hyperparameters = {};
        if (regModelSelect.value === 'ridge') {
            hyperparameters.alpha = parseFloat(ridgeAlphaSlider.value);
        } else if (regModelSelect.value === 'random_forest') {
            hyperparameters.n_estimators = parseInt(rfEstimatorsSlider.value);
        }

        const payload = {
            dataset_name: state.selectedDataset,
            target_column: target,
            feature_columns: features,
            imputation_strategies: imputation_strategies,
            scale_features: scaleFeaturesCheckbox.checked,
            scaler_type: 'standard',
            model_type: regModelSelect.value,
            test_size: 0.2,
            hyperparameters: hyperparameters
        };

        try {
            const res = await fetch('/api/train/regression', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if (!res.ok) throw new Error(data.detail || 'Training failed.');

            renderRegressionResults(data, payload);
            switchTab('regression-results');
        } catch (err) {
            alert(`Error fitting regression model: ${err.message}`);
        } finally {
            hideLoading();
        }
    }

    function renderRegressionResults(data, config) {
        // Set badges
        regBadgeModel.textContent = config.model_type.replace('_', ' ');
        regBadgeDataset.textContent = config.dataset_name;

        // Set metrics text
        metricR2.textContent = data.metrics.test.r2.toFixed(4);
        metricR2Train.textContent = `Train: ${data.metrics.train.r2.toFixed(4)}`;
        
        metricMae.textContent = data.metrics.test.mae.toLocaleString(undefined, {maximumFractionDigits:2});
        metricMaeTrain.textContent = `Train: ${data.metrics.train.mae.toLocaleString(undefined, {maximumFractionDigits:2})}`;
        
        metricRmse.textContent = data.metrics.test.rmse.toLocaleString(undefined, {maximumFractionDigits:2});
        metricRmseTrain.textContent = `Train: ${data.metrics.train.rmse.toLocaleString(undefined, {maximumFractionDigits:2})}`;

        // 1. SCATTER PLOT (Actual vs Predicted)
        const actVal = data.predictions.map(p => p.actual);
        const predVal = data.predictions.map(p => p.predicted);
        
        const minVal = Math.min(...actVal, ...predVal) * 0.95;
        const maxVal = Math.max(...actVal, ...predVal) * 1.05;

        // Perfect prediction line
        const identityLine = [
            { x: minVal, y: minVal },
            { x: maxVal, y: maxVal }
        ];

        const scatterPoints = data.predictions.map(p => ({ x: p.actual, y: p.predicted }));

        if (state.charts.regScatter) state.charts.regScatter.destroy();
        
        const ctxScatter = document.getElementById('reg-scatter-chart').getContext('2d');
        state.charts.regScatter = new Chart(ctxScatter, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Predictions',
                        data: scatterPoints,
                        backgroundColor: 'rgba(59, 130, 246, 0.7)',
                        borderColor: '#3b82f6',
                        borderWidth: 1.5,
                        pointRadius: 5,
                        pointHoverRadius: 7
                    },
                    {
                        label: 'Perfect Fit',
                        data: identityLine,
                        type: 'line',
                        borderColor: 'rgba(244, 63, 94, 0.6)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: (context) => `Actual: ${context.raw.x.toFixed(2)}, Pred: ${context.raw.y.toFixed(2)}`
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Actual Values' },
                        min: minVal,
                        max: maxVal
                    },
                    y: {
                        title: { display: true, text: 'Predicted Values' },
                        min: minVal,
                        max: maxVal
                    }
                }
            }
        });

        // 2. FEATURE IMPORTANCE CHART
        const featLabels = data.feature_importances.map(f => f.feature);
        const featImportance = data.feature_importances.map(f => f.importance);
        
        // Color mapping depending on coefficient sign or importance
        const backgroundColors = featImportance.map(val => 
            val >= 0 ? 'rgba(168, 85, 247, 0.7)' : 'rgba(244, 63, 94, 0.7)'
        );
        const borderColors = featImportance.map(val => 
            val >= 0 ? '#a855f7' : '#f43f5e'
        );

        if (state.charts.regImportance) state.charts.regImportance.destroy();
        
        const ctxImp = document.getElementById('reg-importance-chart').getContext('2d');
        state.charts.regImportance = new Chart(ctxImp, {
            type: 'bar',
            data: {
                labels: featLabels,
                datasets: [{
                    label: config.model_type.includes('linear') || config.model_type.includes('ridge') ? 'Coefficients' : 'Importance Metric',
                    data: featImportance,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Impact / Importance value' }
                    }
                }
            }
        });

        // 3. RESIDUALS HISTOGRAM
        const hist = calculateHistogram(data.residuals, 12);
        
        if (state.charts.regResidual) state.charts.regResidual.destroy();
        
        const ctxRes = document.getElementById('reg-residual-chart').getContext('2d');
        state.charts.regResidual = new Chart(ctxRes, {
            type: 'bar',
            data: {
                labels: hist.labels,
                datasets: [{
                    label: 'Error Frequency',
                    data: hist.data,
                    backgroundColor: 'rgba(16, 185, 129, 0.6)',
                    borderColor: '#10b981',
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Prediction Error Range (Residuals)' }
                    },
                    y: {
                        title: { display: true, text: 'Count' },
                        ticks: { precision: 0 }
                    }
                }
            }
        });
    }

    // Helper to compute bins for histogram
    function calculateHistogram(values, binCount = 10) {
        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min;
        const binWidth = range / binCount;
        const bins = Array(binCount).fill(0);
        const labels = [];
        
        for (let i = 0; i < binCount; i++) {
            const binStart = min + i * binWidth;
            const binEnd = binStart + binWidth;
            labels.push(`${binStart.toFixed(1)} to ${binEnd.toFixed(1)}`);
        }
        
        values.forEach(val => {
            let binIdx = Math.floor((val - min) / binWidth);
            if (binIdx >= binCount) binIdx = binCount - 1;
            if (binIdx < 0) binIdx = 0;
            bins[binIdx]++;
        });
        
        return { labels, data: bins };
    }

    // ----------------- MODEL TRAINING - TIME SERIES -----------------
    async function trainTimeSeriesModel() {
        showLoading('Fitting time-series forecasting model...');

        const dateCol = tsDateSelect.value;
        const targetCol = tsTargetSelect.value;

        // Assemble hyperparameters
        const hyperparameters = {};
        const model = tsModelSelect.value;
        
        if (model === 'arima') {
            hyperparameters.p = parseInt(document.getElementById('arima-p').value) || 1;
            hyperparameters.d = parseInt(document.getElementById('arima-d').value) || 1;
            hyperparameters.q = parseInt(document.getElementById('arima-q').value) || 1;
        } else if (model === 'exponential_smoothing') {
            hyperparameters.trend = hwTrendSelect.value === 'none' ? null : hwTrendSelect.value;
            hyperparameters.seasonal = hwSeasonalSelect.value === 'none' ? null : hwSeasonalSelect.value;
            if (hwSeasonalSelect.value !== 'none') {
                hyperparameters.seasonal_periods = parseInt(hwSpInput.value) || 12;
            }
        } else if (model === 'moving_average') {
            hyperparameters.window = parseInt(smaWindowInput.value) || 3;
        }

        const horizon = parseInt(tsForecastHorizonInput.value) || 12;

        const payload = {
            dataset_name: state.selectedDataset,
            date_column: dateCol,
            target_column: targetCol,
            model_type: model,
            test_periods: 12, // 12 periods test validation
            forecast_horizon: horizon,
            hyperparameters: hyperparameters
        };

        try {
            const res = await fetch('/api/train/timeseries', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if (!res.ok) throw new Error(data.detail || 'Forecasting failed.');

            renderTimeSeriesResults(data, payload);
            switchTab('timeseries-results');
        } catch (err) {
            alert(`Error fitting forecasting model: ${err.message}`);
        } finally {
            hideLoading();
        }
    }

    function renderTimeSeriesResults(data, config) {
        // Set badges
        tsBadgeModel.textContent = config.model_type.replace('_', ' ');
        tsBadgeDataset.textContent = config.dataset_name;

        // Metrics
        tsMetricMape.textContent = `${data.metrics.mape.toFixed(2)}%`;
        tsMetricMae.textContent = data.metrics.mae.toLocaleString(undefined, {maximumFractionDigits: 1});
        tsMetricRmse.textContent = data.metrics.rmse.toLocaleString(undefined, {maximumFractionDigits: 1});

        // Setup Charts
        // We will combine Historical, Fitted values, Test Predictions, and Out-of-sample Forecast on a single line chart.
        const histDates = data.historical.map(h => h.date);
        const histValues = data.historical.map(h => h.value);

        // Setup aligned arrays of fitted, test pred, and future forecasts
        const fittedMap = {};
        data.fitted.forEach(f => fittedMap[f.date] = f.value);

        const predMap = {};
        data.predictions.forEach(p => predMap[p.date] = p.value);

        // Combine labels
        const allDatesSet = new Set(histDates);
        data.forecast.forEach(f => allDatesSet.add(f.date));
        const allDates = Array.from(allDatesSet).sort();

        // Build data arrays aligned with combined date list
        const chartHist = [];
        const chartFitted = [];
        const chartPreds = [];
        const chartForecast = [];

        // Setup forecasts start point: link forecast line to the last historical point
        let lastHistVal = null;
        let lastHistDate = null;
        if (data.historical.length > 0) {
            const last = data.historical[data.historical.length - 1];
            lastHistVal = last.value;
            lastHistDate = last.date;
        }

        allDates.forEach(date => {
            const isHistorical = histDates.includes(date);
            
            chartHist.push(isHistorical ? histValues[histDates.indexOf(date)] : null);
            chartFitted.push(fittedMap[date] !== undefined ? fittedMap[date] : null);
            chartPreds.push(predMap[date] !== undefined ? predMap[date] : null);
            
            // Forecast line
            const forecastItem = data.forecast.find(f => f.date === date);
            if (forecastItem) {
                chartForecast.push(forecastItem.value);
            } else if (date === lastHistDate) {
                // Connect the forecast line to history
                chartForecast.push(lastHistVal);
            } else {
                chartForecast.push(null);
            }
        });

        if (state.charts.tsForecast) state.charts.tsForecast.destroy();

        const ctxForecast = document.getElementById('ts-forecast-chart').getContext('2d');
        state.charts.tsForecast = new Chart(ctxForecast, {
            type: 'line',
            data: {
                labels: allDates,
                datasets: [
                    {
                        label: 'Historical Data',
                        data: chartHist,
                        borderColor: '#94a3b8',
                        borderWidth: 2,
                        pointRadius: 2,
                        fill: false
                    },
                    {
                        label: 'Model Fit (Training)',
                        data: chartFitted,
                        borderColor: 'rgba(168, 85, 247, 0.6)',
                        borderWidth: 1.5,
                        borderDash: [3, 3],
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: 'Test Validation predictions',
                        data: chartPreds,
                        borderColor: 'rgba(59, 130, 246, 0.8)',
                        borderWidth: 2,
                        pointRadius: 3,
                        fill: false
                    },
                    {
                        label: 'Future Forecast Projection',
                        data: chartForecast,
                        borderColor: '#10b981',
                        borderWidth: 3,
                        pointRadius: 4,
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true, position: 'top' }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Timeline' }
                    },
                    y: {
                        title: { display: true, text: config.target_column }
                    }
                }
            }
        });
    }

});
