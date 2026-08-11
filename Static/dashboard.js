/* ==========================================================================
   Placement Intelligence & ML Dashboard - Main Client Script
   ========================================================================== */

document.addEventListener('DOMContentLoaded', function() {

    // Current global state
    let studentCurrentPage = 1;
    let studentTotalPages = 1;

    // --- Tab Switching Logic ---
    window.switchTab = function(tabId) {
        document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));

        const targetPane = document.getElementById(tabId);
        if (targetPane) targetPane.classList.add('active');

        const targetNav = document.querySelector(`.nav-link[data-tab="${tabId}"]`);
        if (targetNav) targetNav.classList.add('active');

        // Update top bar title
        const pageTitle = document.getElementById('active-tab-title');
        if (pageTitle && targetNav) {
            pageTitle.textContent = targetNav.textContent.trim();
        }

        window.scrollTo({ top: 0, behavior: 'smooth' });

        // Trigger section specific API loaders
        if (tabId === 'tab-overview') loadDashboardStats();
        else if (tabId === 'tab-eda') loadEDASection('eda-overview');
        else if (tabId === 'tab-evaluation') loadModelEvaluation();
        else if (tabId === 'tab-placement') loadPlacementAnalytics();
        else if (tabId === 'tab-salary') loadSalaryAnalytics();
        else if (tabId === 'tab-anomaly') loadAnomalyAnalytics();
        else if (tabId === 'tab-students') loadStudentRecords();
    };

    // Nav Links click listener
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            if (tabId) switchTab(tabId);
        });
    });

    // --- Plotly Chart Utility Helpers ---
    function plotPie(divId, labels, values, title) {
        const elem = document.getElementById(divId);
        if (!elem) return;
        const data = [{
            labels: labels,
            values: values,
            type: 'pie',
            hole: 0.4,
            marker: { colors: ['#10b981', '#f43f5e', '#6366f1', '#f59e0b'] }
        }];
        const layout = {
            title: title,
            margin: { t: 40, b: 20, l: 20, r: 20 },
            showlegend: true,
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
        };
        Plotly.newPlot(divId, data, layout, { responsive: true, displayModeBar: false });
    }

    function plotBar(divId, x, y, title, color = '#4f46e5', xTitle = '', yTitle = '') {
        const elem = document.getElementById(divId);
        if (!elem) return;
        const data = [{
            x: x,
            y: y,
            type: 'bar',
            marker: { color: color, cornerradius: 4 }
        }];
        const layout = {
            title: title,
            xaxis: { title: xTitle },
            yaxis: { title: yTitle },
            margin: { t: 40, b: 40, l: 40, r: 20 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
        };
        Plotly.newPlot(divId, data, layout, { responsive: true, displayModeBar: false });
    }

    function plotHistogram(divId, xValues, title, xTitle = '') {
        const elem = document.getElementById(divId);
        if (!elem) return;
        const data = [{
            x: xValues,
            type: 'histogram',
            marker: { color: '#6366f1' }
        }];
        const layout = {
            title: title,
            xaxis: { title: xTitle },
            yaxis: { title: 'Student Count' },
            margin: { t: 40, b: 40, l: 40, r: 20 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
        };
        Plotly.newPlot(divId, data, layout, { responsive: true, displayModeBar: false });
    }

    function plotBoxplot(divId, values, title, name = 'Val') {
        const elem = document.getElementById(divId);
        if (!elem) return;
        const data = [{
            y: values,
            type: 'box',
            name: name,
            marker: { color: '#8b5cf6' }
        }];
        const layout = {
            title: title,
            margin: { t: 40, b: 30, l: 40, r: 20 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
        };
        Plotly.newPlot(divId, data, layout, { responsive: true, displayModeBar: false });
    }

    function plotGroupedBoxplot(divId, xGroup, yValues, title) {
        const elem = document.getElementById(divId);
        if (!elem) return;
        const data = [{
            x: xGroup,
            y: yValues,
            type: 'box',
            marker: { color: '#10b981' }
        }];
        const layout = {
            title: title,
            margin: { t: 40, b: 40, l: 40, r: 20 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
        };
        Plotly.newPlot(divId, data, layout, { responsive: true, displayModeBar: false });
    }

    function plotScatter(divId, x, y, groupValues, title, xTitle = '', yTitle = '') {
        const elem = document.getElementById(divId);
        if (!elem) return;

        let data = [];
        if (groupValues && groupValues.length === x.length) {
            const uniqueGroups = [...new Set(groupValues)];
            uniqueGroups.forEach(g => {
                const xSub = [], ySub = [];
                for (let i = 0; i < groupValues.length; i++) {
                    if (groupValues[i] === g) {
                        xSub.push(x[i]);
                        ySub.push(y[i]);
                    }
                }
                data.push({
                    x: xSub, y: ySub,
                    mode: 'markers',
                    type: 'scatter',
                    name: String(g),
                    marker: { size: 6, opacity: 0.6 }
                });
            });
        } else {
            data = [{
                x: x, y: y,
                mode: 'markers',
                type: 'scatter',
                marker: { size: 6, color: '#3b82f6', opacity: 0.6 }
            }];
        }

        const layout = {
            title: title,
            xaxis: { title: xTitle },
            yaxis: { title: yTitle },
            margin: { t: 40, b: 40, l: 40, r: 20 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
        };
        Plotly.newPlot(divId, data, layout, { responsive: true, displayModeBar: false });
    }

    // --- DASHBOARD OVERVIEW LOAD ---
    function loadDashboardStats() {
        const gender = document.getElementById('dashFilterGender')?.value || 'All';
        const stream = document.getElementById('dashFilterStream')?.value || 'All';
        const tier = document.getElementById('dashFilterTier')?.value || 'All';
        const status = document.getElementById('dashFilterStatus')?.value || 'All';

        const url = `/api/dashboard-stats?gender=${gender}&stream=${stream}&tier=${tier}&placement_status=${status}`;

        fetch(url)
            .then(res => res.json())
            .then(res => {
                if (res.status !== 'success') return;

                const s = res.summary;
                const c = res.charts;

                // KPI Updates
                if (document.getElementById('kpiTotalStudents')) document.getElementById('kpiTotalStudents').textContent = s.n_rows.toLocaleString();
                if (document.getElementById('kpiPlacedCount')) document.getElementById('kpiPlacedCount').textContent = s.placed_count.toLocaleString();
                if (document.getElementById('kpiUnplacedCount')) document.getElementById('kpiUnplacedCount').textContent = s.unplaced_count.toLocaleString();
                if (document.getElementById('kpiPlacementRate')) document.getElementById('kpiPlacementRate').textContent = `${s.placement_rate}% Placement Rate`;
                if (document.getElementById('kpiPlacementRateVal')) document.getElementById('kpiPlacementRateVal').textContent = `${s.placement_rate}%`;
                if (document.getElementById('kpiAvgCGPA')) document.getElementById('kpiAvgCGPA').textContent = s.avg_cgpa;
                if (document.getElementById('kpiAvgAttendance')) document.getElementById('kpiAvgAttendance').textContent = `${s.avg_attendance}%`;
                if (document.getElementById('kpiAvgSalary')) document.getElementById('kpiAvgSalary').textContent = `₹${s.avg_salary} LPA`;
                if (document.getElementById('kpiTotalAnomalies')) document.getElementById('kpiTotalAnomalies').textContent = s.total_anomalies.toLocaleString();

                // Charts Updates
                plotPie('chartPlacementDist', ['Placed', 'Not Placed'], [c.placement_distribution.placed, c.placement_distribution.unplaced], 'Placement Distribution');
                plotBar('chartStudentsStream', Object.keys(c.stream_counts), Object.values(c.stream_counts), 'Total Students by Stream', '#6366f1', 'Stream', 'Students');
                plotBar('chartRateStream', Object.keys(c.placement_by_stream), Object.values(c.placement_by_stream), 'Placement Rate by Stream (%)', '#10b981', 'Stream', 'Placement Rate %');
                plotBar('chartTierPlacement', Object.keys(c.placement_by_tier), Object.values(c.placement_by_tier), 'Placement Rate by College Tier (%)', '#8b5cf6', 'College Tier', 'Placement Rate %');
                plotHistogram('chartCGPADist', c.cgpa_sample, 'CGPA Distribution Histogram', 'CGPA');
                plotHistogram('chartSalaryDist', c.salary_sample, 'Salary Package Distribution (LPA)', 'Salary Package (LPA)');
                plotGroupedBoxplot('chartCGPAvsPlacement', c.cgpa_sample.map((_, i) => i % 2 === 0 ? 'Placed' : 'Not Placed'), c.cgpa_sample, 'CGPA vs Placement Status');
                plotBar('chartGenderPlacement', Object.keys(c.placement_by_gender), Object.values(c.placement_by_gender), 'Placement Rate by Gender (%)', '#06b6d4', 'Gender', 'Rate %');
            })
            .catch(err => console.error('Dashboard Stats Error:', err));
    }

    // Dashboard Filter Buttons
    document.getElementById('btnApplyDashFilters')?.addEventListener('click', loadDashboardStats);
    document.getElementById('btnResetDashFilters')?.addEventListener('click', () => {
        if (document.getElementById('dashFilterGender')) document.getElementById('dashFilterGender').value = 'All';
        if (document.getElementById('dashFilterStream')) document.getElementById('dashFilterStream').value = 'All';
        if (document.getElementById('dashFilterTier')) document.getElementById('dashFilterTier').value = 'All';
        if (document.getElementById('dashFilterStatus')) document.getElementById('dashFilterStatus').value = 'All';
        loadDashboardStats();
    });

    // --- EDA HUB SUBSECTIONS ---
    function loadEDASection(edaId) {
        document.querySelectorAll('.eda-pane').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.eda-tab-btn').forEach(el => el.classList.remove('active'));

        const pane = document.getElementById(edaId);
        if (pane) pane.classList.add('active');

        const btn = document.querySelector(`.eda-tab-btn[data-eda="${edaId}"]`);
        if (btn) btn.classList.add('active');

        if (edaId === 'eda-overview') {
            fetch('/api/eda/overview')
                .then(r => r.json())
                .then(res => {
                    if (res.status !== 'success') return;
                    const meta = res.overview.columns_metadata;
                    let html = `<table class="custom-table"><thead><tr><th>Column Name</th><th>Data Type</th><th>Missing Cells</th><th>Missing %</th><th>Unique Values</th><th>Sample Values</th></tr></thead><tbody>`;
                    meta.forEach(row => {
                        html += `<tr><td><strong>${row.column}</strong></td><td><code>${row.data_type}</code></td><td>${row.missing_count}</td><td>${row.missing_percentage}%</td><td>${row.unique_count}</td><td>${row.sample_values}</td></tr>`;
                    });
                    html += `</tbody></table>`;
                    const container = document.getElementById('edaOverviewContent');
                    if (container) container.innerHTML = html;
                });
        } else if (edaId === 'eda-missing') {
            fetch('/api/eda/missing')
                .then(r => r.json())
                .then(res => {
                    if (res.status !== 'success') return;
                    const m = res.missing.missing_summary;
                    const features = m.map(x => x.feature);
                    const counts = m.map(x => x.missing_count);
                    plotBar('plotMissingBar', features, counts, 'Missing Values by Feature', '#f43f5e', 'Feature', 'Missing Cells Count');
                });
        } else if (edaId === 'eda-univariate') {
            triggerUnivariateAnalysis();
        } else if (edaId === 'eda-categorical') {
            triggerCategoricalAnalysis();
        } else if (edaId === 'eda-bivariate') {
            triggerBivariateAnalysis();
        } else if (edaId === 'eda-correlation') {
            fetch('/api/eda/correlation')
                .then(r => r.json())
                .then(res => {
                    if (res.status !== 'success') return;
                    const c = res.correlation;
                    const elem = document.getElementById('plotCorrelation');
                    if (elem) {
                        const data = [{
                            z: c.z_matrix, x: c.columns, y: c.columns,
                            type: 'heatmap', colorscale: 'Viridis'
                        }];
                        Plotly.newPlot('plotCorrelation', data, { title: 'Numerical Features Correlation Heatmap', margin: { t: 40, b: 80, l: 80, r: 20 } });
                    }

                    // Top Pos & Neg
                    let posHtml = `<ul style="font-size:13px; padding-left:18px;">`;
                    c.top_positive.forEach(p => { posHtml += `<li><strong>${p.feature_1} & ${p.feature_2}</strong>: +${p.correlation}</li>`; });
                    posHtml += `</ul>`;
                    if (document.getElementById('topPosCorrList')) document.getElementById('topPosCorrList').innerHTML = posHtml;

                    let negHtml = `<ul style="font-size:13px; padding-left:18px;">`;
                    c.top_negative.forEach(p => { negHtml += `<li><strong>${p.feature_1} & ${p.feature_2}</strong>: ${p.correlation}</li>`; });
                    negHtml += `</ul>`;
                    if (document.getElementById('topNegCorrList')) document.getElementById('topNegCorrList').innerHTML = negHtml;
                });
        } else if (edaId === 'eda-outliers') {
            triggerOutlierAnalysis();
        }
    }

    // EDA Nav Buttons
    document.querySelectorAll('.eda-tab-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const edaId = this.getAttribute('data-eda');
            if (edaId) loadEDASection(edaId);
        });
    });

    function triggerUnivariateAnalysis() {
        const feature = document.getElementById('edaUniFeature')?.value || 'CGPA';
        const viz = document.getElementById('edaUniViz')?.value || 'histogram';

        fetch(`/api/eda/univariate?feature=${feature}&viz_type=${viz}`)
            .then(r => r.json())
            .then(res => {
                if (res.status !== 'success') return;
                const d = res.data;
                const st = d.stats;

                if (d.viz_type === 'histogram') plotHistogram('plotUnivariate', d.values, `${st.feature} Distribution`, st.feature);
                else if (d.viz_type === 'boxplot') plotBoxplot('plotUnivariate', d.values, `${st.feature} Boxplot`, st.feature);
                else plotHistogram('plotUnivariate', d.values, `${st.feature} Distribution`, st.feature);

                let statsHtml = `<p>Mean: <strong>${st.mean}</strong></p><p>Median: <strong>${st.median}</strong></p><p>Mode: <strong>${st.mode}</strong></p><p>Std Dev: <strong>${st.std}</strong></p><p>Min: <strong>${st.min}</strong> | Max: <strong>${st.max}</strong></p><p>Q1: <strong>${st.q1}</strong> | Q3: <strong>${st.q3}</strong></p><p>IQR: <strong>${st.iqr}</strong></p>`;
                if (document.getElementById('uniStatsContent')) document.getElementById('uniStatsContent').innerHTML = statsHtml;
            });
    }

    document.getElementById('edaUniFeature')?.addEventListener('change', triggerUnivariateAnalysis);
    document.getElementById('edaUniViz')?.addEventListener('change', triggerUnivariateAnalysis);

    function triggerCategoricalAnalysis() {
        const feature = document.getElementById('edaCatFeature')?.value || 'Stream';

        fetch(`/api/eda/categorical?feature=${feature}`)
            .then(r => r.json())
            .then(res => {
                if (res.status !== 'success') return;
                const d = res.data;

                plotBar('plotCategorical', d.categories, d.counts, `${d.feature} Category Counts`, '#10b981', d.feature, 'Count');

                let html = `<ul style="font-size:13px; padding-left:18px;">`;
                for (let i = 0; i < d.categories.length; i++) {
                    html += `<li><strong>${d.categories[i]}</strong>: ${d.counts[i]} (${d.percentages[i]}%)</li>`;
                }
                html += `</ul>`;
                if (document.getElementById('catFreqContent')) document.getElementById('catFreqContent').innerHTML = html;
            });
    }

    document.getElementById('edaCatFeature')?.addEventListener('change', triggerCategoricalAnalysis);

    function triggerBivariateAnalysis() {
        const x = document.getElementById('edaBiX')?.value || 'CGPA';
        const y = document.getElementById('edaBiY')?.value || 'CodingTestScore';
        const grp = document.getElementById('edaBiGroup')?.value || 'PlacementStatus';

        fetch(`/api/eda/bivariate?x_feature=${x}&y_feature=${y}&group_by=${grp}`)
            .then(r => r.json())
            .then(res => {
                if (res.status !== 'success') return;
                const d = res.data;
                plotScatter('plotBivariate', d.x, d.y, d.group_values, `${d.x_name} vs ${d.y_name}`, d.x_name, d.y_name);
            });
    }

    document.getElementById('edaBiX')?.addEventListener('change', triggerBivariateAnalysis);
    document.getElementById('edaBiY')?.addEventListener('change', triggerBivariateAnalysis);
    document.getElementById('edaBiGroup')?.addEventListener('change', triggerBivariateAnalysis);

    function triggerOutlierAnalysis() {
        const feature = document.getElementById('edaOutlierFeature')?.value || 'CGPA';

        fetch(`/api/eda/outliers?feature=${feature}`)
            .then(r => r.json())
            .then(res => {
                if (res.status !== 'success') return;
                const d = res.data;

                plotBoxplot('plotOutlier', d.sample_values, `${d.feature} Outlier Analysis Boxplot`, d.feature);

                let html = `<p>Q1 (25%): <strong>${d.q1}</strong></p><p>Q3 (75%): <strong>${d.q3}</strong></p><p>IQR: <strong>${d.iqr}</strong></p><p>Lower Bound: <strong>${d.lower_bound}</strong></p><p>Upper Bound: <strong>${d.upper_bound}</strong></p><hr style="margin:8px 0;"><p>Outlier Count: <strong>${d.outlier_count}</strong> (${d.outlier_percentage}%)</p>`;
                if (document.getElementById('outlierStatsContent')) document.getElementById('outlierStatsContent').innerHTML = html;
            });
    }

    document.getElementById('edaOutlierFeature')?.addEventListener('change', triggerOutlierAnalysis);

    // --- PREPROCESSING EXECUTION ---
    document.getElementById('btnRunPreprocessing')?.addEventListener('click', function() {
        const strategy = document.getElementById('prepImputeStrategy')?.value || 'mean';
        const testSize = document.getElementById('prepTestSize')?.value || '0.2';

        fetch('/api/preprocess', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ imputation_strategy: strategy, test_size: testSize })
        })
        .then(r => r.json())
        .then(res => {
            if (res.status !== 'success') return;
            const b = res.before;
            const a = res.after;

            let html = `<div style="font-size:14px; line-height:1.8;">
                <p>✔ Missing Value Imputation: <strong>${a.imputation_strategy.toUpperCase()}</strong></p>
                <p>✔ Categorical One-Hot Encoding: Created <strong>${a.encoded_features_count}</strong> numerical features</p>
                <p>✔ StandardScaler Applied to Training & Testing Data</p>
                <p>✔ Train Set Size: <strong>${a.train_samples.toLocaleString()}</strong> rows | Test Set Size: <strong>${a.test_samples.toLocaleString()}</strong> rows</p>
            </div>`;

            const box = document.getElementById('prepResultContainer');
            const content = document.getElementById('prepResultContent');
            if (box) box.classList.remove('hidden');
            if (content) content.innerHTML = html;
        });
    });

    // --- MODEL TRAINING EXECUTION ---
    document.getElementById('btnTrainModel')?.addEventListener('click', function() {
        const modelName = document.getElementById('trainModelName')?.value || 'Random Forest';
        const maxDepth = document.getElementById('trainMaxDepth')?.value || '10';
        const nEstimators = document.getElementById('trainNEstimators')?.value || '100';

        const statusBox = document.getElementById('trainingStatusBox');
        if (statusBox) statusBox.classList.remove('hidden');

        fetch('/api/train-model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName, max_depth: maxDepth, n_estimators: nEstimators })
        })
        .then(r => r.json())
        .then(res => {
            if (statusBox) statusBox.classList.add('hidden');
            if (res.status !== 'success') return;

            const m = res.metrics;
            let html = `<div class="stats-grid" style="margin-top:16px;">
                <div class="kpi-card green"><div class="kpi-info"><span class="kpi-label">Accuracy</span><h3 class="kpi-value">${m.accuracy}%</h3></div></div>
                <div class="kpi-card blue"><div class="kpi-info"><span class="kpi-label">Precision</span><h3 class="kpi-value">${m.precision}%</h3></div></div>
                <div class="kpi-card purple"><div class="kpi-info"><span class="kpi-label">Recall</span><h3 class="kpi-value">${m.recall}%</h3></div></div>
                <div class="kpi-card gold"><div class="kpi-info"><span class="kpi-label">F1-Score</span><h3 class="kpi-value">${m.f1_score}%</h3></div></div>
            </div>`;

            const resPanel = document.getElementById('trainingResultPanel');
            const summaryBox = document.getElementById('trainingMetricsSummary');
            if (resPanel) resPanel.classList.remove('hidden');
            if (summaryBox) summaryBox.innerHTML = html;
        })
        .catch(err => {
            if (statusBox) statusBox.classList.add('hidden');
            console.error('Training Error:', err);
        });
    });

    // --- MODEL EVALUATION LOAD ---
    function loadModelEvaluation() {
        fetch('/api/model-comparison')
            .then(r => r.json())
            .then(res => {
                if (res.status !== 'success') return;
                const comp = res.comparison;
                const container = document.getElementById('modelComparisonTableContainer');

                if (!comp.has_models || comp.models.length === 0) {
                    if (container) container.innerHTML = `<p class="text-muted">No trained model metrics found. Please train a model under Model Training.</p>`;
                    return;
                }

                let html = `<table class="custom-table"><thead><tr><th>Model Algorithm</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th><th>Status</th></tr></thead><tbody>`;
                comp.models.forEach(m => {
                    const isBest = m.model_name === comp.best_model;
                    html += `<tr><td><strong>${m.model_name}</strong></td><td>${m.accuracy}%</td><td>${m.precision}%</td><td>${m.recall}%</td><td>${m.f1_score}%</td><td>${m.auc}</td><td>${isBest ? '<span class="badge status-placed">🏆 Best Model</span>' : '<span class="badge tier-badge">Trained</span>'}</td></tr>`;
                });
                html += `</tbody></table>`;
                if (container) container.innerHTML = html;

                // Render latest trained model Confusion Matrix & ROC Curve & Feature Importance
                const latest = comp.models[comp.models.length - 1];
                if (latest) {
                    // Confusion Matrix Heatmap
                    const cmElem = document.getElementById('plotConfusionMatrix');
                    if (cmElem) {
                        const data = [{
                            z: latest.confusion_matrix, x: ['Pred Unplaced', 'Pred Placed'], y: ['Actual Unplaced', 'Actual Placed'],
                            type: 'heatmap', colorscale: 'Blues'
                        }];
                        Plotly.newPlot('plotConfusionMatrix', data, { title: `${latest.model_name} Confusion Matrix`, margin: { t: 40, b: 40, l: 80, r: 20 } });
                    }

                    // ROC Curve
                    const rocElem = document.getElementById('plotROCCurve');
                    if (rocElem && latest.roc_curve) {
                        const data = [{
                            x: latest.roc_curve.fpr, y: latest.roc_curve.tpr,
                            type: 'scatter', mode: 'lines', name: `ROC (AUC = ${latest.auc})`,
                            line: { color: '#4f46e5', width: 3 }
                        }];
                        Plotly.newPlot('plotROCCurve', data, { title: `${latest.model_name} ROC Curve`, xaxis: { title: 'False Positive Rate' }, yaxis: { title: 'True Positive Rate' } });
                    }

                    // Feature Importance
                    if (latest.feature_importances) {
                        const features = latest.feature_importances.map(x => x.feature).reverse();
                        const scores = latest.feature_importances.map(x => x.importance).reverse();
                        const fiElem = document.getElementById('plotFeatureImportance');
                        if (fiElem) {
                            const data = [{
                                x: scores, y: features, type: 'bar', orientation: 'h',
                                marker: { color: '#8b5cf6' }
                            }];
                            Plotly.newPlot('plotFeatureImportance', data, { title: 'Top Feature Importance Coefficients', margin: { t: 40, b: 40, l: 180, r: 20 } });
                        }
                    }
                }
            });
    }

    // --- PLACEMENT ANALYTICS LOAD ---
    function loadPlacementAnalytics() {
        fetch('/api/placement-analytics')
            .then(r => r.json())
            .then(res => {
                if (res.status !== 'success') return;
                const a = res.analytics;

                plotBar('plotPlacementGender', a.by_gender.map(x => x.col_val), a.by_gender.map(x => x.rate), 'Placement Rate by Gender (%)', '#06b6d4', 'Gender', 'Rate %');
                plotBar('plotPlacementStream', a.by_stream.map(x => x.col_val), a.by_stream.map(x => x.rate), 'Placement Rate by Stream (%)', '#10b981', 'Stream', 'Rate %');
                plotBar('plotPlacementTier', a.by_tier.map(x => x.col_val), a.by_tier.map(x => x.rate), 'Placement Rate by College Tier (%)', '#8b5cf6', 'College Tier', 'Rate %');
                plotBar('plotPlacementBacklogs', a.by_backlogs.map(x => x.col_val), a.by_backlogs.map(x => x.rate), 'Placement Rate by Backlog History (%)', '#f43f5e', 'Backlogs', 'Rate %');

                // Associations table
                let html = `<table class="custom-table"><thead><tr><th>Feature</th><th>Statistical Association with Placement</th><th>Strength</th></tr></thead><tbody>`;
                a.associations.forEach(item => {
                    const str = item.correlation > 0 ? 'Positive Association' : 'Negative Association';
                    html += `<tr><td><strong>${item.feature}</strong></td><td>${item.correlation}</td><td><span class="badge tier-badge">${str}</span></td></tr>`;
                });
                html += `</tbody></table>`;
                if (document.getElementById('placementAssociationsTable')) document.getElementById('placementAssociationsTable').innerHTML = html;
            });
    }

    // --- SALARY ANALYTICS LOAD ---
    function loadSalaryAnalytics() {
        fetch('/api/salary-analytics')
            .then(r => r.json())
            .then(res => {
                if (res.status !== 'success') return;
                const a = res.analytics;
                if (!a.has_data) return;

                if (document.getElementById('salAvgVal')) document.getElementById('salAvgVal').textContent = `₹${a.stats.avg_salary} LPA`;
                if (document.getElementById('salMedianVal')) document.getElementById('salMedianVal').textContent = `₹${a.stats.median_salary} LPA`;
                if (document.getElementById('salMaxVal')) document.getElementById('salMaxVal').textContent = `₹${a.stats.max_salary} LPA`;

                plotHistogram('plotSalaryDistPage', a.salary_values, 'Salary Package Histogram (Placed Students)', 'Salary LPA');
                plotBar('plotSalaryByStream', a.by_stream.map(x => x.Stream), a.by_stream.map(x => x.avg), 'Average Salary by Stream (LPA)', '#eab308', 'Stream', 'Avg LPA');
                plotScatter('plotSalaryVsCGPA', a.salary_vs_cgpa.cgpa, a.salary_vs_cgpa.salary, null, 'Salary Package vs CGPA', 'CGPA', 'Salary Package (LPA)');
                plotScatter('plotSalaryVsCoding', a.salary_vs_coding.coding, a.salary_vs_coding.salary, null, 'Salary Package vs Coding Score', 'Coding Score', 'Salary Package (LPA)');
            });
    }

    // --- ANOMALY ANALYTICS LOAD ---
    function loadAnomalyAnalytics() {
        fetch('/api/anomaly-analytics')
            .then(r => r.json())
            .then(res => {
                if (res.status !== 'success') return;
                const a = res.analytics;
                if (!a.has_data) return;

                if (document.getElementById('anomTotalVal')) document.getElementById('anomTotalVal').textContent = a.total_records.toLocaleString();
                if (document.getElementById('anomNormalVal')) document.getElementById('anomNormalVal').textContent = a.normal_records.toLocaleString();
                if (document.getElementById('anomCountVal')) document.getElementById('anomCountVal').textContent = a.anomalous_records.toLocaleString();
                if (document.getElementById('anomRateVal')) document.getElementById('anomRateVal').textContent = `${a.anomaly_percentage}%`;

                plotBar('plotAnomalyStream', a.by_stream.map(x => x.Stream), a.by_stream.map(x => x.anomalies), 'Anomalies by Academic Stream', '#f43f5e', 'Stream', 'Anomaly Count');
                plotBar('plotAnomalyTier', a.by_tier.map(x => x.CollegeTier), a.by_tier.map(x => x.anomalies), 'Anomalies by College Tier', '#3b82f6', 'College Tier', 'Anomaly Count');

                // Anomaly Table
                const tbody = document.querySelector('#tableAnomalyRecords tbody');
                if (tbody) {
                    let html = '';
                    a.anomaly_table.forEach(r => {
                        html += `<tr><td>#${r.StudentID}</td><td>${r.Gender}</td><td>${r.City}</td><td>${r.Stream}</td><td>${r.CollegeTier}</td><td>${r.CGPA}</td><td>${r.CodingTestScore}</td><td>${r.PlacementStatus === 1 ? '<span class="badge status-placed">Placed</span>' : '<span class="badge status-unplaced">Unplaced</span>'}</td><td>₹${r['Salary Package']} LPA</td><td><span class="badge status-unplaced">IsAnomaly</span></td></tr>`;
                    });
                    tbody.innerHTML = html;
                }
            });
    }

    // --- STUDENT RECORDS EXPLORER ---
    function loadStudentRecords() {
        const gender = document.getElementById('stuFilterGender')?.value || 'All';
        const stream = document.getElementById('stuFilterStream')?.value || 'All';
        const tier = document.getElementById('stuFilterTier')?.value || 'All';
        const status = document.getElementById('stuFilterStatus')?.value || 'All';
        const search = document.getElementById('stuSearchInput')?.value || '';

        const url = `/api/students?gender=${gender}&stream=${stream}&tier=${tier}&placement_status=${status}&search=${encodeURIComponent(search)}&page=${studentCurrentPage}&per_page=25`;

        fetch(url)
            .then(r => r.json())
            .then(res => {
                if (res.status !== 'success') return;
                const d = res.data;
                studentTotalPages = d.total_pages;

                if (document.getElementById('stuPageInfo')) document.getElementById('stuPageInfo').textContent = `Showing Page ${d.current_page} of ${d.total_pages} (${d.total_records.toLocaleString()} total records)`;

                const tbody = document.querySelector('#studentsFullTable tbody');
                if (tbody) {
                    let html = '';
                    d.records.forEach(r => {
                        const statusBadge = r.PlacementStatus == 1 ? '<span class="badge status-placed">Placed</span>' : '<span class="badge status-unplaced">Unplaced</span>';
                        html += `<tr><td>#${r.StudentID}</td><td>${r.Gender}</td><td>${r.City}</td><td><span class="badge tier-badge">${r.CollegeTier}</span></td><td>${r.Stream}</td><td><strong>${r.CGPA}</strong></td><td>${r.HistoryOfBacklogs}</td><td>${r.Internships}</td><td>${r.CodingTestScore}</td><td>${r.MockInterviewScore}</td><td>${statusBadge}</td><td>₹${r['Salary Package']} LPA</td></tr>`;
                    });
                    tbody.innerHTML = html;
                }
            });
    }

    document.getElementById('stuFilterGender')?.addEventListener('change', () => { studentCurrentPage = 1; loadStudentRecords(); });
    document.getElementById('stuFilterStream')?.addEventListener('change', () => { studentCurrentPage = 1; loadStudentRecords(); });
    document.getElementById('stuFilterTier')?.addEventListener('change', () => { studentCurrentPage = 1; loadStudentRecords(); });
    document.getElementById('stuFilterStatus')?.addEventListener('change', () => { studentCurrentPage = 1; loadStudentRecords(); });
    document.getElementById('stuSearchInput')?.addEventListener('keyup', () => { studentCurrentPage = 1; loadStudentRecords(); });

    document.getElementById('btnPrevPage')?.addEventListener('click', () => {
        if (studentCurrentPage > 1) { studentCurrentPage--; loadStudentRecords(); }
    });
    document.getElementById('btnNextPage')?.addEventListener('click', () => {
        if (studentCurrentPage < studentTotalPages) { studentCurrentPage++; loadStudentRecords(); }
    });

    document.getElementById('btnExportCSV')?.addEventListener('click', () => {
        const gender = document.getElementById('stuFilterGender')?.value || 'All';
        const stream = document.getElementById('stuFilterStream')?.value || 'All';
        const tier = document.getElementById('stuFilterTier')?.value || 'All';
        const status = document.getElementById('stuFilterStatus')?.value || 'All';

        window.location.href = `/api/export-students-csv?gender=${gender}&stream=${stream}&tier=${tier}&placement_status=${status}`;
    });

    // --- RANGE SLIDERS LIVE FEEDBACK ---
    const sliderPairs = [
        ['input_cgpa', 'val_cgpa'],
        ['input_coding', 'val_coding'],
        ['input_aptitude', 'val_aptitude'],
        ['input_interview', 'val_interview']
    ];

    sliderPairs.forEach(([inputId, valId]) => {
        const inp = document.getElementById(inputId);
        const val = document.getElementById(valId);
        if (inp && val) {
            inp.addEventListener('input', function() { val.textContent = this.value; });
        }
    });

    // --- PREDICTOR FORM SUBMISSION ---
    const predForm = document.getElementById('predictionForm');
    if (predForm) {
        predForm.addEventListener('submit', function(e) {
            e.preventDefault();

            const formData = new FormData(predForm);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'success') {
                    const placeholder = document.getElementById('resultPlaceholder');
                    const card = document.getElementById('resultCard');
                    if (placeholder) placeholder.classList.add('hidden');
                    if (card) card.classList.remove('hidden');

                    if (document.getElementById('resStudentName')) document.getElementById('resStudentName').textContent = data.student_name;
                    if (document.getElementById('resSubDetails')) document.getElementById('resSubDetails').textContent = `${data.stream} • ${data.tier} (${data.model_used})`;

                    const badge = document.getElementById('resStatusBadge');
                    if (badge) {
                        badge.textContent = data.placement_status;
                        badge.className = data.is_placed ? 'result-badge' : 'result-badge at-risk';
                    }

                    if (document.getElementById('resProbability')) document.getElementById('resProbability').textContent = `${data.probability}%`;
                    if (document.getElementById('resProbBar')) document.getElementById('resProbBar').style.width = `${data.probability}%`;

                    if (document.getElementById('resPackage')) {
                        document.getElementById('resPackage').textContent = data.is_placed ? `₹${data.estimated_package} LPA` : '₹0.00 LPA (Unplaced)';
                    }

                    const recsList = document.getElementById('resRecsList');
                    if (recsList) {
                        recsList.innerHTML = '';
                        data.recommendations.forEach(r => {
                            const li = document.createElement('li');
                            li.textContent = r;
                            recsList.appendChild(li);
                        });
                    }
                }
            })
            .catch(err => console.error('Prediction Error:', err));
        });
    }

    // Initial load
    loadDashboardStats();
});
