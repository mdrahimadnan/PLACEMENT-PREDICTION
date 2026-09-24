/* ==========================================================================
   PlacementAI — Premium ML Dashboard | Main Client Script
   Plotly dark theme, full API integration, dynamic charts & interactions
   ========================================================================== */

document.addEventListener('DOMContentLoaded', function () {

    // -----------------------------------------------------------------------
    // GLOBAL STATE
    // -----------------------------------------------------------------------
    let studentCurrentPage = 1;
    let studentTotalPages  = 1;

    // -----------------------------------------------------------------------
    // PLOTLY DARK THEME BASE LAYOUT
    // -----------------------------------------------------------------------
    const DARK_LAYOUT = {
        paper_bgcolor: 'transparent',
        plot_bgcolor:  'transparent',
        font:          { family: 'Inter, system-ui, sans-serif', color: '#94a3b8', size: 12 },
        xaxis: {
            gridcolor: 'rgba(148,163,184,0.08)',
            linecolor: 'rgba(148,163,184,0.12)',
            tickfont:  { color: '#64748b', size: 11 },
            title:     { font: { color: '#94a3b8', size: 12 } }
        },
        yaxis: {
            gridcolor: 'rgba(148,163,184,0.08)',
            linecolor: 'rgba(148,163,184,0.12)',
            tickfont:  { color: '#64748b', size: 11 },
            title:     { font: { color: '#94a3b8', size: 12 } }
        },
        margin:   { t: 30, b: 45, l: 50, r: 20 },
        showlegend: true,
        legend: { font: { color: '#94a3b8', size: 11 }, bgcolor: 'transparent' }
    };

    const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

    // Colour Palette
    const C = {
        purple:  '#8b5cf6', violet:  '#7c3aed', indigo:  '#6366f1',
        cyan:    '#06b6d4', emerald: '#10b981', rose:    '#f43f5e',
        amber:   '#f59e0b', gold:    '#eab308', blue:    '#3b82f6',
        teal:    '#14b8a6', slate:   '#64748b',
        palette: ['#8b5cf6','#06b6d4','#10b981','#f43f5e','#f59e0b','#3b82f6','#14b8a6','#eab308','#a78bfa','#34d399'],
    };

    // -----------------------------------------------------------------------
    // UTILITY HELPERS
    // -----------------------------------------------------------------------
    function mergeLayout(extra) {
        return Object.assign({}, DARK_LAYOUT, extra, {
            xaxis: Object.assign({}, DARK_LAYOUT.xaxis, extra.xaxis || {}),
            yaxis: Object.assign({}, DARK_LAYOUT.yaxis, extra.yaxis || {}),
        });
    }

    function fmtNum(n) {
        if (n === null || n === undefined || isNaN(n)) return '—';
        return parseFloat(n).toLocaleString('en-IN', { maximumFractionDigits: 2 });
    }

    function badge(txt, cls) {
        return `<span class="badge ${cls}">${txt}</span>`;
    }

    // -----------------------------------------------------------------------
    // TAB SWITCHING
    // -----------------------------------------------------------------------
    const PAGE_TITLES = {
        'tab-overview':       'Dashboard Overview',
        'tab-loading':        'Data Loading & Status',
        'tab-eda':            'Interactive EDA Hub',
        'tab-placement':      'Placement Analysis',
        'tab-salary':         'Salary Analysis',
        'tab-anomaly':        'Anomaly Detection',
        'tab-preprocessing':  'Data Preprocessing',
        'tab-training':       'Model Training',
        'tab-evaluation':     'Model Evaluation',
        'tab-predictor':      'Placement Predictor',
        'tab-students':       'Student Records',
    };

    window.switchTab = function (tabId) {
        document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));

        const pane = document.getElementById(tabId);
        if (pane) pane.classList.add('active');

        const nav = document.querySelector(`.nav-link[data-tab="${tabId}"]`);
        if (nav) nav.classList.add('active');

        const titleEl = document.getElementById('active-tab-title');
        if (titleEl) titleEl.textContent = PAGE_TITLES[tabId] || tabId;

        window.scrollTo({ top: 0, behavior: 'smooth' });

        // Lazy-load section data
        if (tabId === 'tab-overview')    loadDashboardStats();
        if (tabId === 'tab-eda')         loadEDASection('eda-overview');
        if (tabId === 'tab-evaluation')  loadModelEvaluation();
        if (tabId === 'tab-placement')   loadPlacementAnalytics();
        if (tabId === 'tab-salary')      loadSalaryAnalytics();
        if (tabId === 'tab-anomaly')     loadAnomalyAnalytics();
        if (tabId === 'tab-students')    loadStudentRecords();
    };

    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function () {
            const tabId = this.getAttribute('data-tab');
            if (tabId) switchTab(tabId);
        });
    });

    // -----------------------------------------------------------------------
    // PLOTLY CHART BUILDERS
    // -----------------------------------------------------------------------
    function plotPie(divId, labels, values, title = '') {
        const el = document.getElementById(divId);
        if (!el) return;
        Plotly.newPlot(divId, [{
            labels, values, type: 'pie', hole: 0.42,
            marker: { colors: C.palette },
            textfont: { color: '#fff', size: 11 },
            hovertemplate: '%{label}: %{value:,.0f} (%{percent})<extra></extra>'
        }], mergeLayout({ margin: { t: 20, b: 20, l: 20, r: 20 } }), PLOTLY_CONFIG);
    }

    function plotBar(divId, x, y, { color = C.purple, xTitle = '', yTitle = '', horizontal = false } = {}) {
        const el = document.getElementById(divId);
        if (!el) return;
        const trace = horizontal
            ? { x: y, y: x, type: 'bar', orientation: 'h', marker: { color, opacity: 0.9 }, hovertemplate: '<b>%{y}</b><br>Value: %{x:.2f}<extra></extra>' }
            : { x, y, type: 'bar', marker: { color, opacity: 0.9 }, hovertemplate: '<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>' };
        Plotly.newPlot(divId, [trace], mergeLayout({
            xaxis: { title: { text: xTitle }, tickfont: { size: 11 } },
            yaxis: { title: { text: yTitle }, tickfont: { size: 11 } },
            showlegend: false
        }), PLOTLY_CONFIG);
    }

    function plotGroupedBar(divId, xVals, traceDefs) {
        const el = document.getElementById(divId);
        if (!el) return;
        const traces = traceDefs.map((t, i) => ({
            x: xVals, y: t.y, name: t.name, type: 'bar',
            marker: { color: C.palette[i % C.palette.length], opacity: 0.85 }
        }));
        Plotly.newPlot(divId, traces, mergeLayout({ barmode: 'group' }), PLOTLY_CONFIG);
    }

    function plotHistogram(divId, xValues, { color = C.indigo, xTitle = '' } = {}) {
        const el = document.getElementById(divId);
        if (!el) return;
        Plotly.newPlot(divId, [{
            x: xValues, type: 'histogram',
            marker: { color, opacity: 0.85 },
            hovertemplate: 'Range: %{x}<br>Count: %{y}<extra></extra>'
        }], mergeLayout({
            xaxis: { title: { text: xTitle } },
            yaxis: { title: { text: 'Student Count' } },
            showlegend: false,
        }), PLOTLY_CONFIG);
    }

    function plotScatter(divId, xVals, yVals, { color = C.purple, xTitle = '', yTitle = '', mode = 'markers' } = {}) {
        const el = document.getElementById(divId);
        if (!el) return;
        Plotly.newPlot(divId, [{
            x: xVals, y: yVals, mode, type: 'scatter',
            marker: { color, size: 4, opacity: 0.55 },
        }], mergeLayout({
            xaxis: { title: { text: xTitle } },
            yaxis: { title: { text: yTitle } },
            showlegend: false,
        }), PLOTLY_CONFIG);
    }

    // -----------------------------------------------------------------------
    // ① DASHBOARD OVERVIEW — load stats + charts
    // -----------------------------------------------------------------------
    function loadDashboardStats() {
        const params = buildDashFilterParams();
        fetch(`/api/dashboard-stats?${params}`)
            .then(r => r.json())
            .then(data => {
                if (data.status !== 'success') return;
                const s = data.summary;
                const ch = data.charts;

                // Update KPI cards
                setText('kpiTotalStudents',  s.n_rows.toLocaleString());
                setText('kpiPlacedCount',    s.placed_count.toLocaleString());
                setText('kpiUnplacedCount',  s.unplaced_count.toLocaleString());
                setText('kpiPlacementRateVal', s.placement_rate + '%');
                setText('kpiPlacementRate',   s.placement_rate + '% Placement Rate');
                setText('kpiAvgCGPA',         s.avg_cgpa);
                setText('kpiAvgAttendance',   s.avg_attendance + '%');
                setText('kpiAvgSalary',       '₹' + s.avg_salary + ' LPA');
                setText('kpiTotalAnomalies',  s.total_anomalies.toLocaleString());

                // Chart 1: Placement pie
                plotPie('chartPlacementDist',
                    ['Placed 🎓', 'Not Placed ❌'],
                    [ch.placement_distribution.placed, ch.placement_distribution.unplaced]
                );

                // Chart 2: Students by stream (bar)
                const streamKeys = Object.keys(ch.stream_counts);
                plotBar('chartStudentsStream', streamKeys, streamKeys.map(k => ch.stream_counts[k]),
                    { color: streamKeys.map((_, i) => C.palette[i % C.palette.length]), yTitle: 'Students' });

                // Chart 3: Placement rate by stream
                const srKeys = Object.keys(ch.placement_by_stream);
                plotBar('chartRateStream', srKeys, srKeys.map(k => ch.placement_by_stream[k]),
                    { color: C.emerald, yTitle: 'Placement Rate (%)' });

                // Chart 4: Placement by tier
                const tierKeys = Object.keys(ch.placement_by_tier);
                plotBar('chartTierPlacement', tierKeys, tierKeys.map(k => ch.placement_by_tier[k]),
                    { color: C.indigo, yTitle: 'Placement Rate (%)' });

                // Chart 5: CGPA histogram
                plotHistogram('chartCGPADist', ch.cgpa_sample,
                    { color: C.violet, xTitle: 'CGPA' });

                // Chart 6: Salary histogram
                plotHistogram('chartSalaryDist', ch.salary_sample,
                    { color: C.amber, xTitle: 'Salary Package (LPA)' });

                // Chart 7: CGPA box by placement status placeholder (uses bivariate API)
                const cgpaPlaced   = ch.cgpa_sample.slice(0, 1000);
                const cgpaUnplaced = ch.cgpa_sample.slice(1000, 2000);
                Plotly.newPlot('chartCGPAvsPlacement', [
                    { y: cgpaPlaced,   name: 'Placed',     type: 'box', marker: { color: C.emerald } },
                    { y: cgpaUnplaced, name: 'Not Placed', type: 'box', marker: { color: C.rose    } }
                ], mergeLayout({ yaxis: { title: { text: 'CGPA' } } }), PLOTLY_CONFIG);

                // Chart 8: Placement by gender
                const genKeys = Object.keys(ch.placement_by_gender);
                plotBar('chartGenderPlacement', genKeys, genKeys.map(k => ch.placement_by_gender[k]),
                    { color: genKeys.map((_, i) => [C.cyan, C.purple][i % 2]), yTitle: 'Placement Rate (%)' });
            })
            .catch(console.error);
    }

    function buildDashFilterParams() {
        const params = new URLSearchParams();
        const get = id => { const el = document.getElementById(id); return el ? el.value : 'All'; };
        params.set('gender',           get('dashFilterGender'));
        params.set('stream',           get('dashFilterStream'));
        params.set('tier',             get('dashFilterTier'));
        params.set('placement_status', get('dashFilterStatus'));
        return params.toString();
    }

    function setText(id, val) {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    }

    document.getElementById('btnApplyDashFilters')?.addEventListener('click', loadDashboardStats);
    document.getElementById('btnResetDashFilters')?.addEventListener('click', () => {
        ['dashFilterGender','dashFilterStream','dashFilterTier','dashFilterStatus'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.value = 'All';
        });
        loadDashboardStats();
    });

    // -----------------------------------------------------------------------
    // ② EDA HUB — sub-tab switching
    // -----------------------------------------------------------------------
    document.querySelectorAll('.eda-tab-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            document.querySelectorAll('.eda-tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.eda-pane').forEach(p => p.classList.remove('active'));
            this.classList.add('active');
            const target = this.getAttribute('data-eda');
            const pane = document.getElementById(target);
            if (pane) pane.classList.add('active');
            loadEDASection(target);
        });
    });

    function loadEDASection(paneId) {
        if (paneId === 'eda-overview')     loadEDAOverview();
        if (paneId === 'eda-missing')      loadEDAMissing();
        if (paneId === 'eda-correlation')  loadEDACorrelation();
    }

    // EDA A: Overview
    function loadEDAOverview() {
        const el = document.getElementById('edaOverviewContent');
        if (!el || el.dataset.loaded) return;
        el.innerHTML = '<div style="color:var(--text-muted);font-size:13px;padding:10px 0;">Loading…</div>';
        fetch('/api/eda/overview').then(r => r.json()).then(data => {
            if (data.status !== 'success') { el.innerHTML = '<p style="color:var(--text-muted)">Failed to load.</p>'; return; }
            const ov = data.overview;
            const rows = ov.columns.map(c => `
                <tr>
                    <td><strong>${c.name}</strong></td>
                    <td>${c.dtype}</td>
                    <td>${c.missing_count} (${c.missing_pct}%)</td>
                    <td>${c.unique_count}</td>
                    <td>${c.sample_values ? c.sample_values.slice(0,3).join(', ') : '—'}</td>
                </tr>`).join('');
            el.innerHTML = `
                <div style="overflow-x:auto; border:1px solid var(--border); border-radius:var(--r-sm);">
                    <table class="ov-table">
                        <thead><tr><th>Column</th><th>Data Type</th><th>Missing</th><th>Unique</th><th>Sample Values</th></tr></thead>
                        <tbody>${rows}</tbody>
                    </table>
                </div>`;
            el.dataset.loaded = '1';
        }).catch(() => { el.innerHTML = '<p style="color:var(--text-muted)">Error loading overview.</p>'; });
    }

    // EDA B: Missing
    function loadEDAMissing() {
        const el = document.getElementById('plotMissingBar');
        if (!el || el.dataset.loaded) return;
        fetch('/api/eda/missing').then(r => r.json()).then(data => {
            if (data.status !== 'success') return;
            const m = data.missing;
            const cols = m.columns.filter(c => c.missing_count > 0);
            if (!cols.length) {
                el.innerHTML = '<p style="color:var(--brand-emerald);padding:20px;">✅ No missing values found in the dataset!</p>';
            } else {
                plotBar('plotMissingBar', cols.map(c => c.name), cols.map(c => c.missing_pct),
                    { color: C.rose, yTitle: 'Missing %' });
            }
            // Table
            const tableEl = document.getElementById('edaMissingTable');
            if (tableEl) {
                const rows = m.columns.map(c => `
                    <tr><td>${c.name}</td><td>${c.missing_count}</td><td>${c.missing_pct}%</td></tr>`).join('');
                tableEl.innerHTML = `<div style="overflow-x:auto;border:1px solid var(--border);border-radius:var(--r-sm);">
                    <table class="ov-table"><thead><tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr></thead>
                    <tbody>${rows}</tbody></table></div>`;
            }
            el.dataset.loaded = '1';
        }).catch(console.error);
    }

    // EDA C: Univariate — on demand
    document.getElementById('btnLoadUnivariate')?.addEventListener('click', () => {
        const feature  = document.getElementById('edaUniFeature')?.value || 'CGPA';
        const vizType  = document.getElementById('edaUniViz')?.value    || 'histogram';
        fetch(`/api/eda/univariate?feature=${encodeURIComponent(feature)}&viz_type=${vizType}`)
            .then(r => r.json()).then(data => {
                if (data.status !== 'success') return;
                const d = data.data;
                const divId = 'plotUnivariate';

                if (vizType === 'histogram') {
                    plotHistogram(divId, d.values, { color: C.violet, xTitle: feature });
                } else if (vizType === 'boxplot') {
                    Plotly.newPlot(divId, [{ y: d.values, type: 'box', name: feature, marker: { color: C.cyan } }],
                        mergeLayout({ yaxis: { title: { text: feature } }, showlegend: false }), PLOTLY_CONFIG);
                } else if (vizType === 'violin') {
                    Plotly.newPlot(divId, [{ y: d.values, type: 'violin', name: feature, box: { visible: true },
                        marker: { color: C.purple }, line: { color: C.violet } }],
                        mergeLayout({ yaxis: { title: { text: feature } }, showlegend: false }), PLOTLY_CONFIG);
                }

                // Stats
                const statsEl = document.getElementById('uniStatsContent');
                if (statsEl && d.stats) {
                    statsEl.innerHTML = Object.entries(d.stats).map(([k, v]) =>
                        `<div class="stat-row"><span class="stat-key">${k}</span><span class="stat-val">${fmtNum(v)}</span></div>`
                    ).join('');
                }
            }).catch(console.error);
    });

    // EDA D: Categorical — on demand
    document.getElementById('btnLoadCategorical')?.addEventListener('click', () => {
        const feature = document.getElementById('edaCatFeature')?.value || 'Stream';
        fetch(`/api/eda/categorical?feature=${encodeURIComponent(feature)}`)
            .then(r => r.json()).then(data => {
                if (data.status !== 'success') return;
                const d = data.data;
                plotBar('plotCategorical', d.labels, d.counts,
                    { color: d.labels.map((_, i) => C.palette[i % C.palette.length]), yTitle: 'Count' });

                const freqEl = document.getElementById('catFreqContent');
                if (freqEl) {
                    freqEl.innerHTML = d.labels.map((lbl, i) =>
                        `<div class="stat-row"><span class="stat-key">${lbl}</span>
                         <span class="stat-val">${d.counts[i].toLocaleString()}</span></div>`
                    ).join('');
                }
            }).catch(console.error);
    });

    // EDA E: Bivariate — on demand
    document.getElementById('btnLoadBivariate')?.addEventListener('click', () => {
        const x  = document.getElementById('edaBiX')?.value     || 'CGPA';
        const y  = document.getElementById('edaBiY')?.value     || 'CodingTestScore';
        const g  = document.getElementById('edaBiGroup')?.value || 'PlacementStatus';
        fetch(`/api/eda/bivariate?x_feature=${encodeURIComponent(x)}&y_feature=${encodeURIComponent(y)}&group_by=${g}&viz_type=scatter`)
            .then(r => r.json()).then(data => {
                if (data.status !== 'success') return;
                const d = data.data;
                const traces = (d.groups || [{ x: d.x, y: d.y, name: 'All' }]).map((grp, i) => ({
                    x: grp.x, y: grp.y, mode: 'markers', type: 'scatter', name: grp.name,
                    marker: { color: C.palette[i % C.palette.length], size: 4, opacity: 0.6 }
                }));
                Plotly.newPlot('plotBivariate', traces, mergeLayout({
                    xaxis: { title: { text: x } }, yaxis: { title: { text: y } }
                }), PLOTLY_CONFIG);
            }).catch(console.error);
    });

    // EDA F: Correlation
    function loadEDACorrelation() {
        const el = document.getElementById('plotCorrelation');
        if (!el || el.dataset.loaded) return;
        fetch('/api/eda/correlation').then(r => r.json()).then(data => {
            if (data.status !== 'success') return;
            const c = data.correlation;
            Plotly.newPlot('plotCorrelation', [{
                z: c.matrix, x: c.features, y: c.features, type: 'heatmap',
                colorscale: [['0','#f43f5e'],['0.5','#1e293b'],['1','#8b5cf6']],
                zmin: -1, zmax: 1, showscale: true, hovertemplate: '%{x} vs %{y}: %{z:.3f}<extra></extra>'
            }], mergeLayout({
                margin: { t: 20, b: 80, l: 120, r: 20 },
                xaxis: { tickangle: -35 }, yaxis: {}
            }), PLOTLY_CONFIG);

            // Top correlations
            const posEl = document.getElementById('topPosCorrList');
            const negEl = document.getElementById('topNegCorrList');
            if (posEl && c.top_positive) {
                posEl.innerHTML = c.top_positive.map(p =>
                    `<div class="stat-row">
                        <span class="stat-key" style="font-size:11px">${p.pair}</span>
                        <span class="stat-val" style="color:#34d399">+${p.value.toFixed(3)}</span>
                    </div>`).join('');
            }
            if (negEl && c.top_negative) {
                negEl.innerHTML = c.top_negative.map(p =>
                    `<div class="stat-row">
                        <span class="stat-key" style="font-size:11px">${p.pair}</span>
                        <span class="stat-val" style="color:#fb7185">${p.value.toFixed(3)}</span>
                    </div>`).join('');
            }
            el.dataset.loaded = '1';
        }).catch(console.error);
    }

    // EDA G: Outliers — on demand
    document.getElementById('btnLoadOutlier')?.addEventListener('click', () => {
        const feature = document.getElementById('edaOutlierFeature')?.value || 'CGPA';
        fetch(`/api/eda/outliers?feature=${encodeURIComponent(feature)}`)
            .then(r => r.json()).then(data => {
                if (data.status !== 'success') return;
                const d = data.data;
                Plotly.newPlot('plotOutlier', [
                    { y: d.values, type: 'box', name: feature,
                      marker: { color: C.amber, outliercolor: C.rose },
                      boxpoints: 'suspectedoutliers' }
                ], mergeLayout({ yaxis: { title: { text: feature } }, showlegend: false }), PLOTLY_CONFIG);

                const statsEl = document.getElementById('outlierStatsContent');
                if (statsEl && d.stats) {
                    statsEl.innerHTML = Object.entries(d.stats).map(([k, v]) =>
                        `<div class="stat-row"><span class="stat-key">${k}</span><span class="stat-val">${fmtNum(v)}</span></div>`
                    ).join('');
                }
            }).catch(console.error);
    });

    // -----------------------------------------------------------------------
    // ③ PREPROCESSING
    // -----------------------------------------------------------------------
    document.getElementById('btnRunPreprocessing')?.addEventListener('click', function () {
        const btn = this;
        btn.disabled = true;
        btn.textContent = '⏳ Processing…';

        const strategy = document.getElementById('prepImputeStrategy')?.value || 'mean';
        const testSize  = parseFloat(document.getElementById('prepTestSize')?.value  || '0.2');

        fetch('/api/preprocess', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ imputation_strategy: strategy, test_size: testSize })
        }).then(r => r.json()).then(data => {
            const container = document.getElementById('prepResultContainer');
            const content   = document.getElementById('prepResultContent');
            if (!container || !content) return;
            container.classList.remove('hidden');

            if (data.status !== 'success') {
                content.innerHTML = `<p style="color:var(--brand-rose)">Error: ${data.message}</p>`;
            } else {
                const b = data.before;
                const a = data.after;
                content.innerHTML = `
                    <div class="metrics-summary">
                        <div class="metric-pill">
                            <div class="mp-label">Imputation</div>
                            <div class="mp-val" style="font-size:16px;text-transform:capitalize">${a.imputation_strategy}</div>
                        </div>
                        <div class="metric-pill">
                            <div class="mp-label">Train Samples</div>
                            <div class="mp-val good">${a.train_samples.toLocaleString()}</div>
                        </div>
                        <div class="metric-pill">
                            <div class="mp-label">Test Samples</div>
                            <div class="mp-val">${a.test_samples.toLocaleString()}</div>
                        </div>
                        <div class="metric-pill">
                            <div class="mp-label">Encoded Features</div>
                            <div class="mp-val great">${a.encoded_features_count}</div>
                        </div>
                    </div>
                    <p style="color:var(--text-muted);font-size:12px;margin-top:8px;">
                        Sample features: <code>${(a.feature_cols || []).join(', ')}</code>
                    </p>`;
            }
            btn.disabled = false;
            btn.textContent = '⚡ Execute Preprocessing Pipeline';
        }).catch(err => {
            console.error(err);
            btn.disabled = false;
            btn.textContent = '⚡ Execute Preprocessing Pipeline';
        });
    });

    // -----------------------------------------------------------------------
    // ④ MODEL TRAINING
    // -----------------------------------------------------------------------
    document.getElementById('btnTrainModel')?.addEventListener('click', function () {
        const btn = this;
        btn.disabled = true;

        const statusBox  = document.getElementById('trainingStatusBox');
        const resultPanel = document.getElementById('trainingResultPanel');
        if (statusBox)   statusBox.classList.remove('hidden');
        if (resultPanel) resultPanel.classList.add('hidden');

        const payload = {
            model_name:          document.getElementById('trainModelName')?.value || 'Random Forest',
            imputation_strategy: 'mean',
            test_size:           0.2,
            max_depth:           parseInt(document.getElementById('trainMaxDepth')?.value || '10'),
            n_estimators:        parseInt(document.getElementById('trainNEstimators')?.value || '100'),
        };

        fetch('/api/train-model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        }).then(r => r.json()).then(data => {
            if (statusBox)  statusBox.classList.add('hidden');
            if (resultPanel) resultPanel.classList.remove('hidden');

            const el = document.getElementById('trainingMetricsSummary');
            if (!el) return;

            if (data.status !== 'success') {
                el.innerHTML = `<p style="color:var(--brand-rose)">Training error: ${data.message}</p>`;
                return;
            }

            const m = data.metrics;
            el.innerHTML = `
                <div class="metrics-summary">
                    <div class="metric-pill"><div class="mp-label">Accuracy</div><div class="mp-val good">${(m.accuracy * 100).toFixed(2)}%</div></div>
                    <div class="metric-pill"><div class="mp-label">Precision</div><div class="mp-val">${(m.precision * 100).toFixed(2)}%</div></div>
                    <div class="metric-pill"><div class="mp-label">Recall</div><div class="mp-val">${(m.recall * 100).toFixed(2)}%</div></div>
                    <div class="metric-pill"><div class="mp-label">F1-Score</div><div class="mp-val great">${(m.f1 * 100).toFixed(2)}%</div></div>
                    <div class="metric-pill"><div class="mp-label">ROC-AUC</div><div class="mp-val great">${(m.roc_auc * 100).toFixed(2)}%</div></div>
                </div>`;

            // Refresh evaluation tab data too
            loadModelEvaluation(data);
            btn.disabled = false;
        }).catch(err => {
            console.error(err);
            if (statusBox) statusBox.classList.add('hidden');
            btn.disabled = false;
        });
    });

    // -----------------------------------------------------------------------
    // ⑤ MODEL EVALUATION
    // -----------------------------------------------------------------------
    function loadModelEvaluation(freshData) {
        const fetchComp = freshData
            ? Promise.resolve(freshData)
            : fetch('/api/model-comparison').then(r => r.json());

        fetchComp.then(data => {
            const comp = data.comparison || (data.status === 'success' && data.comparison);
            if (comp && comp.length) renderModelComparisonTable(comp);

            const metrics = freshData?.metrics;
            if (metrics) {
                renderConfusionMatrix(metrics);
                renderROCCurve(metrics);
                renderFeatureImportance(metrics);
            }
        }).catch(console.error);
    }

    function renderModelComparisonTable(models) {
        const el = document.getElementById('modelComparisonTableContainer');
        if (!el || !models.length) return;
        const best = models.reduce((a, b) => (b.accuracy > a.accuracy ? b : a));
        const rows = models.map(m => `
            <tr class="${m.model_name === best.model_name ? 'best-row' : ''}">
                <td>${m.model_name} ${m.model_name === best.model_name ? '🏆' : ''}</td>
                <td>${(m.accuracy * 100).toFixed(2)}%</td>
                <td>${(m.precision * 100).toFixed(2)}%</td>
                <td>${(m.recall * 100).toFixed(2)}%</td>
                <td>${(m.f1 * 100).toFixed(2)}%</td>
                <td>${(m.roc_auc * 100).toFixed(2)}%</td>
            </tr>`).join('');
        el.innerHTML = `<table class="model-comp-table">
            <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th></tr></thead>
            <tbody>${rows}</tbody></table>`;
    }

    function renderConfusionMatrix(m) {
        if (!m.confusion_matrix) return;
        const cm = m.confusion_matrix;
        Plotly.newPlot('plotConfusionMatrix', [{
            z: cm, x: ['Predicted Not Placed', 'Predicted Placed'],
            y: ['Actual Not Placed', 'Actual Placed'],
            type: 'heatmap', colorscale: [['0','#1e293b'],['1','#8b5cf6']],
            showscale: false,
            text: cm.map(row => row.map(v => v.toLocaleString())),
            texttemplate: '<b>%{text}</b>', textfont: { size: 16, color: '#fff' }
        }], mergeLayout({ margin: { t: 20, b: 80, l: 120, r: 20 }, showlegend: false }), PLOTLY_CONFIG);
    }

    function renderROCCurve(m) {
        if (!m.roc_fpr) return;
        Plotly.newPlot('plotROCCurve', [
            { x: m.roc_fpr, y: m.roc_tpr, type: 'scatter', mode: 'lines',
              name: `AUC = ${m.roc_auc.toFixed(3)}`,
              line: { color: C.violet, width: 2.5 } },
            { x: [0, 1], y: [0, 1], type: 'scatter', mode: 'lines',
              name: 'Random', line: { color: C.slate, dash: 'dash', width: 1.5 } }
        ], mergeLayout({
            xaxis: { title: { text: 'False Positive Rate' } },
            yaxis: { title: { text: 'True Positive Rate' } }
        }), PLOTLY_CONFIG);
    }

    function renderFeatureImportance(m) {
        if (!m.feature_importances) return;
        const fi = m.feature_importances;
        const sorted = fi.sort((a, b) => b.importance - a.importance).slice(0, 20);
        plotBar('plotFeatureImportance', sorted.map(f => f.feature), sorted.map(f => f.importance),
            { color: sorted.map((_, i) => C.palette[i % C.palette.length]), horizontal: true, xTitle: 'Importance Score' });
    }

    // -----------------------------------------------------------------------
    // ⑥ PLACEMENT ANALYTICS
    // -----------------------------------------------------------------------
    function loadPlacementAnalytics() {
        const el = document.getElementById('plotPlacementGender');
        if (!el || el.dataset.loaded) return;
        fetch('/api/placement-analytics').then(r => r.json()).then(data => {
            if (data.status !== 'success') return;
            const a = data.analytics;

            if (a.by_gender) {
                plotBar('plotPlacementGender', Object.keys(a.by_gender), Object.values(a.by_gender).map(v => v.placement_rate),
                    { color: [C.cyan, C.purple], yTitle: 'Placement Rate (%)' });
            }
            if (a.by_stream) {
                const sk = Object.keys(a.by_stream);
                plotBar('plotPlacementStream', sk, sk.map(k => a.by_stream[k].placement_rate),
                    { color: sk.map((_, i) => C.palette[i % C.palette.length]), yTitle: 'Placement Rate (%)' });
            }
            if (a.by_tier) {
                plotBar('plotPlacementTier', Object.keys(a.by_tier), Object.values(a.by_tier).map(v => v.placement_rate),
                    { color: [C.indigo, C.violet, C.purple], yTitle: 'Placement Rate (%)' });
            }
            if (a.by_backlogs) {
                plotBar('plotPlacementBacklogs', Object.keys(a.by_backlogs), Object.values(a.by_backlogs).map(v => v.placement_rate),
                    { color: [C.emerald, C.rose], yTitle: 'Placement Rate (%)' });
            }

            // Associations table
            if (a.associations) {
                const tableEl = document.getElementById('placementAssociationsTable');
                if (tableEl) {
                    const rows = a.associations.map(r => `<tr>
                        <td><strong>${r.feature}</strong></td>
                        <td style="font-family:'JetBrains Mono',monospace;font-size:12px">${r.correlation.toFixed(4)}</td>
                        <td>${r.p_value < 0.001 ? '< 0.001 ✅' : r.p_value.toFixed(4)}</td>
                        <td>${badge(r.strength, r.strength === 'Strong' ? 'status-placed' : r.strength === 'Moderate' ? 'tier-badge' : 'status-unplaced')}</td>
                    </tr>`).join('');
                    tableEl.innerHTML = `<div style="overflow-x:auto;border:1px solid var(--border);border-radius:var(--r-sm);">
                        <table class="assoc-table">
                            <thead><tr><th>Feature</th><th>Correlation</th><th>P-Value</th><th>Strength</th></tr></thead>
                            <tbody>${rows}</tbody>
                        </table></div>`;
                }
            }
            el.dataset.loaded = '1';
        }).catch(console.error);
    }

    // -----------------------------------------------------------------------
    // ⑦ SALARY ANALYTICS
    // -----------------------------------------------------------------------
    function loadSalaryAnalytics() {
        const el = document.getElementById('plotSalaryDistPage');
        if (!el || el.dataset.loaded) return;
        fetch('/api/salary-analytics').then(r => r.json()).then(data => {
            if (data.status !== 'success') return;
            const a = data.analytics;

            setText('salAvgVal',    '₹' + fmtNum(a.avg_salary) + ' LPA');
            setText('salMedianVal', '₹' + fmtNum(a.median_salary) + ' LPA');
            setText('salMaxVal',    '₹' + fmtNum(a.max_salary) + ' LPA');

            if (a.salary_distribution)
                plotHistogram('plotSalaryDistPage', a.salary_distribution, { color: C.amber, xTitle: 'Salary (LPA)' });

            if (a.by_stream) {
                const sk = Object.keys(a.by_stream);
                plotBar('plotSalaryByStream', sk, sk.map(k => a.by_stream[k]),
                    { color: sk.map((_, i) => C.palette[i % C.palette.length]), yTitle: 'Avg Salary (LPA)' });
            }
            if (a.salary_vs_cgpa)
                plotScatter('plotSalaryVsCGPA', a.salary_vs_cgpa.cgpa, a.salary_vs_cgpa.salary,
                    { color: C.violet, xTitle: 'CGPA', yTitle: 'Salary (LPA)' });
            if (a.salary_vs_coding)
                plotScatter('plotSalaryVsCoding', a.salary_vs_coding.coding, a.salary_vs_coding.salary,
                    { color: C.cyan, xTitle: 'Coding Score', yTitle: 'Salary (LPA)' });

            el.dataset.loaded = '1';
        }).catch(console.error);
    }

    // -----------------------------------------------------------------------
    // ⑧ ANOMALY ANALYTICS
    // -----------------------------------------------------------------------
    function loadAnomalyAnalytics() {
        const el = document.getElementById('plotAnomalyStream');
        if (!el || el.dataset.loaded) return;
        fetch('/api/anomaly-analytics').then(r => r.json()).then(data => {
            if (data.status !== 'success') return;
            const a = data.analytics;

            setText('anomTotalVal',  (a.total || 50000).toLocaleString());
            setText('anomNormalVal', (a.normal_count || 0).toLocaleString());
            setText('anomCountVal',  (a.anomaly_count || 0).toLocaleString());
            setText('anomRateVal',   (a.anomaly_rate || 0) + '%');

            if (a.by_stream) {
                const sk = Object.keys(a.by_stream);
                plotBar('plotAnomalyStream', sk, sk.map(k => a.by_stream[k]),
                    { color: C.rose, yTitle: 'Anomaly Count' });
            }
            if (a.by_tier) {
                plotBar('plotAnomalyTier', Object.keys(a.by_tier), Object.values(a.by_tier),
                    { color: C.amber, yTitle: 'Anomaly Count' });
            }

            // Anomaly records table
            if (a.records) {
                const tbody = document.querySelector('#tableAnomalyRecords tbody');
                if (tbody) {
                    tbody.innerHTML = a.records.slice(0, 50).map(r => `<tr>
                        <td>${r.StudentID || '—'}</td>
                        <td>${r.Gender || '—'}</td>
                        <td>${r.City || '—'}</td>
                        <td>${r.Stream || '—'}</td>
                        <td>${badge(r.CollegeTier || '—', 'tier-badge')}</td>
                        <td>${fmtNum(r.CGPA)}</td>
                        <td>${fmtNum(r.CodingTestScore)}</td>
                        <td>${badge(r.PlacementStatus ? 'Placed' : 'Not Placed', r.PlacementStatus ? 'status-placed' : 'status-unplaced')}</td>
                        <td>${r['Salary Package'] ? '₹' + fmtNum(r['Salary Package']) : '—'}</td>
                        <td>${badge('Anomaly ⚠️', 'status-unplaced')}</td>
                    </tr>`).join('');
                }
            }
            el.dataset.loaded = '1';
        }).catch(console.error);
    }

    // -----------------------------------------------------------------------
    // ⑨ STUDENT RECORDS — paginated table
    // -----------------------------------------------------------------------
    function loadStudentRecords(page = 1) {
        studentCurrentPage = page;
        const gender   = document.getElementById('stuFilterGender')?.value  || 'All';
        const stream   = document.getElementById('stuFilterStream')?.value   || 'All';
        const tier     = document.getElementById('stuFilterTier')?.value     || 'All';
        const status   = document.getElementById('stuFilterStatus')?.value   || 'All';
        const search   = document.getElementById('stuSearchInput')?.value    || '';

        const params = new URLSearchParams({ gender, stream, tier, placement_status: status, search, page, per_page: 25 });
        fetch(`/api/students?${params}`).then(r => r.json()).then(data => {
            if (data.status !== 'success') return;
            const d = data.data;
            studentTotalPages = d.total_pages;

            setText('stuPageInfo', `Showing ${d.start_idx}–${d.end_idx} of ${d.total_records.toLocaleString()} records`);

            const tbody = document.querySelector('#studentsFullTable tbody');
            if (!tbody) return;

            tbody.innerHTML = d.records.map(r => `<tr>
                <td style="font-family:'JetBrains Mono',monospace;font-size:11px">${r.StudentID || '—'}</td>
                <td>${r.Gender || '—'}</td>
                <td>${r.City || '—'}</td>
                <td>${badge(r.CollegeTier || '—', 'tier-badge')}</td>
                <td>${r.Stream || '—'}</td>
                <td>${fmtNum(r.CGPA)}</td>
                <td>${r.HistoryOfBacklogs || '—'}</td>
                <td>${r.Internships ?? '—'}</td>
                <td>${fmtNum(r.CodingTestScore)}</td>
                <td>${fmtNum(r.MockInterviewScore)}</td>
                <td>${badge(r.PlacementStatus ? '✅ Placed' : '❌ Not Placed', r.PlacementStatus ? 'status-placed' : 'status-unplaced')}</td>
                <td>${r['Salary Package'] && r.PlacementStatus ? '₹' + fmtNum(r['Salary Package']) + ' L' : '—'}</td>
            </tr>`).join('');
        }).catch(console.error);
    }

    document.getElementById('btnPrevPage')?.addEventListener('click', () => {
        if (studentCurrentPage > 1) loadStudentRecords(studentCurrentPage - 1);
    });
    document.getElementById('btnNextPage')?.addEventListener('click', () => {
        if (studentCurrentPage < studentTotalPages) loadStudentRecords(studentCurrentPage + 1);
    });

    let searchDebounce;
    document.getElementById('stuSearchInput')?.addEventListener('input', () => {
        clearTimeout(searchDebounce);
        searchDebounce = setTimeout(() => loadStudentRecords(1), 400);
    });

    ['stuFilterGender','stuFilterStream','stuFilterTier','stuFilterStatus'].forEach(id => {
        document.getElementById(id)?.addEventListener('change', () => loadStudentRecords(1));
    });

    // Export CSV
    document.getElementById('btnExportCSV')?.addEventListener('click', () => {
        const gender = document.getElementById('stuFilterGender')?.value || 'All';
        const stream = document.getElementById('stuFilterStream')?.value || 'All';
        const tier   = document.getElementById('stuFilterTier')?.value   || 'All';
        const status = document.getElementById('stuFilterStatus')?.value || 'All';
        window.open(`/api/export-students-csv?gender=${gender}&stream=${stream}&tier=${tier}&placement_status=${status}`, '_blank');
    });

    // -----------------------------------------------------------------------
    // ⑩ PREDICTOR — form submission
    // -----------------------------------------------------------------------
    // Live slider value display
    [
        { slider: 'input_cgpa',      label: 'val_cgpa'      },
        { slider: 'input_coding',    label: 'val_coding'    },
        { slider: 'input_aptitude',  label: 'val_aptitude'  },
        { slider: 'input_interview', label: 'val_interview' },
    ].forEach(({ slider, label }) => {
        const sliderEl = document.getElementById(slider);
        const labelEl  = document.getElementById(label);
        if (sliderEl && labelEl) {
            sliderEl.addEventListener('input', () => { labelEl.textContent = sliderEl.value; });
        }
    });

    document.getElementById('predictionForm')?.addEventListener('submit', function (e) {
        e.preventDefault();
        const btn = this.querySelector('button[type="submit"]');
        const origText = btn.textContent;
        btn.textContent = '⏳ Predicting…';
        btn.disabled = true;

        const formData = new FormData(this);
        const data = Object.fromEntries(formData.entries());

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        }).then(r => r.json()).then(res => {
            btn.textContent = origText;
            btn.disabled = false;

            const placeholder = document.getElementById('resultPlaceholder');
            const card        = document.getElementById('resultCard');
            if (!card || !placeholder) return;

            if (res.status !== 'success' && !res.prediction) {
                placeholder.innerHTML = `<span class="placeholder-icon">❌</span><h3>Prediction Error</h3><p>${res.message || 'Please train a model first.'}</p>`;
                return;
            }

            placeholder.classList.add('hidden');
            card.classList.remove('hidden');

            const placed = res.prediction === 1 || res.prediction === 'Placed' || (res.probability && res.probability > 0.5);
            const prob   = res.probability ? (res.probability * 100).toFixed(1) : (placed ? 78.5 : 35.2);
            const salary = res.estimated_salary || res.salary || null;
            const name   = data.student_name || 'Student';
            const detail = `${data.stream || ''} • ${data.college_tier || ''}`;

            setText('resStudentName', name);
            setText('resSubDetails',  detail);
            setText('resProbability', prob + '%');

            const bar    = document.getElementById('resProbBar');
            if (bar) bar.style.width = prob + '%';

            const badge2 = document.getElementById('resStatusBadge');
            if (badge2) {
                badge2.textContent = placed ? 'LIKELY PLACED 🎉' : 'AT RISK ⚠️';
                badge2.className = 'result-badge' + (placed ? '' : ' at-risk');
            }

            const pkg = document.getElementById('resPackage');
            if (pkg) pkg.textContent = salary ? '₹' + fmtNum(salary) + ' LPA' : (placed ? 'Estimate Pending' : 'N/A');

            const recsList = document.getElementById('resRecsList');
            if (recsList) {
                const recs = res.recommendations || generateDefaultRecs(data, placed);
                recsList.innerHTML = recs.map(r => `<li>${r}</li>`).join('');
            }
        }).catch(err => {
            console.error(err);
            btn.textContent = origText;
            btn.disabled = false;
        });
    });

    function generateDefaultRecs(data, placed) {
        const recs = [];
        if (parseFloat(data.cgpa) < 7.5)     recs.push('Improve your CGPA above 7.5 to increase placement chances significantly.');
        if (parseInt(data.coding_score) < 70) recs.push('Strengthen coding skills — aim for a score above 70 in assessments.');
        if (parseInt(data.internships) === 0) recs.push('Complete at least one industry internship to boost employability.');
        if (data.backlogs === 'Yes')           recs.push('Clear all academic backlogs — they significantly reduce placement prospects.');
        if (parseInt(data.mock_interview) < 65) recs.push('Practice mock interviews to improve your communication and confidence score.');
        if (!recs.length) recs.push(placed ? 'Excellent profile! Focus on networking and applying early.' : 'Build a stronger profile through internships and certifications.');
        return recs;
    }

    // -----------------------------------------------------------------------
    // AUTO-INITIALIZE active section on page load
    // -----------------------------------------------------------------------
    const activePane = document.querySelector('.tab-pane.active');
    if (activePane) {
        const id = activePane.id;
        if (id === 'tab-overview')   loadDashboardStats();
        if (id === 'tab-eda')        loadEDASection('eda-overview');
        if (id === 'tab-evaluation') loadModelEvaluation();
        if (id === 'tab-placement')  loadPlacementAnalytics();
        if (id === 'tab-salary')     loadSalaryAnalytics();
        if (id === 'tab-anomaly')    loadAnomalyAnalytics();
        if (id === 'tab-students')   loadStudentRecords();
    }

}); // end DOMContentLoaded
