document.addEventListener('DOMContentLoaded', () => {
    // Initial UI setup
    loadLocalConfigs();
    updateHistoryView();
    setupEventListeners();
    updateParallelismOptions();
    validateParallelismLive();
    toggleEpBasedOnConfig(); // Disable EP initially
    toggleVppDependentOptions(); // 初始化 VPP 相关复选框显隐
});

// Utility: convert ANSI color codes (red 31, green 32) to HTML spans for display
function ansiToHtml(str) {
    if (!str) return '';
    // Replace known ANSI codes
    return str
        .replace(/\u001b\[31m/g, '<span class="ansi-red">')
        .replace(/\u001b\[32m/g, '<span class="ansi-green">')
        .replace(/\u001b\[33m/g, '<span class="ansi-yellow">')
        .replace(/\u001b\[34m/g, '<span class="ansi-blue">')
        .replace(/\u001b\[35m/g, '<span class="ansi-magenta">')
        .replace(/\u001b\[36m/g, '<span class="ansi-cyan">')
        .replace(/\u001b\[0m/g, '</span>');
}

function setupEventListeners() {
    document.getElementById('config-form').addEventListener('submit', (e) => {
        e.preventDefault();
        submitForm();
    });

    document.getElementById('model-select').addEventListener('change', loadSelectedModelConfig);
    
    document.getElementById('recompute-granularity').addEventListener('change', (e) => {
        const recomputeOptions = document.querySelectorAll('.recompute-options');
        recomputeOptions.forEach(opt => {
            opt.style.display = e.target.value === 'full' ? 'block' : 'none';
        });

        // 新增：Selective 模式下展示复选框
        const selectiveOptions = document.querySelectorAll('.selective-options');
        selectiveOptions.forEach(opt => {
            opt.style.display = e.target.value === 'selective' ? 'block' : 'none';
        });
    });

    const liveValidationInputs = ['num-gpus', 'tp', 'pp', 'ep', 'cp', 'etp', 'vpp', 'config-editor', 'pipeline-layout'];
    liveValidationInputs.forEach(id => {
        const input = document.getElementById(id);
        if(input) {
            input.addEventListener('change', validateParallelismLive);
            if (id === 'num-gpus') {
                input.addEventListener('change', updateParallelismOptions);
            }
            if (id === 'vpp') {
                input.addEventListener('change', toggleVppDependentOptions);
            }
        }
    });

    document.getElementById('config-editor').addEventListener('input', toggleEpBasedOnConfig);
    document.getElementById('history-table').addEventListener('click', handleHistoryAction);
    document.getElementById('clear-history').addEventListener('click', clearHistory);
        }


async function loadLocalConfigs() {
    const modelSelect = document.getElementById('model-select');
    const defaultConfigName = 'Qwen/Qwen3-235B-A22B'; // Updated default model

        try {
        const response = await fetch('/local-hf-configs');
        const configs = await response.json();
        
        modelSelect.innerHTML = '<option value="">Select a model...</option>';
        // Add custom option to allow user supplied configs
        modelSelect.innerHTML += '<option value="__custom__">Custom (paste JSON below)...</option>';
        configs.forEach(config => {
            modelSelect.innerHTML += `<option value="${config}">${config}</option>`;
        });

        // Check if the default config exists and select it
        if (configs.includes(defaultConfigName)) {
            modelSelect.value = defaultConfigName;
            // Await the loading of the model config to ensure it's ready
            await loadSelectedModelConfig();
            }

        } catch (error) {
        modelSelect.innerHTML = '<option value="">Error loading configs</option>';
        console.error('Error loading local configs:', error);
        }
    }

async function loadSelectedModelConfig() {
    const modelSelect = document.getElementById('model-select');
    const editor = document.getElementById('config-editor');
    const selectedConfig = modelSelect.value;
    const messageDiv = document.getElementById('validation-message'); // move early for use in all branches
    let configData = null; // declare for wider scope
    
    if (!selectedConfig) {
        editor.value = '';
        toggleEpBasedOnConfig();
        if (messageDiv) messageDiv.style.display = 'none';
        return;
    } else if (selectedConfig === '__custom__') {
        // Custom config: do not fetch, user must paste JSON
        editor.value = '';
        toggleEpBasedOnConfig();
        if (messageDiv) messageDiv.style.display = 'none';
        return;
    }

        // 优先直接从 HuggingFace 仓库拉取配置文件
        const hfUrl = `https://huggingface.co/${selectedConfig}/raw/main/config.json`;
        try {
            const resp = await fetch(hfUrl, { mode: 'cors' });
            if (resp.ok) {
                configData = await resp.json();
                editor.value = JSON.stringify(configData, null, 2);
            } else {
                throw new Error(`HF returned status ${resp.status}`);
            }
        } catch (hfErr) {
            console.warn('Direct HF fetch failed, fallback to backend:', hfErr);
            // 回退到后端接口（兼容本地部署无 CORS 或私有模型）
            try {
                const response = await fetch(`/get-megatron-config/${encodeURIComponent(selectedConfig)}`);
                configData = await response.json();
                if (configData.error) {
                    editor.value = `Error: ${configData.error}`;
                } else {
                    editor.value = JSON.stringify(configData, null, 2);
                }
            } catch (beErr) {
                editor.value = 'Failed to fetch model configuration.';
                console.error('Backend config fetch error:', beErr);
            }
        }

    // Trigger validation and UI updates after loading new config
    validateParallelismLive();
    toggleEpBasedOnConfig();

    // Show Kimi-K2-Instruct warning if needed
    if (selectedConfig.includes('Kimi-K2-Instruct') && configData && configData.model_type !== 'deepseek_v3') {
        messageDiv.textContent = 'Notice: For Kimi-K2-Instruct the config field "model_type" must be set to "deepseek_v3" before memory estimation.';
        messageDiv.style.display = 'block';
    } else if (messageDiv) {
        messageDiv.style.display = 'none';
    }
}


function getFormValues(isSubmission = false) {
    const form = document.getElementById('config-form');
    const formData = new FormData(form);
    const modelSelect = document.getElementById('model-select');
    
    const hfPath = modelSelect.value;
        if (!hfPath) {
        // We will now handle this case in the submitForm function instead of an alert.
        return null;
    }

    const editor = document.getElementById('config-editor');
    let customConfig = null;
    try {
        // Only parse if the editor has content
        if (editor.value) {
            customConfig = JSON.parse(editor.value);
        }
    } catch (e) {
        // Only alert on final submission, not on live validation
        if (isSubmission) {
            // alert('Model Config is not valid JSON.'); // Removing alert
        }
        return null; // Return null if JSON is invalid
        }

    const vppInput = formData.get('vpp');
    const etpInput = formData.get('etp');
    const pipelineLayoutInput = formData.get('pipeline_model_parallel_layout');

    // 新增：收集 selective 模式下用户选择的模块
    const recomputeModules = formData.getAll('recompute_modules');

    return {
            hf_model_path: hfPath,
        custom_hf_config: customConfig, // Renamed for clarity
        num_gpus: parseInt(formData.get('num_gpus')),
        mbs: parseInt(formData.get('mbs')),
        seq_len: parseInt(formData.get('seq-len')),
        use_distributed_optimizer: document.getElementById('use-distributed-optimizer').checked,
        recompute_granularity: formData.get('recompute_granularity'),
        recompute_method: formData.get('recompute_method'),
        recompute_num_layers: parseInt(formData.get('recompute_num_layers')),
        // 新增字段
        recompute_modules: recomputeModules,
        tp: parseInt(formData.get('tp')),
        pp: parseInt(formData.get('pp')),
        ep: parseInt(formData.get('ep')) || 1, // Default to 1 if disabled/null
        cp: parseInt(formData.get('cp')),
        vpp: vppInput ? parseInt(vppInput) : null,
        etp: etpInput ? parseInt(etpInput) : null,
        num_layers_in_first_pipeline_stage: formData.get('num_layers_in_first_pipeline_stage') ? parseInt(formData.get('num_layers_in_first_pipeline_stage')) : null,
        num_layers_in_last_pipeline_stage: formData.get('num_layers_in_last_pipeline_stage') ? parseInt(formData.get('num_layers_in_last_pipeline_stage')) : null,
        pipeline_model_parallel_layout: pipelineLayoutInput ? pipelineLayoutInput.trim() : null,
        overhead: parseInt(formData.get('overhead')),
        // 新增:
        account_for_embedding_in_pipeline_split: document.getElementById('account_for_embedding_in_pipeline_split').checked,
        account_for_loss_in_pipeline_split: document.getElementById('account_for_loss_in_pipeline_split').checked,
    };
}

async function submitForm() {
    const messageDiv = document.getElementById('validation-message');
    messageDiv.textContent = '';
    messageDiv.style.display = 'none';

    // Get all form values first. We use getFormValues(false) to avoid any legacy alerts
    // and handle all validation directly within this function for clarity.
    const formValues = getFormValues(false);

    // === START SUBMISSION VALIDATION ===

    // 1. Check if form values could be retrieved. This catches both missing model selection
    //    and invalid JSON, as getFormValues returns null in those cases.
    if (!formValues) {
        if (!document.getElementById('model-select').value) {
            messageDiv.textContent = 'Validation Error: Please select a model config.';
        } else {
            messageDiv.textContent = 'Validation Error: Model Config is not valid JSON.';
        }
        messageDiv.style.display = 'block';
        return;
    }

    // Custom config must have valid JSON
    if (document.getElementById('model-select').value === '__custom__' && !formValues.custom_hf_config) {
        messageDiv.textContent = 'Validation Error: Please paste a valid model configuration JSON for the custom model.';
        messageDiv.style.display = 'block';
        return;
    }

    // 2. Perform all numeric and parallelism validation.
    const { num_gpus, tp, pp, ep, cp, etp, custom_hf_config } = formValues;
    const num_kv_heads = custom_hf_config?.num_key_value_heads || null;
    
    let errors = [];
    if (tp * pp * cp > num_gpus) {
        errors.push(`TP*PP*CP (${tp * pp * cp}) > GPUs (${num_gpus}).`);
    }
    if (etp){
        if (etp * pp * cp * ep > num_gpus) {
            errors.push(`ETP*PP*CP*EP (${etp * pp * cp * ep}) > GPUs (${num_gpus}).`);
        }
    } else {
        if (tp * pp * cp * ep > num_gpus) {
            errors.push(`TP*PP*CP*EP (${tp * pp * cp * ep}) > GPUs (${num_gpus}) when ETP is not set.`);
        }
    }
    if (num_kv_heads && tp > num_kv_heads) {
        errors.push(`TP (${tp}) > Num KV Heads (${num_kv_heads}).`);
    }

    if (errors.length > 0) {
        messageDiv.textContent = 'Validation Error: ' + errors.join(' ');
        messageDiv.style.display = 'block';
        return;
    }
    // === END SUBMISSION VALIDATION ===

    const loading = document.getElementById('loading');
    const submitBtn = document.querySelector('#config-form button[type="submit"]');
    loading.style.display = 'block';
    if (submitBtn) submitBtn.disabled = true;

        try {
            const response = await fetch('/estimate_with_mbridge', {
                method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formValues) // Send the now fully-validated formValues
            });

        console.log('Response Status:', response.status);

        if (response.ok) {
            const data = await response.json();

            // FIX: Ensure history wrapper is visible before updating and showing details
            document.getElementById('history-wrapper').style.display = 'block';

            saveToHistory(formValues, data);
            updateHistoryView();
            const newEntryRow = document.querySelector('#history-table tbody tr:first-child');
            if (newEntryRow) {
                const detailBtn = newEntryRow.querySelector('.detail-btn');
                if (detailBtn) {
                    // We need to pass the event object structure to handleHistoryAction
                    handleHistoryAction({ target: detailBtn });
                }
            }
        } else {
            const error = await response.text();
            console.error('Server error response:', error);
            // Since we removed the main results display, show error in the validation div
            messageDiv.textContent = `Server Error: ${error}`;
            messageDiv.style.display = 'block';
        }
    } catch (error) {
        console.error('Fetch API Error:', error);
        messageDiv.textContent = `Client Error: ${error.message}`;
        messageDiv.style.display = 'block';
    } finally {
        loading.style.display = 'none';
        if (submitBtn) submitBtn.disabled = false;
    }
}

function renderTable(details, rawFullReport) {
    if (!details || details.length === 0) {
        return '<p>No detailed memory breakdown available.</p>';
    }

    const headers = Object.keys(details[0]);
    headers.push('Breakdown');

    let table = '<table><thead><tr>';
    headers.forEach(h => table += `<th>${h}</th>`);
    table += '</tr></thead><tbody>';

    details.forEach(row => {
        const ppRank = row.pp_rank;
        // FIX: Look in the full raw report array passed in.
        const rawDataForRank = rawFullReport ? rawFullReport.find(r => r.pp_rank === ppRank) : null;
        
        // FIX: Change to `let` to allow modification for highlighting.
        let modelBreakdown = (rawDataForRank && rawDataForRank.model_breakdown) 
            ? rawDataForRank.model_breakdown 
            : 'No breakdown available.';

        // Add syntax-like highlighting for params and activations
        // Basic HTML escaping for safety before inserting spans
        modelBreakdown = modelBreakdown.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        modelBreakdown = modelBreakdown
            .replace(/(n_params=[0-9.]+[a-zA-Z]*)/g, '<span class="highlight-red">$1</span>')
            .replace(/(n_act=[0-9.]+[a-zA-Z]*)/g, '<span class="highlight-red">$1</span>');

        // Main row with data
        table += `<tr data-pp-rank="${ppRank}">`;
        headers.forEach(h => {
            if (h !== 'Breakdown') {
                table += `<td>${row[h]}</td>`;
            }
        });
        table += `<td><button class="action-btn raw-per-rank-btn" data-pp-rank="${ppRank}">Raw</button></td>`;
        table += '</tr>';

        // Hidden row for the breakdown
        table += `<tr class="raw-breakdown-row" data-pp-rank="${ppRank}" style="display: none;">
                    <td colspan="${headers.length}">
                        <pre>${modelBreakdown}</pre>
                    </td>
                  </tr>`;
    });

    table += '</tbody></table>';
    return table;
}

function saveToHistory(params, resultData) {
    let history = JSON.parse(localStorage.getItem('estimationHistory')) || [];
    const historyEntry = {
        params: params,
        result: resultData, // Store the full result object { processed_report, raw_report }
        id: new Date().getTime()
    };
    history.unshift(historyEntry); // Add to the beginning
    if (history.length > 20) { // Keep history size manageable
        history.pop();
    }
    localStorage.setItem('estimationHistory', JSON.stringify(history));
}

function updateHistoryView() {
    const history = JSON.parse(localStorage.getItem('estimationHistory')) || [];
    const historyTableBody = document.querySelector('#history-table tbody');
    const historyWrapper = document.getElementById('history-wrapper');
    historyTableBody.innerHTML = '';

    if (history.length === 0) {
        historyWrapper.style.display = 'none';
        return;
    }

    historyWrapper.style.display = 'block';

    history.forEach(item => {
        const row = document.createElement('tr');
        
        const params = item.params;
        const resultData = item.result || {};
        
        // FIX: Handle both old and new data structures for compatibility.
        const details = (resultData.report && resultData.report.details) ? resultData.report.details : (resultData.processed_report || []);
        const pp0Result = details.find(r => r.pp_rank === 0) || details[0] || {};

        const modelName = params.hf_model_path.split('/').pop();
        
        // Build parallelism string, e.g., "TP2 PP2 VPP2"
        const parallelismParts = [];
        ['tp', 'pp', 'ep', 'cp', 'vpp', 'etp'].forEach(p => {
            const value = params[p];
            if (value && value > 1) {
                parallelismParts.push(`${p.toUpperCase()}${value}`);
            }
        });
        const parallelismInfo = parallelismParts.join(' ') || 'No Parallelism';

        const overheadGb = params.overhead ? parseInt(params.overhead) : 0;
        const baseTotal = details.length > 0 ? Math.max(...details.map(r => r.total_gb || 0)) : null;
        const totalGb = baseTotal !== null ? (baseTotal + overheadGb).toFixed(2) : 'N/A';
        
        const seqLen = params.seq_len || 0;
        const formattedSeqLen = seqLen >= 1024 ? `${seqLen / 1024}k` : seqLen;
        const sequenceInfo = `${params.mbs || 'N/A'}*${formattedSeqLen}`;

        // 新增：生成重算方式描述
        let recomputeInfo = '';
        switch (params.recompute_granularity) {
            case 'none':
                recomputeInfo = 'Recompute: None';
                break;
            case 'full':
                const method = params.recompute_method || 'uniform';
                const layers = params.recompute_num_layers ? params.recompute_num_layers : '';
                recomputeInfo = `Recompute: Full (${method}${layers ? ',' + layers + 'L' : ''})`;
                break;
            case 'selective':
                const mods = Array.isArray(params.recompute_modules) && params.recompute_modules.length ? params.recompute_modules.join('+') : '';
                recomputeInfo = `Recompute: Selective${mods ? ' (' + mods + ')' : ''}`;
                break;
            default:
                recomputeInfo = '';
        }

        row.innerHTML = `
            <td>
                <div>${modelName}</div>
                <div class="model-meta-info">
                    <span>GPUs: ${params.num_gpus || 'N/A'}</span>
                    <span>${parallelismInfo}</span>
                    <span>Sequence: ${sequenceInfo}</span>
                    ${recomputeInfo ? `<span>${recomputeInfo}</span>` : ''}
                </div>
            </td>
            <td>${pp0Result.weight_grad_optim_gb || 'N/A'}</td>
            <td>${pp0Result.activation_gb || 'N/A'}</td>
            <td>${totalGb}</td>
            <td>
                <button class="restore-btn" data-id="${item.id}">Restore</button>
                <button class="detail-btn" data-id="${item.id}">Detail</button>
                <button class="delete-btn" data-id="${item.id}">Delete</button>
            </td>
        `;
        historyTableBody.appendChild(row);
    });
}

async function handleHistoryAction(e) {
    const button = e.target.closest('button');
    if (!button) return;

    // Handle breakdown toggle first
    if (button.classList.contains('breakdown-btn')) {
        const ppRank = button.dataset.ppRank;
        const detailTable = button.closest('table');
        if (!detailTable) return;

        const breakdownRow = detailTable.querySelector(`tr.breakdown-row[data-pp-rank="${ppRank}"]`);
        if (!breakdownRow) return;

        const isVisible = breakdownRow.style.display !== 'none';
        breakdownRow.style.display = isVisible ? 'none' : 'table-row';
        button.textContent = isVisible ? 'Breakdown' : 'Hide';
        return; // Do not continue to other handlers
    }

    if (!button.matches('.detail-btn, .restore-btn, .delete-btn')) return;

    const id = parseInt(button.dataset.id, 10);
    const history = JSON.parse(localStorage.getItem('estimationHistory')) || [];
    const entry = history.find(item => item.id === id);

    if (!entry) {
        console.error('History entry not found for id:', id);
            return;
        }

    const row = button.closest('tr');

    if (button.classList.contains('detail-btn')) {
        const isDetailsVisible = row.nextElementSibling && row.nextElementSibling.classList.contains('detail-row');

        document.querySelectorAll('.detail-row').forEach(detailRow => {
            const prevRow = detailRow.previousElementSibling;
            const detailBtn = prevRow.querySelector('.detail-btn');
            if (detailRow !== row.nextElementSibling) {
                detailRow.remove();
                if (detailBtn) detailBtn.textContent = 'Detail';
            }
        });

        if (isDetailsVisible) {
            row.nextElementSibling.remove();
            button.textContent = 'Detail';
        } else {
            const detailRow = document.createElement('tr');
            detailRow.classList.add('detail-row');
            const detailCell = detailRow.insertCell();
            detailCell.colSpan = row.cells.length;

            // FIX: Handle both old and new data structures for compatibility.
            const report = entry.result.report;
            const details = (report && report.details) ? report.details : (entry.result.processed_report || []);
            const modelBreakdown = (report && report.model_breakdown) ? report.model_breakdown : null;

            if (details && details.length > 0) {
                const newTable = document.createElement('table');
                // Determine if breakdown information exists per-row or globally
                let headers = Object.keys(details[0]);

                // If old-format data, there is a 'model_breakdown' key on each detail row
                const hasRowBreakdown = headers.includes('model_breakdown');

                // Remove the raw model_breakdown column from headers to keep table compact
                if (hasRowBreakdown) {
                    headers = headers.filter(h => h !== 'model_breakdown');
                }

                // Include global breakdown if provided, or row breakdowns if present
                const includeBreakdown = hasRowBreakdown || (modelBreakdown && typeof modelBreakdown === 'string');

                if (includeBreakdown) {
                    headers.push('Breakdown');
                }

                const headerRow = newTable.insertRow();
                headers.forEach(h => {
                    const th = document.createElement('th');
                    th.textContent = h;
                    headerRow.appendChild(th);
                });

                details.forEach(detail => {
                    const newRow = newTable.insertRow();
                    headers.forEach(header => {
                        if (header === 'Breakdown') {
                            const cell = newRow.insertCell();
                            cell.innerHTML = `<button class="breakdown-btn" data-pp-rank="${detail.pp_rank}">Breakdown</button>`;
                        } else {
                            const cell = newRow.insertCell();
                            let value = detail[header];
                            if (typeof value === 'number' && !Number.isInteger(value)) {
                                value = value.toFixed(4);
                            }
                            cell.textContent = value;
                        }
                    });

                    // Hidden breakdown row
                    if (includeBreakdown) {
                        const breakdownRow = newTable.insertRow();
                        breakdownRow.classList.add('breakdown-row');
                        breakdownRow.dataset.ppRank = detail.pp_rank;
                        breakdownRow.style.display = 'none';
                        const breakdownCell = breakdownRow.insertCell();
                        breakdownCell.colSpan = headers.length;
                        const rowSpecificBreakdown = hasRowBreakdown ? (detail.model_breakdown || '') : modelBreakdown;
                        const htmlBreakdown = ansiToHtml(rowSpecificBreakdown);
                        breakdownCell.innerHTML = `<pre class="model-breakdown-view">${htmlBreakdown || 'No breakdown available.'}</pre>`;
                    }
                });
                
                detailCell.appendChild(newTable);
            } else {
                detailCell.innerHTML = 'No detailed per-rank results available.';
            }

            row.after(detailRow);
            button.textContent = 'Hide';
        }
    } else if (button.classList.contains('restore-btn')) {
        restoreForm(entry.params);
    } else if (button.classList.contains('delete-btn')) {
        deleteHistoryEntry(id);
    }
}

function deleteHistoryEntry(id) {
    let history = JSON.parse(localStorage.getItem('estimationHistory')) || [];
    const updatedHistory = history.filter(item => item.id != id);
    localStorage.setItem('estimationHistory', JSON.stringify(updatedHistory));
    updateHistoryView();
    
    // If history is now empty, hide the whole output container
    if (updatedHistory.length === 0) {
        // document.getElementById('output-container').style.display = 'none';
    }
}

function clearHistory() {
    localStorage.removeItem('estimationHistory');
    updateHistoryView();
    // document.getElementById('output-container').style.display = 'none';
}


function restoreForm(params) {
    if (!params) return;

    const setElementValue = (id, value, defaultValue = '') => {
        const element = document.getElementById(id);
        if (element) {
            if (element.type === 'checkbox') {
                element.checked = value ?? defaultValue;
            } else {
                element.value = value ?? defaultValue;
            }
        }
    };

    setElementValue('num-gpus', params.num_gpus, 8);
    setElementValue('mbs', params.mbs, 1);
    setElementValue('seq-len', params.seq_len, 4096);
    setElementValue('use-distributed-optimizer', params.use_distributed_optimizer, true);
    setElementValue('recompute_granularity', params.recompute_granularity, 'selective');
    setElementValue('recompute_method', params.recompute_method, 'uniform');
    setElementValue('recompute_num_layers', params.recompute_num_layers, 1);
    setElementValue('tp', params.tp, 1);
    setElementValue('pp', params.pp, 1);
    setElementValue('ep', params.ep, 1);
    setElementValue('cp', params.cp, 1);
    setElementValue('vpp', params.vpp);
    // 在设置 vpp 之后更新依赖显示
    toggleVppDependentOptions();
    setElementValue('etp', params.etp);
    setElementValue('num_layers_in_first_pipeline_stage', params.num_layers_in_first_pipeline_stage);
    setElementValue('num_layers_in_last_pipeline_stage', params.num_layers_in_last_pipeline_stage);
    setElementValue('pipeline-layout', params.pipeline_model_parallel_layout);
    setElementValue('overhead', params.overhead, 10);
    
    // 新增 checkbox 恢复
    setElementValue('account_for_embedding_in_pipeline_split', params.account_for_embedding_in_pipeline_split, false);
    setElementValue('account_for_loss_in_pipeline_split', params.account_for_loss_in_pipeline_split, false);
    
    const modelSelect = document.getElementById('model-select');
    if (modelSelect && params.hf_model_path) {
        modelSelect.value = params.hf_model_path;
    }
    
    // Manually trigger change event for UI updates
    const recomputeSelect = document.getElementById('recompute_granularity');
    if (recomputeSelect) {
        recomputeSelect.dispatchEvent(new Event('change'));
            }
} 

function updateParallelismOptions() {
    const numGpusInput = document.getElementById('num-gpus');
    if (!numGpusInput) return;

    const numGpus = parseInt(numGpusInput.value);
    if (isNaN(numGpus) || numGpus <= 0) {
        return; // Don't update if GPU count is invalid
    }
    
    const tpSelect = document.getElementById('tp');
    const epSelect = document.getElementById('ep');
    const cpSelect = document.getElementById('cp');
    
    // PP is now a manual input, so we only handle TP, EP, CP here.
    const selects = [tpSelect, epSelect, cpSelect];
    
    const powersOfTwo = [1];
    for (let i = 1; (1 << i) <= numGpus; i++) {
        powersOfTwo.push(1 << i);
    }

    selects.forEach(select => {
        if (!select) return;
        const currentVal = select.value;
        select.innerHTML = ''; // Clear existing options
        
        powersOfTwo.forEach(val => {
            const option = document.createElement('option');
            option.value = val;
            option.textContent = val;
            select.appendChild(option);
        });

        // Try to restore the previous value, otherwise default to 1
        if (powersOfTwo.includes(parseInt(currentVal))) {
            select.value = currentVal;
        } else {
            select.value = 1;
        }
    });
} 

function validateParallelismLive() {
    const messageDiv = document.getElementById('validation-message');
    // Pass isSubmission = false to getFormValues to prevent alerts during live validation
    const formValues = getFormValues(false);

    if (!formValues) {
        messageDiv.textContent = '';
        return true; 
        }

    const { num_gpus, tp, pp, ep, cp, etp, custom_hf_config } = formValues;
    // The key is the same in the HF config, so this logic remains valid.
    const num_kv_heads = custom_hf_config?.num_key_value_heads || null;
    
    let errors = [];
    if (tp * pp * cp > num_gpus) {
        errors.push(`TP*PP*CP (${tp*pp*cp}) > GPUs (${num_gpus}).`);
        }
    if (etp) {
        if (etp * pp * cp * ep > num_gpus) {
            errors.push(`ETP*PP*CP*EP (${etp*pp*cp*ep}) > GPUs (${num_gpus}).`);
        }
    } else {
        if (tp * pp * cp * ep > num_gpus) {
            errors.push(`TP*PP*CP*EP (${tp*pp*cp*ep}) > GPUs (${num_gpus}) when ETP is not set.`);
        }
    }
    if (num_kv_heads && tp > num_kv_heads) {
        errors.push(`TP (${tp}) > Num KV Heads (${num_kv_heads}).`);
    }

    if (errors.length > 0) {
        messageDiv.textContent = 'Validation Error: ' + errors.join(' ');
        messageDiv.style.display = 'block';
    } else {
        messageDiv.textContent = '';
        messageDiv.style.display = 'none';
    }
    return errors.length === 0;
} 

function toggleEpBasedOnConfig() {
    const editor = document.getElementById('config-editor');
    const epSelect = document.getElementById('ep');
    if (!editor || !epSelect) return;

    let config = null;
    try {
        if (editor.value) {
            config = JSON.parse(editor.value);
        }
    } catch (e) {
        // Invalid JSON, disable EP as a safety measure
        epSelect.disabled = true;
        return;
    }

    if (config && config.num_experts_per_tok) {
        epSelect.disabled = false;
    } else {
        epSelect.disabled = true;
        epSelect.value = 1; // Reset to 1 if disabled
    }
} 

// 新增：根据 vpp 输入显示/隐藏依赖选项
function toggleVppDependentOptions() {
    const vppInput = document.getElementById('vpp');
    const dependents = document.querySelectorAll('.vpp-dependent');
    if (!vppInput) return;
    const shouldShow = vppInput.value && parseInt(vppInput.value) > 0;
    dependents.forEach(el => {
        el.style.display = shouldShow ? 'block' : 'none';
    });
} 