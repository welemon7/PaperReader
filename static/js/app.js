// API 基础地址
const API_BASE = '/api';

// DOM 元素
const form = document.getElementById('generateForm');
const arxivInput = document.getElementById('arxivId');
const promptTextarea = document.getElementById('customPrompt');
const thresholdInput = document.getElementById('qualityThreshold');
const maxRoundsInput = document.getElementById('maxRounds');
const submitBtn = document.getElementById('submitBtn');

const progressSection = document.getElementById('progressSection');
const progressBar = document.getElementById('progressBar');
const progressMessage = document.getElementById('progressMessage');
const progressPercent = document.getElementById('progressPercent');

const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

// Harness 面板
const harnessPanel = document.getElementById('harnessPanel');
const harnessBadge = document.getElementById('harnessBadge');
const scoreHistoryEl = document.getElementById('scoreHistory');
const roundSnapshotsEl = document.getElementById('roundSnapshots');
const harnessIssuesEl = document.getElementById('harnessIssues');
const harnessReportLink = document.getElementById('harnessReportLink');

// 任务轮询
let pollInterval = null;
let currentTaskId = null;

// 检查服务健康状态
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        const badge = document.getElementById('healthStatus');
        const dot = badge.querySelector('.dot');
        const text = badge.querySelector('span:last-child');

        if (data.status === 'ok') {
            dot.className = 'dot online';
            text.textContent = `在线 (${data.model || 'LLM'})`;
        } else {
            dot.className = 'dot offline';
            text.textContent = '服务异常';
        }
    } catch (e) {
        const badge = document.getElementById('healthStatus');
        badge.querySelector('.dot').className = 'dot offline';
        badge.querySelector('span:last-child').textContent = '连接失败';
    }
}

// 提交表单
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const arxivId = arxivInput.value.trim();
    if (!arxivId) {
        showError('请输入 arXiv ID');
        return;
    }

    // 禁用按钮
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="btn-icon">⏳</span> 生成中...';

    // 隐藏旧结果
    resultSection.style.display = 'none';
    harnessPanel.style.display = 'none';
    errorSection.style.display = 'none';

    const qualityThreshold = parseInt(thresholdInput.value, 10) || 8;
    const maxRounds = parseInt(maxRoundsInput.value, 10) || 5;

    try {
        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                arxiv_id: arxivId,
                custom_prompt: promptTextarea.value.trim(),
                quality_threshold: qualityThreshold,
                max_rounds: maxRounds,
                enable_qa_eval: true
            })
        });

        const data = await response.json();
        if (data.error) {
            showError(data.error);
            return;
        }

        // 开始轮询状态
        currentTaskId = data.task_id;
        progressSection.style.display = 'block';
        startPolling(currentTaskId);

    } catch (error) {
        showError('提交失败: ' + error.message);
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span class="btn-icon">🚀</span> 生成海报';
    }
});

// 轮询任务状态
function startPolling(taskId) {
    if (pollInterval) {
        clearInterval(pollInterval);
    }

    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/status/${taskId}`);
            const data = await response.json();

            if (data.error) {
                stopPolling();
                showError(data.error);
                return;
            }

            updateProgress(data);

            if (data.status === 'complete') {
                stopPolling();
                showResult(data);
            } else if (data.status === 'error') {
                stopPolling();
                showError(data.error || '生成失败');
            }

        } catch (error) {
            // 网络错误时继续轮询
            console.warn('Status check failed:', error);
        }
    }, 2000);
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<span class="btn-icon">🚀</span> 生成海报';
}

// 更新进度
function updateProgress(data) {
    const progress = data.progress || 0;
    progressBar.style.width = `${progress}%`;
    progressPercent.textContent = `${Math.round(progress)}%`;
    progressMessage.textContent = data.message || '处理中...';

    // 更新步骤
    const steps = document.querySelectorAll('.step');
    const stepMap = {
        20: 0,  // 解析
        40: 1,  // 理解
        55: 2,  // 设计
        70: 3,  // 初稿
        85: 4,  // 视觉审查
        100: 4  // 完成
    };

    let activeStep = 0;
    for (const [threshold, stepIndex] of Object.entries(stepMap)) {
        if (progress >= parseInt(threshold)) {
            activeStep = stepIndex;
        }
    }

    steps.forEach((step, index) => {
        step.classList.remove('active', 'completed');
        if (index < activeStep) {
            step.classList.add('completed');
        } else if (index === activeStep) {
            step.classList.add('active');
        }
    });
}

// 显示结果
function showResult(data) {
    resultSection.style.display = 'block';
    progressSection.style.display = 'none';

    document.getElementById('resultArxivId').textContent = data.arxiv_id || '未知';
    document.getElementById('resultOutputDir').textContent = data.output_dir || '未知';

    // 设置预览和下载链接
    const taskId = data.task_id;
    document.getElementById('viewDraft').href = `${API_BASE}/view/${taskId}/draft`;
    document.getElementById('viewOptimized').href = `${API_BASE}/view/${taskId}/optimized`;
    document.getElementById('downloadDraft').href = `${API_BASE}/download/${taskId}?file=draft`;
    document.getElementById('downloadOptimized').href = `${API_BASE}/download/${taskId}?file=optimized`;
    document.getElementById('downloadAllBtn').onclick = () => {
        window.location.href = `${API_BASE}/download/${taskId}`;
    };

    // Harness 状态行
    const harnessStatus = data.harness_status;
    const harnessLine = document.getElementById('resultHarnessLine');
    const harnessStatusEl = document.getElementById('resultHarnessStatus');
    if (harnessStatus) {
        harnessLine.style.display = '';
        harnessStatusEl.textContent = harnessStatusLabel(harnessStatus);
    } else {
        harnessLine.style.display = 'none';
    }

    // 加载视觉审查报告
    loadHarness(taskId);

    // 滚动到结果
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function harnessStatusLabel(status) {
    if (status === 'passed') return '✅ 已达标（视觉审查通过）';
    if (status === 'fallback') return '🔄 视觉审查不可用，已回退到单次优化';
    return '⚠️ 未完全达标，已保留最优版本';
}

// 加载并渲染 harness 报告
async function loadHarness(taskId) {
    try {
        const response = await fetch(`${API_BASE}/harness/${taskId}`);
        const data = await response.json();
        if (!data || data.error) return;

        harnessPanel.style.display = 'block';
        const report = data.harness_report;
        const rounds = (report && report.rounds) || data.harness_rounds || [];

        if (!rounds.length) {
            // 回退模式：没有逐轮审查记录
            harnessBadge.textContent = '未执行视觉审查（回退模式）';
            harnessBadge.className = 'harness-badge fallback';
            scoreHistoryEl.innerHTML = '';
            roundSnapshotsEl.innerHTML = '';
            harnessIssuesEl.innerHTML = '<p class="muted-note">当前输出为单次 LLM 优化结果，无逐轮评分历史。</p>';
            harnessReportLink.href = `${API_BASE}/harness/${taskId}`;
            return;
        }

        // 状态徽标
        harnessBadge.textContent = harnessStatusLabel(data.harness_status);
        harnessBadge.className = 'harness-badge ' + (data.harness_status || 'done');

        // 评分历史
        const scores = report.scores || rounds.map(r => r.quality_score);
        renderScoreHistory(scores, report.threshold || 8);

        // 逐轮快照
        renderSnapshots(rounds, taskId);

        // 问题列表
        renderIssues(rounds);

        harnessReportLink.href = `${API_BASE}/harness/${taskId}`;

    } catch (error) {
        console.warn('Harness report load failed:', error);
    }
}

function renderScoreHistory(scores, threshold) {
    if (!scoreHistoryEl) return;
    const max = Math.max(10, ...scores, threshold);
    const bars = scores.map((score, idx) => {
        const height = Math.max(4, Math.round((score / max) * 100));
        const reached = score >= threshold;
        return `
            <div class="score-col" title="第 ${idx + 1} 轮评分 ${score}/10">
                <div class="score-bar ${reached ? 'reached' : ''}" style="height: ${height}%">
                    <span class="score-val">${score}</span>
                </div>
                <div class="score-label">R${idx + 1}</div>
            </div>`;
    }).join('');
    const thresholdLine = `<div class="threshold-line" style="bottom: ${Math.max(4, Math.round((threshold / max) * 100))}%">
        <span>阈值 ${threshold}</span></div>`;
    scoreHistoryEl.innerHTML = `
        <div class="score-chart">
            ${thresholdLine}
            <div class="score-bars">${bars}</div>
        </div>`;
}

function renderSnapshots(rounds, taskId) {
    if (!roundSnapshotsEl) return;
    const cards = rounds.map(r => `
        <div class="round-card ${r.needs_improvement ? '' : 'passed'}">
            <div class="round-head">
                <strong>第 ${r.round_no} 轮</strong>
                <span class="round-score ${r.quality_score >= 8 ? 'good' : ''}">${r.quality_score}/10</span>
            </div>
            <img class="round-img" src="${API_BASE}/round_image/${taskId}/${r.round_no}"
                 alt="round ${r.round_no} poster" loading="lazy"
                 onerror="this.style.display='none'">
            ${r.summary ? `<p class="round-summary">${escapeHtml(r.summary)}</p>` : ''}
        </div>`).join('');
    roundSnapshotsEl.innerHTML = cards;
}

function renderIssues(rounds) {
    if (!harnessIssuesEl) return;
    const sections = rounds.map(r => {
        const issues = (r.issues || []).map(issue => `
            <li class="issue-item ${issue.severity || 'warning'}">
                <span class="issue-sev">${(issue.severity || 'info').toUpperCase()}</span>
                <span class="issue-text">${escapeHtml(issue.issue)}</span>
                ${issue.target ? `<code>${escapeHtml(issue.target)}</code>` : ''}
                ${issue.suggestion ? `<span class="issue-suggestion">→ ${escapeHtml(issue.suggestion)}</span>` : ''}
                <span class="issue-action">[${escapeHtml(issue.action || 'rewrite')}]</span>
            </li>`).join('');
        return `
            <div class="issue-round">
                <h5>第 ${r.round_no} 轮 · 评分 ${r.quality_score}/10</h5>
                ${issues ? `<ul>${issues}</ul>` : '<p class="muted-note">无问题记录</p>'}
            </div>`;
    }).join('');
    harnessIssuesEl.innerHTML = `<h5 class="issue-heading">逐轮问题与反馈</h5>${sections}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

// 显示错误
function showError(message) {
    errorSection.style.display = 'block';
    errorMessage.textContent = message;
    progressSection.style.display = 'none';
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<span class="btn-icon">🚀</span> 生成海报';

    errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// 重试
document.getElementById('retryBtn').addEventListener('click', () => {
    errorSection.style.display = 'none';
    form.dispatchEvent(new Event('submit'));
});

// 新任务
document.getElementById('newTaskBtn').addEventListener('click', () => {
    resultSection.style.display = 'none';
    harnessPanel.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
    arxivInput.focus();
});

// 初始化
checkHealth();
setInterval(checkHealth, 30000);
