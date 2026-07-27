// API 基础地址
const API_BASE = '/api';

// DOM 元素
const form = document.getElementById('generateForm');
const arxivInput = document.getElementById('arxivId');
const promptTextarea = document.getElementById('customPrompt');
const submitBtn = document.getElementById('submitBtn');

const progressSection = document.getElementById('progressSection');
const progressBar = document.getElementById('progressBar');
const progressMessage = document.getElementById('progressMessage');
const progressPercent = document.getElementById('progressPercent');

const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

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
    errorSection.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                arxiv_id: arxivId,
                custom_prompt: promptTextarea.value.trim()
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
        85: 4,  // 优化
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

    // 滚动到结果
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
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
    window.scrollTo({ top: 0, behavior: 'smooth' });
    arxivInput.focus();
});

// 初始化
checkHealth();
setInterval(checkHealth, 30000);