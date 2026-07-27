from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import threading
import tempfile
import zipfile
from io import BytesIO

# 添加项目根目录到路径
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.agents.parse_agent import run_parse_paper
from src.agents.understand_agent import run_understand_paper
from src.agents.poster_planner import generate_blueprint, normalize_analysis_for_poster
from src.agents.poster_v2 import run_poster_v2
from src.renderers.html_renderer import HtmlPosterRenderer
from src.agents.html_optimizer import optimize_html_with_llm
from src.storage.sqlite import PaperDatabase
from src.utils.output_paths import resolve_paper_output_dir
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 配置
app.config['OUTPUT_DIR'] = Path('output')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['TEMP_DIR'] = Path('temp_downloads')

# 确保目录存在
app.config['OUTPUT_DIR'].mkdir(exist_ok=True)
app.config['TEMP_DIR'].mkdir(exist_ok=True)

# ✅ 提示词文件路径（项目根目录）
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "LLM-up.txt"

# 全局任务状态
tasks = {}


class TaskStatus:
    """任务状态管理"""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = 'pending'
        self.progress = 0
        self.message = '任务已创建'
        self.result = None
        self.error = None
        self.arxiv_id = None
        self.output_dir = None
        self.html_draft = None
        self.html_optimized = None

    def to_dict(self):
        return {
            'task_id': self.task_id,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'result': self.result,
            'error': self.error,
            'arxiv_id': self.arxiv_id,
            'html_draft': self.html_draft,
            'html_optimized': self.html_optimized,
        }


def get_default_prompt() -> str:
    """获取默认提示词内容"""
    if DEFAULT_PROMPT_PATH.exists():
        return DEFAULT_PROMPT_PATH.read_text(encoding='utf-8')
    else:
        # 如果文件不存在，返回内置默认提示词
        logger.warning(f"默认提示词文件不存在: {DEFAULT_PROMPT_PATH}")
        return _BUILTIN_PROMPT


# ✅ 内置默认提示词（当 LLM-up.txt 不存在时使用）
_BUILTIN_PROMPT = """## HTML Optimization Instructions

Please optimize this academic poster HTML with the following improvements:

### Visual Improvements
1. **Typography**: Use a clear font hierarchy with appropriate sizes
2. **Spacing**: Improve padding and margins for better readability
3. **Color Scheme**: Use a professional academic palette (navy blue + gold accents)
4. **Layout**: Ensure balanced column widths and section spacing

### Content Preservation
- Keep ALL scientific content exactly as is
- Preserve ALL formulas and equations
- Maintain ALL figure references
- Do not change any factual information

### Technical Requirements
- Keep MathJax configuration intact
- Maintain responsive design
- Preserve all CSS styling
- Output ONLY valid HTML

### Style Guidelines
- Clean, modern academic design
- Professional color palette
- Readable font sizes
- Clear visual hierarchy
- Balanced whitespace

Please return the complete optimized HTML document."""


def generate_poster_task(task_id: str, arxiv_id: str, custom_prompt: str | None = None):
    """后台任务：生成海报"""
    task = tasks.get(task_id)
    if not task:
        return

    task.status = 'running'
    task.progress = 10
    task.message = f'开始处理 arXiv: {arxiv_id}'
    task.arxiv_id = arxiv_id

    try:
        # ---- Phase 1: Parse ----
        task.message = '正在下载和解析论文...'
        task.progress = 20
        logger.info(f"Task {task_id}: Phase 1 - Parse")
        doc = run_parse_paper(arxiv_id, force=True)

        # ---- Phase 2: Understand (LLM) ----
        task.message = '正在理解论文内容 (LLM)...'
        task.progress = 40
        logger.info(f"Task {task_id}: Phase 2 - Understand")
        analysis = run_understand_paper(arxiv_id)
        analysis = normalize_analysis_for_poster(analysis)

        # ---- Phase 3: Plan ----
        task.message = '正在生成海报蓝图...'
        task.progress = 55
        logger.info(f"Task {task_id}: Phase 3 - Plan")
        output_dir = resolve_paper_output_dir(app.config['OUTPUT_DIR'], arxiv_id)
        blueprint = generate_blueprint(doc, analysis)

        # ---- Phase 4: Render (初稿) ----
        task.message = '正在渲染初稿 HTML...'
        task.progress = 70
        logger.info(f"Task {task_id}: Phase 4 - Render Draft")
        renderer = HtmlPosterRenderer()
        draft_path = output_dir / "poster_draft.html"
        renderer.render_to_file(blueprint, doc, draft_path, optimize_with_llm=False)

        # ---- Phase 5: Optimize (LLM) ----
        task.message = '正在使用 LLM 优化海报...'
        task.progress = 85
        logger.info(f"Task {task_id}: Phase 5 - Optimize")

        # ✅ 优先使用用户自定义提示词，否则使用项目根目录的 LLM-up.txt
        if custom_prompt:
            # 使用用户自定义提示词
            prompt_path = app.config['TEMP_DIR'] / f"{task_id}_custom_prompt.txt"
            prompt_path.write_text(custom_prompt, encoding='utf-8')
            user_prompt = prompt_path
        else:
            # 使用项目根目录的 LLM-up.txt
            user_prompt = DEFAULT_PROMPT_PATH
            if not user_prompt.exists():
                logger.warning(f"LLM-up.txt not found at {user_prompt}, using built-in prompt")
                user_prompt = app.config['TEMP_DIR'] / f"{task_id}_default_prompt.txt"
                user_prompt.write_text(get_default_prompt(), encoding='utf-8')

        optimized_path = output_dir / "poster_optimized.html"
        try:
            optimize_html_with_llm(
                html_path=draft_path,
                prompt_path=user_prompt,
                output_path=optimized_path,
            )
            task.html_optimized = str(optimized_path)
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            # 如果优化失败，复制初稿作为优化版本
            shutil.copy(draft_path, optimized_path)
            task.html_optimized = str(optimized_path)

        task.html_draft = str(draft_path)
        task.progress = 100
        task.status = 'complete'
        task.message = '海报生成完成！'
        task.result = {
            'arxiv_id': arxiv_id,
            'draft': str(draft_path),
            'optimized': str(optimized_path),
            'output_dir': str(output_dir),
        }

        logger.info(f"Task {task_id}: Complete")

    except Exception as e:
        logger.exception(f"Task {task_id} failed")
        task.status = 'error'
        task.error = str(e)
        task.message = f'生成失败: {str(e)}'


# ============ Routes ============

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    """提交生成任务"""
    data = request.get_json()
    arxiv_id = data.get('arxiv_id', '').strip()
    custom_prompt = data.get('custom_prompt', '')

    if not arxiv_id:
        return jsonify({'error': '请提供 arXiv ID'}), 400

    # 生成任务 ID
    import uuid
    task_id = str(uuid.uuid4())[:8]

    # 创建任务状态
    task = TaskStatus(task_id)
    task.arxiv_id = arxiv_id
    tasks[task_id] = task

    # 在后台线程中运行
    thread = threading.Thread(
        target=generate_poster_task,
        args=(task_id, arxiv_id, custom_prompt if custom_prompt else None),
        daemon=True
    )
    thread.start()

    return jsonify({'task_id': task_id, 'status': 'pending'})


@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """获取任务状态"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404

    return jsonify(task.to_dict())


@app.route('/api/download/<task_id>', methods=['GET'])
def download(task_id):
    """下载生成的文件"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404

    if task.status != 'complete':
        return jsonify({'error': '任务未完成'}), 400

    output_dir = Path(task.result['output_dir'])

    # 创建包含所有文件的 ZIP
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 遍历输出目录中的所有文件
        for file_path in output_dir.rglob('*'):
            if file_path.is_file():
                # 保持相对路径结构
                rel_path = file_path.relative_to(output_dir.parent)  # 保留 arxiv_id 目录结构
                zip_file.write(file_path, str(rel_path))

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'{task.arxiv_id}_poster_package.zip'
    )

@app.route('/api/view/<task_id>/<version>', methods=['GET'])
def view_html(task_id, version):
    """预览 HTML"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404

    if version == 'draft':
        html_path = task.html_draft
    elif version == 'optimized':
        html_path = task.html_optimized
    else:
        return jsonify({'error': '无效版本'}), 400

    if not html_path or not Path(html_path).exists():
        return jsonify({'error': '文件不存在'}), 404

    return send_file(html_path, mimetype='text/html')


@app.route('/api/prompt', methods=['GET'])
def get_prompt():
    """获取默认提示词内容"""
    content = get_default_prompt()
    return jsonify({
        'content': content,
        'path': str(DEFAULT_PROMPT_PATH) if DEFAULT_PROMPT_PATH.exists() else None
    })


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'llm_configured': bool(settings.llm_api_key),
        'model': settings.llm_model,
        'prompt_exists': DEFAULT_PROMPT_PATH.exists(),
        'prompt_path': str(DEFAULT_PROMPT_PATH),
    })


import webbrowser
import threading
import time

if __name__ == '__main__':
    # 检查提示词文件
    if DEFAULT_PROMPT_PATH.exists():
        logger.info(f"✅ 使用提示词文件: {DEFAULT_PROMPT_PATH}")
    else:
        logger.warning(f"⚠️ 提示词文件不存在: {DEFAULT_PROMPT_PATH}")
        logger.info("将使用内置默认提示词")

    if not os.environ.get('WERKZEUG_RUN_MAIN'):
        def open_browser():
            """延迟打开浏览器"""
            time.sleep(1)
            webbrowser.open('http://localhost:5000')

        threading.Thread(target=open_browser, daemon=True).start()
        logger.info("🌐 正在打开浏览器...")

    # 启动服务
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )