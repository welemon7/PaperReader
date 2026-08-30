from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import threading
import time
import tempfile
import zipfile
from io import BytesIO
import json
# 添加项目根目录到路径
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.agents.parse_agent import run_parse_paper
from src.agents.understand_agent import run_understand_paper
from src.agents.poster_planner import generate_blueprint, normalize_analysis_for_poster
from src.agents.poster_v2 import run_poster_v2
from src.agents.poster_harness import run_poster_harness
from src.renderers.html_renderer import HtmlPosterRenderer
from src.schemas.poster_harness import HarnessConfig
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
        self.harness_rounds = []
        self.harness_status = None
        self.harness_report = None
        self.best_png = None
        self.started_at = None
        self.finished_at = None
        self.estimated_total_seconds = 180

    def to_dict(self):
        now = time.time()
        if self.started_at is None:
            elapsed_seconds = 0
        else:
            end_time = self.finished_at or now
            elapsed_seconds = max(0, int(end_time - self.started_at))
        progress_ratio = max(0.0, min(1.0, float(self.progress or 0) / 100.0))
        if self.status == 'complete':
            remaining_seconds = 0
        elif self.status == 'error':
            remaining_seconds = 0
        elif progress_ratio > 0:
            # Use observed throughput after the first real phase.  Keep a
            # small baseline for the initial startup period, when progress is
            # intentionally coarse and a linear projection is unstable.
            projected_total = max(
                float(self.estimated_total_seconds),
                elapsed_seconds / progress_ratio,
            )
            remaining_seconds = max(1, int(round(projected_total - elapsed_seconds)))
        else:
            remaining_seconds = int(self.estimated_total_seconds)
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
            'harness_rounds': self.harness_rounds,
            'harness_status': self.harness_status,
            'harness_report': self.harness_report,
            'best_png': self.best_png,
            'elapsed_seconds': elapsed_seconds,
            'estimated_total_seconds': self.estimated_total_seconds,
            'estimated_remaining_seconds': remaining_seconds,
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


def generate_poster_task(
        task_id: str,
        arxiv_id: str,
        custom_prompt: str | None = None,
        quality_threshold: int | None = None,
        max_rounds: int | None = None,
        enable_qa_eval: bool | None = None,
):
    """后台任务：生成海报"""
    task = tasks.get(task_id)
    if not task:
        return

    task.status = 'running'
    task.started_at = time.time()
    task.finished_at = None
    task.estimated_total_seconds = max(150, 135 + 45 * max(0, int(max_rounds or 1) - 1))
    task.progress = 10
    task.message = f'开始处理 arXiv: {arxiv_id}'
    task.arxiv_id = arxiv_id

    try:
        # ---- Phase 1: Parse ----
        task.message = '正在下载和解析论文...'
        task.progress = 20
        logger.info(f"Task {task_id}: Phase 1 - Parse")
        paper_doc = run_parse_paper(arxiv_id, force=True)

        # ---- Phase 2: Understand (LLM) ----
        task.message = '正在理解论文内容 (LLM)...'
        task.progress = 40
        logger.info(f"Task {task_id}: Phase 2 - Understand")
        analysis = None
        for attempt in range(1, 4):
            try:
                analysis = run_understand_paper(arxiv_id)
                break
            except Exception as e:
                logger.warning(f"Understand attempt {attempt}/3 failed: {e}")
                if attempt == 3:
                    raise
                time.sleep(0.01)
        analysis = normalize_analysis_for_poster(analysis)

        # ---- Phase 3: Plan ----
        task.message = '正在生成海报蓝图...'
        task.progress = 55
        logger.info(f"Task {task_id}: Phase 3 - Plan")
        output_dir = resolve_paper_output_dir(app.config['OUTPUT_DIR'], arxiv_id)
        blueprint = generate_blueprint(paper_doc, analysis)

        # ---- Phase 4: Render (初稿) ----
        task.message = '正在渲染初稿 HTML...'
        task.progress = 70
        logger.info(f"Task {task_id}: Phase 4 - Render Draft")
        renderer = HtmlPosterRenderer()
        draft_path = output_dir / "poster_draft.html"
        renderer.render_to_file(blueprint, paper_doc, draft_path, optimize_with_llm=False)

        # ---- Phase 5: Preliminary Supplement ----
        task.message = '正在检测板块空白区域并生成补充图...'
        task.progress = 85
        logger.info(f"Task {task_id}: Phase 5 - Visual Harness")

        def _on_harness_round(round_no: int, total: int, score: int, needs_improvement: bool, summary: str):
            """回调：更新 Preliminary Supplement 状态。"""
            task.message = '补充图处理中...'
            task.progress = min(99, 85 + int(round_no / max(total, 1) * 14))
            task.harness_rounds.append({
                'round_no': round_no,
                'quality_score': score,
                'needs_improvement': needs_improvement,
                'summary': summary,
            })

        harness_config = HarnessConfig(
            threshold=quality_threshold if quality_threshold is not None else settings.harness_threshold,
            max_rounds=max_rounds if max_rounds is not None else settings.harness_max_rounds,
            enable_qa_eval=False,
            qa_threshold=settings.harness_qa_threshold,
            zoom_crops=settings.harness_zoom_crops,
            max_crops=0,
            vision_model=settings.harness_vision_model or None,
        )

        harness_result = run_poster_harness(
            doc=paper_doc,
            analysis=analysis,
            blueprint=blueprint,
            html_path=draft_path,
            output_dir=output_dir,
            config=harness_config,
            on_round=_on_harness_round,
        )

        task.harness_status = (
            'passed' if harness_result.passed
            else ('fallback' if harness_result.fallback else 'done')
        )
        task.harness_report = harness_result.report_path
        task.best_png = harness_result.final_png
        final_html_path = Path(harness_result.final_html)
        if final_html_path.exists():
            task.html_optimized = str(final_html_path)
        else:
            # 兜底：确保存在可下载/可预览的优化版本
            task.html_optimized = str(draft_path)

        task.html_draft = str(draft_path)
        task.progress = 100
        task.status = 'complete'
        task.finished_at = time.time()
        task.message = '海报补充图生成完成' if harness_result.passed else '海报补充图生成失败'
        task.result = {
            'arxiv_id': arxiv_id,
            'draft': str(draft_path),
            'optimized': task.html_optimized,
            'output_dir': str(output_dir),
            'harness_passed': harness_result.passed,
            'harness_stop_reason': harness_result.stop_reason,
            'harness_fallback': harness_result.fallback,
            'harness_rounds': len(harness_result.rounds),
        }

        logger.info(f"Task {task_id}: Complete (harness stop_reason={harness_result.stop_reason})")

    except Exception as e:
        logger.exception(f"Task {task_id} failed")
        task.status = 'error'
        task.finished_at = time.time()
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
    data = request.get_json() or {}
    arxiv_id = data.get('arxiv_id', '').strip()
    custom_prompt = data.get('custom_prompt', '')
    quality_threshold = data.get('quality_threshold')
    max_rounds = data.get('max_rounds')

    if not arxiv_id:
        return jsonify({'error': '请提供 arXiv ID'}), 400

    # 参数校验
    if quality_threshold is not None:
        try:
            quality_threshold = int(quality_threshold)
        except (TypeError, ValueError):
            return jsonify({'error': 'quality_threshold 必须是整数'}), 400
    if max_rounds is not None:
        try:
            max_rounds = int(max_rounds)
        except (TypeError, ValueError):
            return jsonify({'error': 'max_rounds 必须是整数'}), 400

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
        args=(
            task_id,
            arxiv_id,
            custom_prompt if custom_prompt else None,
            quality_threshold,
            max_rounds,
            False,
        ),
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
    elif version in ('optimized', 'best'):
        html_path = task.html_optimized
    else:
        return jsonify({'error': '无效版本'}), 400

    if not html_path or not Path(html_path).exists():
        return jsonify({'error': '文件不存在'}), 404

    return send_file(html_path, mimetype='text/html')


@app.route('/api/harness/<task_id>', methods=['GET'])
def get_harness(task_id):
    """获取 harness 审查报告与轮次历史"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404

    report = None
    if task.harness_report and Path(task.harness_report).exists():
        try:
            report = json.loads(Path(task.harness_report).read_text(encoding='utf-8'))
        except Exception:
            report = None

    return jsonify({
        'task_status': task.status,
        'harness_status': task.harness_status,
        'harness_rounds': task.harness_rounds,
        'harness_report': report,
        'best_png': task.best_png,
    })


@app.route('/api/round_image/<task_id>/<int:round_no>', methods=['GET'])
def round_image(task_id, round_no):
    """获取 Preliminary Supplement 处理后的海报快照 PNG"""
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404

    if task.status != 'complete':
        return jsonify({'error': '任务未完成'}), 400

    output_dir = Path(task.result['output_dir'])
    harness_dir = (output_dir / 'harness').resolve()
    png_path = (harness_dir / f'round_{round_no}' / 'poster.png').resolve()
    if not png_path.is_relative_to(harness_dir) or not png_path.exists():
        return jsonify({'error': '图像不存在'}), 404

    return send_file(png_path, mimetype='image/png')


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
            time.sleep(0.01)
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
