# 最终 HTML 生成流程概览

这个项目里，最终 HTML 海报的生成主线可以概括为：**解析论文 -> 理解内容 -> 生成海报蓝图 -> 渲染初稿 HTML -> 视觉审查/LLM 优化 -> 输出最终 HTML**。

## 一、总体流程

1. **论文解析**
   把 arXiv 论文下载并解析成结构化数据，例如标题、章节、公式、图表、表格和原始 Markdown。这里不是单纯“抓文本”，而是要把论文拆成后续能被布局和摘要直接消费的对象：章节树、扁平正文、图像索引、表格索引、公式索引都会尽量保留下来。这样后面的阶段才能知道哪些内容该进海报、哪些内容只是背景噪声。

2. **论文理解**
   用 LLM 对论文内容做抽取与归纳，形成可用于海报设计的分析结果，例如问题、贡献、方法、实验结果。这个阶段的重点不是“总结得长”，而是“总结得适合做海报”：问题陈述要短、贡献要像短句、方法要是高层描述、实验要能落到少量关键结果上。它的输出会直接成为后面蓝图里的正文素材，所以这里会有比较强的长度和结构约束。

3. **海报蓝图生成**
   根据论文结构和分析结果，生成 `PosterBlueprint`，决定哪些内容放进哪些版块、图表怎么摆、整张海报的布局和配色是什么。这个阶段本质上是在做“内容到版面”的映射：把论文里的长叙述压缩成几个固定故事块，例如 motivation、method overview、main method、experiments、contributions 等；同时给每一块分配行列位置、跨列宽度、图像占位和颜色方案。它更像静态排版规划，而不是让渲染器自己去猜。

4. **HTML 初稿渲染**
   把蓝图交给 HTML 渲染器，生成第一版海报 HTML，并把需要的图像资源整理到输出目录。这里的初版 HTML 不是自由流式排版，而是按 `PosterBlueprint` 里的行列、跨列跨度和宽高占位来落位的**硬性布局**，所以第一版通常会比较规整、稳定。

   同时，论文里的图像资源在进入最终输出目录前会先做浏览器可用化处理：PDF 图会先栅格化，图片会统一复制和重命名，位图还会检测并裁掉四周无效的近白空白区域，尽量保留图像有效内容，避免空白边距把版面撑松。

5. **视觉审查与优化**
   通过 harness 对渲染后的 HTML 做图像级检查，必要时调用 LLM 做内容或版式修正，再输出最终版本。这里不是只看“HTML 是否能打开”，而是看截图里是否真的像一张能读的海报：有没有溢出、重叠、留白失衡、图太小、文字太密、关键结果看不见。若发现问题，会反向推动内容压缩、重排或替换图像。

6. **结果落盘与预览**
   最终 HTML 会保存在对应论文的输出目录里，同时接口层提供预览和下载。除了最终成品，流程中间产物也会保留下来，方便回溯每一步是怎么从论文原文走到可视化海报的。

## 二、对应的核心代码位置

### 1. Web/API 入口与总串联

- [`app.py`](./app.py)
  - 前端提交生成任务的入口。
  - `generate_poster_task()` 里把整条链路串起来。
  - 关键步骤：`run_parse_paper()` -> `run_understand_paper()` -> `generate_blueprint()` -> `HtmlPosterRenderer().render_to_file()` -> `run_poster_harness()`。
  - 最终 HTML 在这里通常会落到：`poster_draft.html`、`poster_optimized.html`、`poster_final.html`。
  - 这里更像任务编排器：负责按顺序跑各阶段、接住异常、把阶段产物写进输出目录，而不是做具体内容生成。

### 2. 批处理/命令行主入口

- [`src/main.py`](./src/main.py)
  - 命令行主入口。
  - `pipeline` / `pipeline-v2` / `render` / `optimize-html` 等子命令都在这里分发。
  - 如果想从命令行跑完整流程，这里是最直接的入口。
  - 它对应的是工程上的“跑一遍管线”，适合批处理、调试和复现，而不是面向交互任务的细粒度控制。

### 3. 传统流水线主实现

- [`src/agents/pipeline_agent.py`](./src/agents/pipeline_agent.py)
  - 负责经典五阶段流水线。
  - 重点是 Phase 4：`HtmlPosterRenderer().render_to_file(..., "poster.html")`。
  - 这里会把最终 HTML 写成 `poster.html`。
  - 这个实现把“解析 -> 理解 -> 规划 -> 渲染 -> 校验”串成一个比较明确的阶段式流程，便于逐步检查每一步输出是否合理。

### 4. v2 流水线与布局树

- [`src/agents/poster_v2.py`](./src/agents/poster_v2.py)
  - 先构建 `LayoutTree`，再转成 `PosterBlueprint`。
  - `render_layout_tree()` 里调用 `HtmlPosterRenderer` 输出 HTML。
  - `run_poster_v2()` 会额外写出 `layout_tree.json` 和 `blueprint_v2.json`。
  - 这条线比传统流程更强调布局结构先行：先有树，再有蓝图，再到 HTML，便于在渲染前就检查层级、阅读顺序和空间分配。

### 5. HTML 渲染器

- [`src/renderers/html_renderer.py`](./src/renderers/html_renderer.py)
  - 真正把 `PosterBlueprint` 变成 HTML 的地方。
  - 负责：
    - 清洗内容
    - Markdown 转 HTML
    - LaTeX 保护/还原
    - 图像资源准备
    - 套用 `poster.html.j2` 模板
    - 可选的 LLM HTML 优化
  - 这是“最终 HTML 长什么样”的核心实现。
  - 这里本质上是蓝图驱动布局：`PosterBlueprint.sections` 里的 `row`、`column`、`col_span`、`row_span` 决定了区块的位置和占位方式，不是让浏览器自己去自由流式排版。
  - 图像进入页面前会走 `copy_or_rasterize_asset()` 之类的整理步骤，近白边缘裁剪就在这条链路里完成。
  - 这一层更像“把设计稿落成网页”，重点是把文本、公式和图片组织成稳定、可截图的版面。

### 6. 单次/批量 HTML 优化

- [`src/agents/html_optimizer.py`](./src/agents/html_optimizer.py)
  - 独立的 HTML 优化工具。
  - 输入已有 HTML 和提示词文件，输出优化后的完整 HTML。
  - 会校验 LLM 返回是否还是完整 HTML，并修正图像路径。
  - 它更像“在已有成品上做局部修饰”，用来修内容措辞、结构细节或轻微的样式问题，而不是重新发明版式。

### 7. 图像裁剪与资源整理

- [`src/utils/figure_assets.py`](./src/utils/figure_assets.py)
  - 负责把论文里的 figure 资产整理成浏览器能直接加载的本地文件。
  - 对图片会先做近白边界裁剪，再输出到海报自己的 `figures/` 目录。
  - 这一步的目标不是改图意，而是去掉论文原图里多余的空白，让 HTML 版面更紧凑。
  - 这里也承担了“统一资源格式”的工作，所以后面无论是 HTML 预览还是 harness 截图，看到的都是同一套本地化资源。

### 8. 视觉审查与最终版生成

- [`src/agents/poster_harness.py`](./src/agents/poster_harness.py)
  - 对渲染后的 HTML 做视觉审查。
  - 根据审查结果决定是否继续迭代。
  - 最终会产出 harness 的最终 HTML、PNG 和审查报告。
  - 在 `app.py` 中，它决定最终对外展示的 HTML 是哪一版。
  - 这里也会关注无效空白区域，尤其是 section 内部大块留白、Core Results 区域异常空旷这类问题；如果空白比例过高，通常会触发重排、压缩文本或补图。
  - 这一步的作用是把“能打开”提升到“能读、像样、视觉上成立”。

## 三、最终 HTML 通常输出到哪里

- `output/<arxiv_id>/poster.html`：传统流水线和 v2 流水线常见的渲染输出。
- `output/<arxiv_id>/poster_draft.html`：Web 任务里的初稿 HTML。
- `output/<arxiv_id>/poster_optimized.html`：LLM 优化或回退优化后的 HTML。
- `output/<arxiv_id>/poster_final.html`：视觉审查 harness 通过后生成的最终版。

## 四、最该先看的几个文件

如果你只想快速弄懂“最终 HTML 是怎么来的”，建议按这个顺序看：

1. [`app.py`](./app.py)
2. [`src/agents/poster_harness.py`](./src/agents/poster_harness.py)
3. [`src/renderers/html_renderer.py`](./src/renderers/html_renderer.py)
4. [`src/agents/pipeline_agent.py`](./src/agents/pipeline_agent.py)
5. [`src/agents/poster_v2.py`](./src/agents/poster_v2.py)
6. [`src/agents/html_optimizer.py`](./src/agents/html_optimizer.py)

## 五、一句话总结

这个项目的最终 HTML 不是单点生成的，而是由“论文解析 + LLM 理解 + 海报规划 + HTML 渲染 + 视觉审查优化”共同完成；其中真正把蓝图变成 HTML 的核心代码在 `src/renderers/html_renderer.py`，而最终落盘和对外输出的主控制点在 `app.py` 和 `src/agents/poster_harness.py`。

更具体一点说，第一版 HTML 本质上是按蓝图固定网格排出来的硬布局，图片资源在进入页面之前也会先被清理空白边缘，后面再通过视觉审查去修正那些“看着像排版问题、其实是内容密度或无效留白问题”的细节。
