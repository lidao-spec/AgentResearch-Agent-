# Multi-Agent 智能研报系统

基于 **LangGraph** 编排 3 个 AI Agent（研究员、分析师、撰写员）协作完成行业研究报告的自动生成，支持联网搜索、信息验证与回退补充搜索。

## 架构

```
用户输入研究主题
    │
    ▼
┌──────────────────────────────────────────┐
│ Node 1: QueryAnalysis（需求拆解）          │
│   将用户问题拆解为 3~5 个子问题             │
│   LLM: DeepSeek                          │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ Node 2: Research（信息采集）               │
│   ResearchAgent 调用 Tavily 搜索子问题      │
│   每个子问题返回 top 3 条结果               │
│   工具: Tavily Search API                 │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ Node 3: Analysis（质量验证）               │
│   AnalystAgent 评估信息充分性与可信度       │
│   LLM: DeepSeek（纯推理，无工具调用）        │
└──────┬───────────────┬───────────────────┘
       │               │
  信息充足           信息不足 && 重试 < 2
       │               │
       ▼               ▼
┌──────────────┐  ┌─────────────────────────┐
│ Node 4:      │  │ 回退到 Node 2            │
│ Writing      │  │ 用建议子问题重新搜索       │
│ (报告生成)    │  └─────────────────────────┘
│ LLM: DeepSeek│
└──────┬───────┘
       │
       ▼
  输出 Markdown 报告 + 保存 .md 文件
```

### 3 个 Agent 职责

| Agent | 角色 | 使用的工具 |
|-------|------|----------|
| **ResearchAgent（研究员）** | 联网搜索采集信息 | Tavily Search API |
| **AnalystAgent（分析师）** | 交叉验证信息质量，判断是否充分 | 仅 LLM 推理 |
| **WriterAgent（撰写员）** | 整合信息生成结构化 Markdown 报告 | 仅 LLM 推理 |

### LangGraph 工作流

- **4 个节点**：QueryAnalysis → Research → Analysis → Writing
- **1 个条件边**：Analysis → 信息不足则回退 Research（最多 2 次）
- **State 管理**：TypedDict 定义，全程持久化，支持断点恢复

## 技术栈

| 组件 | 技术 |
|------|------|
| 编排框架 | LangGraph 1.2.4 |
| 大模型 | DeepSeek (deepseek-chat) |
| 搜索 API | Tavily Search |
| 语言 | Python 3.13 |
| 结构化输出 | JSON / Pydantic |

## 快速开始

### 1. 安装依赖

```bash
pip install langgraph langchain-deepseek langchain-tavily langchain-core python-dotenv
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek 和 Tavily API Key
```

| Key | 获取地址 |
|-----|---------|
| DEEPSEEK_API_KEY | https://platform.deepseek.com |
| TAVILY_API_KEY | https://tavily.com（免费 1000次/月） |

### 3. 运行

```bash
python main.py
```

示例：

```
请输入您想研究的主题:
> AI教育行业2025年发展趋势

[Node 1/4] 需求拆解 — 正在分析用户问题...
    拆解完成，共 5 个子问题
[Node 2/4] 信息采集 — 正在搜索 5 个子问题...
[Node 3/4] 质量验证 — 正在分析信息充分性...
[Node 4/4] 报告生成 — 正在撰写研究报告...

============================================================
  报告正文
============================================================
## 报告标题
...
## 核心结论
1. ...
============================================================
  报告已保存至: reports/report_20260708_230000.md
  总耗时: 57.3s
```

## 项目结构

```
multi_agent_report/
├── main.py                  # CLI 入口
├── config.py                # 配置（从 .env 读取）
├── .env.example             # 配置模板
├── .gitignore
├── requirements.txt
├── README.md
├── agents/
│   ├── __init__.py
│   ├── research_agent.py    # ResearchAgent — Tavily 搜索
│   ├── analyst_agent.py     # AnalystAgent — 信息验证
│   └── writer_agent.py      # WriterAgent — 报告生成
├── graph/
│   ├── __init__.py
│   └── workflow.py          # LangGraph StateGraph 定义
└── reports/                 # 生成的报告输出目录
```

## 报告结构

每份自动生成的报告包含：
- 报告标题
- 核心结论（3-5 条要点）
- 分章节论述（含数据支撑和来源引用）
- 信息局限性声明
- 参考链接列表
