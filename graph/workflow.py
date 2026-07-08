# -*- coding: utf-8 -*-
"""LangGraph 工作流编排 —— 4 节点 + 1 Conditional Edge 的研报生成管线"""
import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_deepseek import ChatDeepSeek

# 导入三个 Agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, MAX_RETRY
from agents.research_agent import ResearchAgent
from agents.analyst_agent import AnalystAgent
from agents.writer_agent import WriterAgent

# ============================================================
# State 定义
# ============================================================
class ReportState(TypedDict):
    query: str                      # 用户原始问题
    sub_questions: List[str]        # 拆解后的子问题
    research_results: List[dict]    # 搜索结果
    analysis: dict                  # 分析师结论
    retry_count: int                # 当前重试次数
    final_report: str               # 最终报告


# ============================================================
# 子问题拆解 Prompt
# ============================================================
DECOMPOSE_PROMPT = """你是一个专业的研究助理。请将用户的研究主题拆解为 3~5 个具体的子问题，以便逐一搜索获取信息。

用户主题：{query}

要求：
- 每个子问题应该是一个可独立搜索的明确问题
- 覆盖市场规模、竞争格局、技术趋势、政策环境等关键维度（根据主题灵活选择）
- 以严格的 JSON 数组格式返回，只返回 JSON，不要包含其他内容

示例格式：["子问题1", "子问题2", "子问题3"]
"""


# ============================================================
# LangGraph 节点函数
# ============================================================

def _get_llm():
    """获取 LLM 实例（每个节点独立创建，避免状态污染）"""
    return ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_BASE_URL,
        temperature=0.0,
        request_timeout=120,
    )


def node_query_analysis(state: ReportState) -> ReportState:
    """Node 1: 将用户问题拆解为 3~5 个子问题"""
    print("\n[Node 1/4] 需求拆解 — 正在分析用户问题...")
    llm = _get_llm()
    prompt = DECOMPOSE_PROMPT.format(query=state["query"])

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # 清洗 JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        sub_questions = json.loads(raw)
    except json.JSONDecodeError:
        # 兜底：按行解析
        sub_questions = [q.strip().lstrip("1234567890.。， ") for q in raw.split("\n") if q.strip()]
        sub_questions = [q for q in sub_questions if len(q) > 5]

    state["sub_questions"] = sub_questions
    print(f"    拆解完成，共 {len(sub_questions)} 个子问题:")
    for i, q in enumerate(sub_questions, 1):
        print(f"      {i}. {q}")
    return state


def node_research(state: ReportState) -> ReportState:
    """Node 2: ResearchAgent 逐个搜索子问题"""
    print(f"\n[Node 2/4] 信息采集 — 正在搜索 {len(state['sub_questions'])} 个子问题...")
    agent = ResearchAgent()

    # 如果已有部分搜索结果（回退场景），保留旧的，追加新的
    existing = state.get("research_results", [])
    new_results = agent.search(state["sub_questions"])
    state["research_results"] = existing + new_results

    total_hits = sum(len(r.get("results", [])) for r in state["research_results"])
    print(f"    采集完成，共 {total_hits} 条搜索结果")
    return state


def node_analysis(state: ReportState) -> ReportState:
    """Node 3: AnalystAgent 验证信息质量"""
    print(f"\n[Node 3/4] 质量验证 — 正在分析信息充分性...")
    agent = AnalystAgent()
    analysis = agent.analyze(state["query"], state["research_results"])
    state["analysis"] = analysis
    state["retry_count"] = state.get("retry_count", 0)

    sufficiency = "充足" if analysis["is_sufficient"] else "不足"
    print(f"    分析完成: 信息{sufficiency} | 置信度 {analysis['confidence']} | 重试 {state['retry_count']}/{MAX_RETRY}")

    if not analysis["is_sufficient"] and analysis.get("suggested_queries"):
        print(f"    缺口: {analysis['gaps'][:100]}...")
        print(f"    建议补充: {analysis['suggested_queries'][:3]}")
        # 用建议的子问题替换当前子问题列表，供回退后的 Research 节点使用
        state["sub_questions"] = analysis["suggested_queries"][:3]
    return state


def node_writing(state: ReportState) -> ReportState:
    """Node 4: WriterAgent 生成最终报告"""
    print(f"\n[Node 4/4] 报告生成 — 正在撰写研究报告...")
    agent = WriterAgent()
    report = agent.write(
        query=state["query"],
        research_results=state["research_results"],
        analysis=state.get("analysis", {}),
    )
    state["final_report"] = report
    print(f"    报告生成完成，共 {len(report)} 字符")
    return state


# ============================================================
# Conditional Edge: 决定是回退还是继续
# ============================================================
def should_retry(state: ReportState) -> str:
    """判断是否需要补充搜索"""
    analysis = state.get("analysis", {})
    retry_count = state.get("retry_count", 0)

    if not analysis.get("is_sufficient") and retry_count < MAX_RETRY:
        state["retry_count"] = retry_count + 1
        print(f"\n    >>> 信息不足，启动第 {state['retry_count']} 次补充搜索 <<<")
        return "research"
    else:
        if not analysis.get("is_sufficient"):
            print(f"\n    >>> 已达最大重试次数 ({MAX_RETRY})，直接生成报告 <<<")
        print(f"\n    >>> 信息充足，进入报告生成阶段 <<<")
        return "writing"


# ============================================================
# 构建 LangGraph
# ============================================================
def build_graph() -> StateGraph:
    """构建并返回编译好的 LangGraph 工作流"""
    workflow = StateGraph(ReportState)

    # 添加节点
    workflow.add_node("query_analysis", node_query_analysis)
    workflow.add_node("research", node_research)
    workflow.add_node("analysis", node_analysis)
    workflow.add_node("writing", node_writing)

    # 设置入口
    workflow.set_entry_point("query_analysis")

    # 边
    workflow.add_edge("query_analysis", "research")
    workflow.add_edge("research", "analysis")

    # Conditional Edge: analysis → research（回退）或 writing（继续）
    workflow.add_conditional_edges(
        "analysis",
        should_retry,
        {
            "research": "research",
            "writing": "writing",
        }
    )

    workflow.add_edge("writing", END)

    return workflow.compile()


# ============================================================
# 对外入口
# ============================================================
def run_report(query: str) -> dict:
    """执行完整的研报生成流程

    返回: {"report": str, "state": ReportState}
    """
    graph = build_graph()
    initial_state: ReportState = {
        "query": query,
        "sub_questions": [],
        "research_results": [],
        "analysis": {},
        "retry_count": 0,
        "final_report": "",
    }

    print("=" * 60)
    print(f"  智能研报系统 — {query}")
    print("=" * 60)

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("  报告正文")
    print("=" * 60)
    print(final_state["final_report"])

    return {"report": final_state["final_report"], "state": final_state}


# ============================================================
# 验证脚本
# ============================================================
if __name__ == "__main__":
    test_query = "AI教育行业2025年的发展趋势"
    result = run_report(test_query)
    print("\n[Done] 工作流执行完毕")
