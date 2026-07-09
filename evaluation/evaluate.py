# -*- coding: utf-8 -*-
"""
消融实验评测脚本
对比方案 A（无 Analyst 朴素版）vs 方案 B（完整版，含质量门控+回溯）
验证多 Agent 架构中 AnalystAgent 的有效性
"""
import sys
import os
import json
import time
from datetime import datetime
from typing import TypedDict, List

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, MAX_RETRY, TAVILY_API_KEY
from langgraph.graph import StateGraph, END
from langchain_deepseek import ChatDeepSeek

from graph.workflow import (
    ReportState, node_query_analysis, node_research,
    node_analysis, node_writing, should_retry,
)
from agents.research_agent import ResearchAgent
from agents.writer_agent import WriterAgent


# ============================================================
# 方案 A：朴素版工作流（无 Analyst + 无回溯）
# ============================================================
def build_graph_variant_a() -> StateGraph:
    """3 节点直线管线：拆解 → 搜索 → 撰写（跳过质检和回溯）"""
    workflow = StateGraph(ReportState)

    workflow.add_node("query_analysis", node_query_analysis)
    workflow.add_node("research", node_research)
    workflow.add_node("writing", node_writing)

    workflow.set_entry_point("query_analysis")
    workflow.add_edge("query_analysis", "research")
    workflow.add_edge("research", "writing")
    workflow.add_edge("writing", END)

    return workflow.compile()


# ============================================================
# LLM Judge：对比打分
# ============================================================
JUDGE_PROMPT = """你是一个专业的研究报告质量评审专家。请对比以下两份由不同方案生成的行业研究报告。

## 用户研究主题
{query}

## 报告 A（朴素方案：搜一次直接写，无质检环节）
{report_a}

## 报告 B（完整方案：子问题拆解 + 信息充分性质检 + 不足时自动回溯补充搜索）
{report_b}

请从以下 4 个维度分别对两份报告打分（1-5 分，5 分最优），并给出简要理由。
每份报告独立评分，不要因为对比而刻意拉大差距。

评分维度：
1. **信息完整度**：是否覆盖了主题的核心维度（市场/技术/竞争/政策等），关键信息是否有遗漏
2. **数据引用密度**：报告中引用的具体数据、案例、来源数量是否充分，是否有数据支撑而非空泛描述
3. **结构逻辑性**：报告章法是否清晰（结论 → 分章节论述 → 局限性声明），逻辑是否连贯
4. **事实准确性**：结论是否有来源支撑，是否存在无依据的推断或编造

以严格的 JSON 格式返回（不要包含任何其他文字）：
{{
    "report_a": {{
        "信息完整度": 分数,
        "数据引用密度": 分数,
        "结构逻辑性": 分数,
        "事实准确性": 分数,
        "总评": "对报告 A 的简要评价（100字以内）"
    }},
    "report_b": {{
        "信息完整度": 分数,
        "数据引用密度": 分数,
        "结构逻辑性": 分数,
        "事实准确性": 分数,
        "总评": "对报告 B 的简要评价（100字以内）"
    }},
    "对比总结": "两份方案的差异总结（100字以内，突出方案 B 相比方案 A 的核心优势或劣势）"
}}"""


def _get_judge():
    """获取 LLM Judge 实例"""
    return ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_BASE_URL,
        temperature=0.0,
        request_timeout=120,
    )


def judge_reports(query: str, report_a: str, report_b: str) -> dict:
    """用 LLM Judge 对比两份报告并打分"""
    llm = _get_judge()
    prompt = JUDGE_PROMPT.format(query=query, report_a=report_a, report_b=report_b)

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        print(f"    [!] Judge 评分失败: {e}")
        return {"error": str(e), "raw": raw if 'raw' in dir() else ""}


# ============================================================
# 主评测流程
# ============================================================
TEST_TOPICS = [
    "2025年中国新能源商用车市场发展现状与趋势",
    "AIGC工具在短视频内容创作中的应用与商业化前景",
    "全球半导体产业链重构对中国芯片企业的机遇与挑战",
]


def run_variant(graph, query: str, label: str) -> dict:
    """运行方案并返回报告 + 耗时 + 中间指标"""
    initial_state: ReportState = {
        "query": query,
        "sub_questions": [],
        "research_results": [],
        "analysis": {},
        "retry_count": 0,
        "final_report": "",
    }

    print(f"\n{'─' * 50}")
    print(f"  [{label}] {query}")
    print(f"{'─' * 50}")

    start = time.time()
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - start

    total_sources = sum(len(r.get("results", [])) for r in final_state.get("research_results", []))

    result = {
        "query": query,
        "label": label,
        "report": final_state.get("final_report", ""),
        "elapsed_seconds": round(elapsed, 1),
        "sub_questions_count": len(final_state.get("sub_questions", [])),
        "total_sources": total_sources,
        "retry_count": final_state.get("retry_count", 0),
        "analysis": final_state.get("analysis", {}),
    }

    print(f"    耗时: {elapsed:.1f}s | 子问题: {result['sub_questions_count']} | 来源: {total_sources} | 重试: {result['retry_count']}")
    return result


def main():
    print("=" * 60)
    print("  消融实验评测 — AnalystAgent 有效性验证")
    print("  方案 A（无质检）vs 方案 B（完整版，含质检+回溯）")
    print("=" * 60)

    # 构建两个方案的工作流
    graph_a = build_graph_variant_a()
    from graph.workflow import build_graph
    graph_b = build_graph()

    all_results = []
    score_table = []

    for i, topic in enumerate(TEST_TOPICS, 1):
        print(f"\n{'=' * 60}")
        print(f"  主题 {i}/{len(TEST_TOPICS)}: {topic}")
        print(f"{'=' * 60}")

        # 跑方案 A
        result_a = run_variant(graph_a, topic, "方案A（朴素版）")
        # 跑方案 B
        result_b = run_variant(graph_b, topic, "方案B（完整版）")

        # LLM Judge 评分
        print(f"\n  [Judge] 正在对比评分...")
        scores = judge_reports(topic, result_a["report"][:4000], result_b["report"][:4000])

        all_results.append({
            "topic": topic,
            "variant_a": result_a,
            "variant_b": result_b,
            "judge_scores": scores,
        })

        if "error" not in scores:
            sa = scores.get("report_a", {})
            sb = scores.get("report_b", {})
            score_table.append({
                "主题": topic,
                "A-信息完整度": sa.get("信息完整度", "-"),
                "B-信息完整度": sb.get("信息完整度", "-"),
                "A-数据引用": sa.get("数据引用密度", "-"),
                "B-数据引用": sb.get("数据引用密度", "-"),
                "A-结构逻辑": sa.get("结构逻辑性", "-"),
                "B-结构逻辑": sb.get("结构逻辑性", "-"),
                "A-事实准确": sa.get("事实准确性", "-"),
                "B-事实准确": sb.get("事实准确性", "-"),
                "A耗时(s)": result_a["elapsed_seconds"],
                "B耗时(s)": result_b["elapsed_seconds"],
                "对比总结": scores.get("对比总结", "-"),
            })

    # ---- 汇总输出 ----
    print("\n\n" + "=" * 60)
    print("  评测汇总")
    print("=" * 60)

    # 计算各维度均分
    dims = ["信息完整度", "数据引用密度", "结构逻辑性", "事实准确性"]
    a_avgs = {}
    b_avgs = {}

    print(f"\n{'维度':<14} {'方案A均分':>10} {'方案B均分':>10} {'提升':>10}")
    print("-" * 46)
    for dim in dims:
        a_vals = [t.get("report_a", {}).get(dim, 0) for t in
                  [r.get("judge_scores", {}) for r in all_results]
                  if isinstance(r.get("judge_scores", {}).get("report_a", {}).get(dim), (int, float))]
        b_vals = [t.get("report_b", {}).get(dim, 0) for t in
                  [r.get("judge_scores", {}) for r in all_results]
                  if isinstance(r.get("judge_scores", {}).get("report_b", {}).get(dim), (int, float))]

        # 重新算：遍历 all_results
        a_vals = []
        b_vals = []
        for r in all_results:
            s = r.get("judge_scores", {})
            if "error" not in s:
                if isinstance(s.get("report_a", {}).get(dim), (int, float)):
                    a_vals.append(s["report_a"][dim])
                if isinstance(s.get("report_b", {}).get(dim), (int, float)):
                    b_vals.append(s["report_b"][dim])

        avg_a = sum(a_vals) / len(a_vals) if a_vals else 0
        avg_b = sum(b_vals) / len(b_vals) if b_vals else 0
        improvement = ((avg_b - avg_a) / avg_a * 100) if avg_a > 0 else 0

        a_avgs[dim] = avg_a
        b_avgs[dim] = avg_b

        print(f"{dim:<14} {avg_a:>10.2f} {avg_b:>10.2f} {improvement:>+9.1f}%")

    # 总评
    all_a = [v for v in a_avgs.values() if v > 0]
    all_b = [v for v in b_avgs.values() if v > 0]
    avg_a_total = sum(all_a) / len(all_a) if all_a else 0
    avg_b_total = sum(all_b) / len(all_b) if all_b else 0
    total_improvement = ((avg_b_total - avg_a_total) / avg_a_total * 100) if avg_a_total > 0 else 0

    print("-" * 46)
    print(f"{'综合均分':<14} {avg_a_total:>10.2f} {avg_b_total:>10.2f} {total_improvement:>+9.1f}%")

    # 保存结果
    os.makedirs("evaluation/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存 JSON
    json_path = f"evaluation/results/results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "test_topics": TEST_TOPICS,
            "summary": {
                "variant_a_avg_score": round(avg_a_total, 2),
                "variant_b_avg_score": round(avg_b_total, 2),
                "total_improvement_pct": round(total_improvement, 1),
                "dimension_scores": {
                    dim: {"a": round(a_avgs[dim], 2), "b": round(b_avgs[dim], 2)}
                    for dim in dims
                }
            },
            "details": score_table,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {json_path}")

    # 保存可读 Markdown
    md_path = f"evaluation/results/results_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 消融实验评测报告\n\n")
        f.write(f"**评测时间**: {timestamp}\n\n")
        f.write(f"**测试主题数**: {len(TEST_TOPICS)}\n\n")
        f.write(f"## 综合评分\n\n")
        f.write(f"| 维度 | 方案A均分 | 方案B均分 | 提升 |\n")
        f.write(f"|------|----------|----------|------|\n")
        for dim in dims:
            f.write(f"| {dim} | {a_avgs[dim]:.2f} | {b_avgs[dim]:.2f} | {((b_avgs[dim]-a_avgs[dim])/a_avgs[dim]*100):+.1f}% |\n")
        f.write(f"| **综合均分** | **{avg_a_total:.2f}** | **{avg_b_total:.2f}** | **{total_improvement:+.1f}%** |\n\n")
        f.write(f"## 各主题详情\n\n")
        for item in score_table:
            f.write(f"### {item['主题']}\n\n")
            f.write(f"| 维度 | 方案A | 方案B |\n")
            f.write(f"|------|-------|-------|\n")
            for dim in dims:
                key_map = {"信息完整度": "A-信息完整度", "数据引用密度": "A-数据引用", "结构逻辑性": "A-结构逻辑", "事实准确性": "A-事实准确"}
                f.write(f"| {dim} | {item.get(key_map[dim], '-')} | {item.get(key_map[dim].replace('A-','B-'), '-')} |\n")
            f.write(f"\n> {item.get('对比总结', '')}\n\n")
    print(f"Markdown 报告已保存至: {md_path}")


if __name__ == "__main__":
    if not DEEPSEEK_API_KEY or not TAVILY_API_KEY:
        print("[!] 请先配置 .env 文件（DEEPSEEK_API_KEY + TAVILY_API_KEY）")
        sys.exit(1)
    main()
