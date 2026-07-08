# -*- coding: utf-8 -*-
"""WriterAgent —— 纯 LLM 推理，将分析后的信息整合为结构化 Markdown 报告"""
import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_deepseek import ChatDeepSeek

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

WRITER_PROMPT = """你是一个专业的行业研究报告撰写员。请根据以下信息，生成一份结构化的 Markdown 研究报告。

## 用户研究主题
{query}

## 分析师摘要
{analysis_summary}
（信息充分性: {is_sufficient}，置信度: {confidence}）

## 搜索到的信息来源
{search_results}

## 要求
1. 报告结构：
   - 报告标题（## 标题）
   - 核心结论（3-5 条要点）
   - 分章节论述（按主题组织，每章包含：发现、数据支撑、来源引用）
   - 信息局限性声明（如果分析师认为信息不够充分，需明确说明哪些结论可能存在不确定性）

2. 格式规范：
   - 使用 Markdown 格式
   - 引用数据时用 [来源N] 标注，并在文末附上参考链接列表
   - 语言专业、客观，不编造搜索结果中没有的数据
   - 如果信息不足以支撑某个结论，请明确说明

请直接输出 Markdown 报告，不要输出其他内容。"""


class WriterAgent:
    """报告撰写 Agent"""

    def __init__(self):
        self.llm = ChatDeepSeek(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            api_base=DEEPSEEK_BASE_URL,
            temperature=0.3,  # 稍微提高温度让报告可读性更好
            request_timeout=120,
        )

    def write(self, query: str, research_results: list[dict],
              analysis: dict) -> str:
        """
        生成结构化 Markdown 报告

        :param query: 用户原始问题
        :param research_results: ResearchAgent 搜索结果
        :param analysis: AnalystAgent 分析结果
        :return: Markdown 格式的完整报告文本
        """
        # 格式化搜索结果，加上来源编号
        results_text = self._format_results_with_sources(research_results)

        prompt = WRITER_PROMPT.format(
            query=query,
            analysis_summary=analysis.get("summary", "无"),
            is_sufficient="是" if analysis.get("is_sufficient") else "否",
            confidence=analysis.get("confidence", "medium"),
            search_results=results_text,
        )

        print("    [撰写] 正在生成报告...")
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"    [!] 报告生成失败: {e}")
            return f"# 报告生成失败\n\n错误信息: {e}"

    @staticmethod
    def _format_results_with_sources(results: list[dict]) -> str:
        """格式化搜索结果，带来源编号"""
        lines = []
        idx = 1
        for group in results:
            for item in group.get("results", []):
                lines.append(f"[来源{idx}] **{item.get('title', '(无标题)')}**")
                lines.append(f"    URL: {item.get('url', '')}")
                lines.append(f"    内容: {item.get('content', '')[:400]}")
                lines.append("")
                idx += 1
        return "\n".join(lines)


# --- 验证脚本 ---
if __name__ == "__main__":
    # Mock 数据：模拟前两步的输出
    mock_results = [
        {
            "question": "AI教育市场规模",
            "results": [
                {
                    "title": "2025年AI教育市场突破2000亿",
                    "content": "据艾瑞咨询报告，2025年中国AI教育市场规模预计突破2000亿元，年增长率保持在35%以上。政策层面，教育部出台多项措施鼓励AI与教育深度融合。",
                    "url": "https://example.com/ai-edu-market"
                },
                {
                    "title": "全球AI教育趋势分析",
                    "content": "北美和亚太地区是AI教育增长最快的市场。个性化学习、智能辅导和自动化评估是三大核心应用场景。",
                    "url": "https://example.com/global-trend"
                }
            ]
        }
    ]

    mock_analysis = {
        "is_sufficient": False,
        "summary": "获取到市场规模数据（2000亿）和主要应用场景，但缺乏行业竞争格局、头部企业营收等关键信息。",
        "gaps": "缺少竞争格局、头部企业数据、技术路线对比",
        "confidence": "medium",
    }

    writer = WriterAgent()
    report = writer.write("2025年AI教育行业发展趋势", mock_results, mock_analysis)

    print("\n" + "=" * 60)
    print("WriterAgent 验证 — 生成报告 (前 500 字)")
    print("=" * 60)
    print(report[:500])
    print(f"\n... (总字数: {len(report)})")
