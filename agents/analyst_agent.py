# -*- coding: utf-8 -*-
"""AnalystAgent —— 纯 LLM 推理，验证搜索结果质量，判断信息是否充分"""
import sys
import os
import json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_deepseek import ChatDeepSeek

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

ANALYST_PROMPT = """你是一个严谨的信息分析师。请根据以下搜索结果，判断信息是否足以回答用户的原始问题。

用户原始问题：
{query}

搜索结果：
{search_results}

请完成以下分析，并以严格的 JSON 格式返回（不要包含任何其他文字）：

{{
    "is_sufficient": true/false,
    "summary": "搜索结果核心信息摘要（200字以内）",
    "gaps": "信息缺口描述（如不足则详细说明缺少什么，如足够则写'无'）",
    "suggested_queries": ["补充搜索建议1", "补充搜索建议2"],
    "confidence": "high/medium/low"
}}
"""


class AnalystAgent:
    """搜索结果质量分析 Agent"""

    def __init__(self):
        self.llm = ChatDeepSeek(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            api_base=DEEPSEEK_BASE_URL,
            temperature=0.0,
            request_timeout=120,
        )

    def analyze(self, query: str, research_results: list[dict]) -> dict:
        """
        分析搜索结果质量

        :param query: 用户原始问题
        :param research_results: ResearchAgent.search() 的返回结果
        :return: {"is_sufficient": bool, "summary": str, "gaps": str,
                  "suggested_queries": list, "confidence": str, "raw_json": dict}
        """
        # 拼成 LLM 可读的文本
        results_text = self._format_results(research_results)
        prompt = ANALYST_PROMPT.format(query=query, search_results=results_text)

        print("    [分析] 正在评估信息质量...")
        try:
            response = self.llm.invoke(prompt)
            raw_text = response.content.strip()

            # DeepSeek 可能返回带 markdown 代码块的 JSON，清洗一下
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            analysis = json.loads(raw_text)

            return {
                "is_sufficient": analysis.get("is_sufficient", True),
                "summary": analysis.get("summary", ""),
                "gaps": analysis.get("gaps", ""),
                "suggested_queries": analysis.get("suggested_queries", []),
                "confidence": analysis.get("confidence", "medium"),
                "raw_json": analysis,
            }
        except Exception as e:
            print(f"    [!] 分析失败: {e}")
            # 兜底：认为信息不够，返回空分析
            return {
                "is_sufficient": False,
                "summary": "分析异常，无法生成摘要",
                "gaps": str(e),
                "suggested_queries": [],
                "confidence": "low",
                "raw_json": {},
            }

    @staticmethod
    def _format_results(results: list[dict]) -> str:
        """将搜索结果格式化为分析可读的文本"""
        lines = []
        idx = 1
        for group in results:
            lines.append(f"\n## 子问题: {group['question']}")
            for item in group.get("results", []):
                lines.append(f"\n[来源 {idx}] {item['title']}")
                lines.append(f"URL: {item['url']}")
                # 截取前 500 字符避免 prompt 过长
                content = item.get("content", "")
                lines.append(f"内容: {content[:500]}")
                idx += 1
        return "\n".join(lines)


# --- 验证脚本 ---
if __name__ == "__main__":
    from agents.research_agent import ResearchAgent

    # 先搜
    print("=" * 60)
    print("Step 1: 搜索")
    print("=" * 60)
    researcher = ResearchAgent()
    query = "DeepSeek大模型2025年发展规划"
    search_results = researcher.search([query])

    # 再分析
    print("\n" + "=" * 60)
    print("Step 2: 分析")
    print("=" * 60)
    analyst = AnalystAgent()
    analysis = analyst.analyze(query, search_results)

    print(f"\n>>> 分析结果:")
    print(f"    信息充分: {analysis['is_sufficient']}")
    print(f"    置信度:   {analysis['confidence']}")
    print(f"    摘要:     {analysis['summary'][:100]}...")
    print(f"    缺口:     {analysis['gaps'][:100] if analysis['gaps'] else '无'}")
    if analysis['suggested_queries']:
        print(f"    建议补充搜索: {analysis['suggested_queries']}")
