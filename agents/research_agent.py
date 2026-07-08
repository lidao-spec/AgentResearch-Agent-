# -*- coding: utf-8 -*-
"""ResearchAgent —— 基于 Tavily Search API 的联网信息采集"""
import sys
import os

# 确保编码正常
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_tavily import TavilySearch

# 项目内配置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TAVILY_API_KEY, MAX_SEARCH_RESULTS


class ResearchAgent:
    """联网搜索采集 Agent"""

    def __init__(self):
        self.tool = TavilySearch(
            tavily_api_key=TAVILY_API_KEY,
            max_results=MAX_SEARCH_RESULTS,
        )

    def search(self, sub_questions: list[str]) -> list[dict]:
        """
        对每个子问题逐一搜索，返回汇总结果

        :param sub_questions: 子问题列表，如 ["AI教育市场规模", "AI教育主要玩家"]
        :return: [{"question": "...", "results": [{title, content, url}, ...]}, ...]
        """
        all_results = []
        total = len(sub_questions)

        for i, question in enumerate(sub_questions, 1):
            print(f"    [{i}/{total}] 搜索: {question}")
            try:
                raw = self.tool.invoke(question)
                results = self._parse_results(raw)
                all_results.append({
                    "question": question,
                    "results": results
                })
            except Exception as e:
                print(f"    [!] 搜索失败 ({question}): {e}")
                all_results.append({
                    "question": question,
                    "results": [],
                    "error": str(e)
                })

        return all_results

    def search_single(self, question: str) -> dict:
        """单次搜索（用于补充搜索）"""
        print(f"    [补充] 搜索: {question}")
        try:
            raw = self.tool.invoke(question)
            return {
                "question": question,
                "results": self._parse_results(raw)
            }
        except Exception as e:
            print(f"    [!] 补充搜索失败: {e}")
            return {"question": question, "results": [], "error": str(e)}

    @staticmethod
    def _parse_results(raw: dict) -> list[dict]:
        """将 Tavily 原始返回（dict 格式，results 字段包含搜索列表）格式化为统一结构"""
        items = raw.get("results", []) if isinstance(raw, dict) else raw
        parsed = []
        for item in items:
            parsed.append({
                "title": item.get("title", "") or "(无标题)",
                "content": item.get("content", ""),
                "url": item.get("url", ""),
            })
        return parsed


# --- 验证脚本 ---
if __name__ == "__main__":
    agent = ResearchAgent()
    questions = ["DeepSeek大模型最新消息 2025", "AI教育行业发展趋势"]
    results = agent.search(questions)

    print("\n" + "=" * 60)
    print("ResearchAgent 验证结果")
    print("=" * 60)
    for r in results:
        print(f"\n>> 子问题: {r['question']}")
        print(f"   命中 {len(r['results'])} 条结果:")
        for item in r['results']:
            print(f"   - [{item['title'][:50]}...]")
            print(f"     {item['url']}")
