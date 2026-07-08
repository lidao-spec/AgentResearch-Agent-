# -*- coding: utf-8 -*-
"""
Multi-Agent 智能研报系统 — CLI 入口
基于 LangGraph 编排 3 个 Agent（研究员、分析师、撰写员）协作生成行业研究报告
"""
import sys
import os
from datetime import datetime
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import validate_config
from graph.workflow import run_report


def main():
    print("=" * 60)
    print("    Multi-Agent 智能研报系统")
    print("    基于 LangGraph | DeepSeek | Tavily")
    print("=" * 60)

    # 配置校验
    if not validate_config():
        print("\n请配置 .env 文件后重新运行。")
        sys.exit(1)

    print()

    # 获取用户输入
    query = input("请输入您想研究的主题:\n> ").strip()
    if not query:
        print("主题不能为空，已退出。")
        sys.exit(0)

    print()

    # 计时
    start_time = time.time()

    # 执行工作流
    try:
        result = run_report(query)
    except KeyboardInterrupt:
        print("\n\n[!] 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[!] 执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time

    # 保存报告
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/report_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {query}\n\n")
        f.write(result["report"])

    print(f"\n{'=' * 60}")
    print(f"  报告已保存至: {filename}")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
