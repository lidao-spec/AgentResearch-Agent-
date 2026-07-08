# -*- coding: utf-8 -*-
"""配置模块：从 .env 文件和环境变量中读取 API Key"""
import os
import sys

# 解决 Windows 控制台 GBK 编码导致的 emoji 显示问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# 加载 .env 文件（基于 config.py 自身所在目录查找）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Tavily 搜索配置
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# 工作流参数
MAX_RETRY = 2           # 搜索验证最大回退次数
MAX_SEARCH_RESULTS = 3  # 每个子问题的搜索结果数


def validate_config() -> bool:
    """启动前校验：检查必要的 API Key 是否已配置"""
    missing = []
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")

    if missing:
        print(f"[!] 缺少必要的 API Key: {', '.join(missing)}")
        print("    请复制 .env.example 为 .env，并填入你的 Key")
        return False
    return True


if __name__ == "__main__":
    if validate_config():
        print("[OK] 配置校验通过")
        print(f"     DeepSeek Model: {DEEPSEEK_MODEL}")
        print(f"     DeepSeek Base URL: {DEEPSEEK_BASE_URL}")
        print(f"     Tavily API Key: {'***' + TAVILY_API_KEY[-4:] if TAVILY_API_KEY else 'NOT SET'}")
    else:
        print("    请先配置 .env 文件")
