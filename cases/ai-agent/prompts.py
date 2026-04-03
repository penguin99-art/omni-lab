"""System prompt construction for the workstation agent."""

import os
import time
from pathlib import Path


def load_file(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def build_system_prompt(
    mode: str,
    *,
    goals_file: Path,
    memory_file: Path,
    today_file: Path,
    state_dir: Path,
) -> str:
    goals = load_file(goals_file)
    memory = load_file(memory_file)
    today = load_file(today_file)
    cwd = os.getcwd()
    now = time.strftime("%Y-%m-%d %H:%M %A")

    prompt = f"""你是这台工作机的 AI 助手。你不只是回答问题——你是这台机器的操作者。
用户是你的掌握者，给你方向和决策权。你负责执行、规划、记忆。

当前时间: {now}
工作目录: {cwd}
状态目录: {state_dir}

你可以：运行命令、读写文件、记住重要信息、更新今日计划。
遇到需要决策的事，简短问用户。执行完毕简短汇报结果。
用中文交流。简洁、直接、有用。不要客套废话。"""

    if goals:
        prompt += f"\n\n## 用户目标\n{goals}"
    if memory:
        prompt += f"\n\n## 持久记忆\n{memory}"
    if today:
        prompt += f"\n\n## 今日记录\n{today}"

    if mode == "morning":
        prompt += """

## 你的任务
现在是早上。请：
1. 回顾用户的目标和昨天的记录
2. 生成今天的工作计划（具体、可执行、有优先级）
3. 如果有需要提醒的事（基于记忆），主动提出
4. 把计划写入今日文件（用 update_today 工具）
5. 简短地跟用户汇报"""

    elif mode == "evening":
        prompt += """

## 你的任务
现在是晚上收工。请：
1. 回顾今天的对话和记录
2. 总结今天完成了什么、没完成什么
3. 有没有需要记住的新事实（用 memory_save）
4. 把总结追加到今日文件（用 update_today）
5. 简短跟用户说"""

    return prompt
