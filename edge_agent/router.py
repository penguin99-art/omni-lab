"""IntentRouter: decides whether user input goes to System 1 (fast) or System 2 (slow)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class IntentRouter(ABC):
    @abstractmethod
    def classify(self, text: str, visual_context: str = "") -> str:
        """Return 'fast' or 'slow'."""
        ...


class KeywordRouter(IntentRouter):
    """
    V1 router: keyword matching. Zero latency, zero extra inference.

    Default: fast (System 1). Only explicit triggers go slow.
    False-positive cost is high (interrupts full-duplex + wait), so triggers are strict.
    """

    TRIGGERS: list[str] = [
        "搜索", "搜一下", "查一下", "查查", "找一下", "找找",
        "帮我", "帮忙",
        "记住", "记一下", "别忘了", "记录",
        "提醒", "提醒我",
        "计算", "算一下", "算算", "分析",
        "打开", "运行", "执行", "下载", "创建", "写一个",
        "总结", "翻译",
        "上次", "上周", "之前那个", "叫什么",
    ]

    def classify(self, text: str, visual_context: str = "") -> str:
        for trigger in self.TRIGGERS:
            if trigger in text:
                return "slow"
        return "fast"
