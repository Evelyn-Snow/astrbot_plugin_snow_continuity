"""连续对话接话的纯逻辑判断（不依赖 AstrBot，可单独测试）。

设计目标：和机器人刚互动过的人，紧接着发的、明显是说给机器人听的消息，
应该确定性得到回复，而不是掷骰子。判定为“呼唤”的条件（按顺序）：
- 发送者在最近窗口内戳过机器人（戳一戳不登记注意力，单独补上）；
- 发送者最近有过一次对话互动（回复中或已回复）且消息包含第二人称标记，
  或是纯图片（“先发图再说”的场景）；
- 发送者注意力分达到阈值且消息包含第二人称标记（兜底）。
"""

from __future__ import annotations

DEFAULT_MARKERS = ("你", "您", "在吗")


def contains_marker(text: str, markers: tuple[str, ...] = DEFAULT_MARKERS) -> bool:
    """消息里是否包含第二人称/呼唤类标记（子串匹配）。"""
    value = str(text or "")
    return any(marker and marker in value for marker in markers)


def is_recent(ts: float, now: float, window_seconds: int) -> bool:
    if ts <= 0:
        return False
    return (now - ts) <= max(1, int(window_seconds))


def should_boost(
    *,
    text: str,
    has_image: bool = False,
    markers: tuple[str, ...] = DEFAULT_MARKERS,
    recent_interaction: bool = False,
    attention_score: float = 0.0,
    attention_threshold: float = 0.3,
    poke_ts: float = 0.0,
    now: float | None = None,
    poke_window_seconds: int = 120,
) -> tuple[bool, str]:
    """判定这条群消息是否应该确定性接话。返回 (是否提升, 原因)。"""
    value = str(text or "").strip()
    if not value and not has_image:
        return False, "empty"
    if value.startswith(("/", "！")):
        return False, "command"

    import time as _time

    now_value = _time.time() if now is None else float(now)
    if is_recent(float(poke_ts or 0), now_value, poke_window_seconds):
        return True, "recent_poke"
    if recent_interaction:
        if has_image:
            return True, "recent_interaction_image"
        if contains_marker(value, markers):
            return True, "recent_interaction_marker"
    if contains_marker(value, markers) and float(attention_score or 0) >= float(
        attention_threshold
    ):
        return True, "second_person_with_attention"
    return False, "no_match"
