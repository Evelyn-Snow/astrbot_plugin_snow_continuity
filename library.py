"""连续对话接话的纯逻辑判断（不依赖 AstrBot，可单独测试）。

设计目标：和机器人刚互动过的人，紧接着发的、明显是说给机器人听的消息，
应该确定性得到回复，而不是掷骰子。判定为“呼唤”的条件（按顺序）：
- 发送者在最近窗口内戳过机器人（戳一戳不登记注意力，单独补上）；
- 发送者最近有过一次对话互动（回复中或已回复）且消息包含第二人称标记，
  或是纯图片（“先发图再说”的场景）；
- 发送者注意力分达到阈值且消息包含第二人称标记（兜底）。
"""

from __future__ import annotations

import re

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


# ---------------- 对话收束（道别回声抑制） ----------------

END_MARK = "[[对话结束]]"

# 道别/收尾类回声：短消息且命中这些说法，视为“对道别的回应”而非新话题
_FAREWELL_RE = re.compile(
    r"(晚安|好梦|拜拜|拜了|再见|回见|明天见|下次见|睡了|睡觉|休息|下线|溜了|"
    r"你也是|我也是|你先|我先|你快点睡|早点睡|早点休息|好梦|嗯嗯|嗯呐|好的|好哒|嗯|哦|噢|行)"
)


def is_farewell_echo(text: str, max_len: int = 12) -> bool:
    """是否为道别回声：短消息且命中道别/附和类说法。长消息或有实质内容不算。"""
    value = re.sub(r"\s+", "", str(text or ""))
    if not value or len(value) > max_len:
        return False
    if value.startswith(("/", "！")):
        return False
    return bool(_FAREWELL_RE.search(value))


def has_end_mark(text: str) -> bool:
    return END_MARK in str(text or "")


def strip_end_mark(text: str) -> str:
    """剥掉结束标记及其所在行，清理多余空行（用于发出前清洗）。"""
    value = str(text or "")
    if END_MARK not in value:
        return value
    lines = [
        line for line in value.splitlines() if END_MARK not in line
    ]
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def echo_suppressed(
    closed_until: float, sender_id: str, active_sender: str, text: str, now: float
) -> bool:
    """收束期内该发送者的道别回声 → 忽略。实质内容或他人消息不拦。"""
    if now >= float(closed_until or 0):
        return False
    if str(sender_id or "") != str(active_sender or ""):
        return False
    return is_farewell_echo(text)
