"""连续对话接话：让“刚互动过的人紧接着说的话”被确定性回复。

群聊插件（group_chat_plus）对非 @ 消息采用读空气概率，连续对话中的追问
偶尔会被骰子吞掉。本插件不自己回复，而是在 GCP 概率判定前把这类消息的
概率临时拉满，由 GCP 走正常流程接话：

- 发送者最近有过对话互动（回复生成中或已回复，钩子自行记录）且消息包含
  “你/您”等第二人称标记，或是纯图片（“先发图再说”场景）
- 发送者注意力分达标（GCP 注意力档案兜底）
- 发送者在最近窗口内戳过机器人（戳一戳不登记注意力，这里补上）

依赖 group_chat_plus（astrbot_plugin_group_chat_plus）内部的
ProbabilityManager / AttentionManager，经插件注册表动态定位；
GCP 未安装时静默停用。
"""

from __future__ import annotations

import importlib
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.message.components import Image, Plain, Poke

from .library import (
    END_MARK,
    echo_suppressed,
    should_boost,
    strip_end_mark,
)

NAME = "astrbot_plugin_snow_continuity"
GCP_PLUGIN = "astrbot_plugin_group_chat_plus"
CLOSE_MARK = "[对话收束]"


class SnowContinuity(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        # (platform, chat_id, sender_id) -> 时间戳
        self._pokes: dict[tuple[str, str, str], float] = {}
        self._interactions: dict[tuple[str, str, str], float] = {}
        self._gcp_warned = False
        # umo -> (收束截止时间, 收束对象 sender)：道别后对回声静默
        self._closed: dict[str, tuple[float, str]] = {}
        self._last_sender: dict[str, str] = {}
        self._turn_count: dict[str, int] = {}

    # ---------------- 配置 ----------------
    def _enabled(self) -> bool:
        return bool(self.conf.get("enabled", True))

    def _markers(self) -> tuple[str, ...]:
        raw = str(self.conf.get("markers") or "你,您,在吗")
        return tuple(m.strip() for m in raw.replace("，", ",").split(",") if m.strip())

    def _interaction_window(self) -> int:
        return max(30, int(self.conf.get("interaction_window_seconds") or 600))

    # ---------------- GCP 内部接口 ----------------
    def _gcp(self):
        """经插件注册表定位 GCP 的概率/注意力管理器；未安装或重载后自动恢复。"""
        try:
            meta = self.context.get_registered_star(GCP_PLUGIN)
            plugin = getattr(meta, "star_cls", None) if meta else None
            module = getattr(meta, "module", None) or type(plugin).__module__
            if plugin is None or not module:
                raise RuntimeError("group_chat_plus 未注册")
            package = str(module).rsplit(".", 1)[0]
            utils = importlib.import_module(f"{package}.utils")
            return utils.probability_manager, utils.attention_manager
        except Exception as exc:
            if not self._gcp_warned:
                self._gcp_warned = True
                logger.info(f"[{NAME}] group_chat_plus 不可用，连续接话停用：{exc}")
            return None, None

    @staticmethod
    def _key(event: AstrMessageEvent, sender: str | None = None) -> tuple[str, str, str]:
        return (
            event.get_platform_name(),
            str(event.get_group_id() or ""),
            str(sender or event.get_sender_id() or ""),
        )

    def _prune(self) -> None:
        now = time.time()
        poke_window = max(60, int(self.conf.get("poke_window_seconds") or 120))
        interaction_window = self._interaction_window()
        stores = (
            (self._pokes, poke_window),
            (self._interactions, interaction_window),
        )
        for store, window in stores:
            for key, ts in list(store.items()):
                if now - ts > window * 4:
                    store.pop(key, None)

    # ---------------- 互动记录 ----------------
    @filter.on_llm_request()
    async def record_reply_in_flight(self, event: AstrMessageEvent, req):
        """回复生成开始：记录发送者并注入收束指引；互动记录不含网页测试台。"""
        if not self._enabled():
            return
        self._record_sender_and_hint(event, req)
        if event.get_platform_name() == "webchat":
            return
        self._interactions[self._key(event)] = time.time()
        self._prune()

    def _record_sender_and_hint(self, event: AstrMessageEvent, req) -> None:
        """记录本次发送者（供道别回声抑制匹配），并注入对话收束指引。"""
        umo = str(event.unified_msg_origin or "")
        sender = str(getattr(event, "get_sender_id", lambda: "")() or "")
        if umo and sender:
            self._last_sender[umo] = sender
            if len(self._last_sender) > 100:
                self._last_sender.pop(next(iter(self._last_sender)))
        if not bool(self.conf.get("farewell_enabled", True)):
            return
        # 收尾指引只在「可能收尾」时注入，省掉每轮 160 字的固定开销：
        # 用户消息含道别词，或本会话已聊够轮数（每 N 轮补一次提醒）。
        if not self._should_hint_close(event, umo):
            return
        prompt = str(getattr(req, "system_prompt", "") or "")
        if CLOSE_MARK in prompt:
            return
        req.system_prompt = prompt + (
            "\n\n[对话收束] 当对话已经自然结束——互道晚安或再见、对方要去休息或下线、"
            "或话题完整收尾——你的回复应是一句简短的道别，并在回复的最末尾另起一行单独输出："
            "[[对话结束]]。除此之外（对话还在进行、刚开头、或对方在提新话题）绝对不要输出这个标记。"
            "输出后系统会替你收尾，对方再发简单的道别也不会打扰你。这段说明不要在回复中提到。"
        )

    _CLOSE_WORDS = (
        "晚安", "睡了", "睡了睡", "睡觉", "下线", "拜拜", "再见", "回见", "明天见",
        "先不聊", "不聊了", "去休息", "要睡了", "先撤", "溜了", "退下了",
    )

    def _should_hint_close(self, event: AstrMessageEvent, umo: str) -> bool:
        """是否需要注入收尾指引：命中道别词，或每 N 轮补一次（保证长对话也能收尾）。"""
        text = str(getattr(event, "message_str", "") or "")
        if any(word in text for word in self._CLOSE_WORDS):
            return True
        turn = self._turn_count.get(umo, 0) + 1
        self._turn_count[umo] = turn
        if len(self._turn_count) > 100:
            self._turn_count.pop(next(iter(self._turn_count)))
        every = int(self.conf.get("farewell_hint_every_turns", 5))
        if every <= 0:
            return False
        return turn % every == 0

    @filter.after_message_sent()
    async def record_reply_sent(self, event: AstrMessageEvent):
        """回复已发送：刷新最近互动时间。"""
        if not self._enabled():
            return
        if event.get_platform_name() == "webchat":
            return
        self._interactions[self._key(event)] = time.time()
        self._prune()

    # ---------------- 对话收束 ----------------
    @filter.on_decorating_result(priority=-1200)
    async def close_conversation(self, event: AstrMessageEvent):
        """响应里带结束标记 → 剥掉标记、记下收束期；之后对方的道别回声不再接。"""
        if not self._enabled():
            return
        if not bool(self.conf.get("farewell_enabled", True)):
            return
        result = event.get_result()
        if result is None or not getattr(result, "chain", None):
            return
        texts = [c.text for c in result.chain if isinstance(c, Plain)]
        if not any(END_MARK in str(t or "") for t in texts):
            return
        for comp in result.chain:
            if isinstance(comp, Plain):
                comp.text = strip_end_mark(comp.text)
        # 纯标记消息（清洗后为空）直接移除，避免发出空白
        result.chain = [
            c
            for c in result.chain
            if not (isinstance(c, Plain) and not str(c.text or "").strip())
        ]
        umo = str(event.unified_msg_origin or "")
        minutes = max(1, int(self.conf.get("farewell_cooldown_minutes", 10)))
        sender = self._last_sender.get(umo) or str(
            getattr(event, "get_sender_id", lambda: "")() or ""
        )
        self._closed[umo] = (time.time() + minutes * 60, sender)
        if len(self._closed) > 100:
            now = time.time()
            for key in [k for k, v in self._closed.items() if v[0] <= now]:
                self._closed.pop(key, None)
        logger.info(f"[{NAME}] 对话已收束：{umo}（{minutes} 分钟内忽略对方的道别回声）")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=(1 << 31) - 4)
    async def suppress_farewell_echo(self, event: AstrMessageEvent):
        """收束期内，同一对象发来的纯道别回声直接忽略（实质内容照常回复）。"""
        if not self._enabled():
            return
        if not bool(self.conf.get("farewell_enabled", True)):
            return
        umo = str(event.unified_msg_origin or "")
        closed = self._closed.get(umo)
        if not closed:
            return
        until, sender = closed
        now = time.time()
        if now >= until:
            self._closed.pop(umo, None)
            return
        if echo_suppressed(
            until,
            str(getattr(event, "get_sender_id", lambda: "")() or ""),
            sender,
            str(event.message_str or ""),
            now,
        ):
            event.stop_event()
            logger.info(f"[{NAME}] 已忽略收束后的道别回声：{umo}")

    # ---------------- 事件 ----------------
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def watch_group_message(self, event: AstrMessageEvent):
        if not self._enabled():
            return
        try:
            if self._record_poke_if_needed(event):
                return
            await self._boost_if_addressed(event)
        except Exception as exc:
            logger.warning(f"[{NAME}] 处理群消息失败: {exc}")

    def _record_poke_if_needed(self, event: AstrMessageEvent) -> bool:
        """记录“戳机器人”的动作，返回这条消息是否为戳一戳。"""
        components = getattr(event.message_obj, "message", None) or []
        if not any(isinstance(comp, Poke) for comp in components):
            return False
        sender = str(event.get_sender_id() or "")
        self_id = str(event.get_self_id() or "")
        if not sender or sender == self_id:
            return True  # 机器人自己的戳，忽略
        for comp in components:
            if not isinstance(comp, Poke):
                continue
            target = str(comp.target_id() or "")
            if target and self_id in (target, ""):
                self._pokes[self._key(event, sender)] = time.time()
                logger.info(f"[{NAME}] 记录戳一戳：{sender}")
                break
        self._prune()
        return True

    async def _boost_if_addressed(self, event: AstrMessageEvent) -> None:
        components = getattr(event.message_obj, "message", None) or []
        has_image = any(isinstance(comp, Image) for comp in components)
        text = str(getattr(event, "message_str", "") or "")
        sender = str(event.get_sender_id() or "")
        chat_id = str(event.get_group_id() or "")
        if (not text and not has_image) or not sender or not chat_id:
            return
        probability_manager, attention_manager = self._gcp()
        if probability_manager is None:
            return

        attention_score = 0.0
        try:
            profile = await attention_manager.AttentionManager.get_attention_info(
                event.get_platform_name(), False, chat_id, sender
            )
            if isinstance(profile, dict):
                attention_score = float(profile.get("attention_score") or 0)
        except Exception:
            attention_score = 0.0

        key = self._key(event, sender)
        now = time.time()
        interaction_window = self._interaction_window()
        recent_interaction = (now - self._interactions.get(key, 0)) <= interaction_window
        boost, reason = should_boost(
            text=text,
            has_image=has_image,
            markers=self._markers(),
            recent_interaction=recent_interaction,
            attention_score=attention_score,
            attention_threshold=float(self.conf.get("attention_threshold") or 0.3),
            poke_ts=self._pokes.get(key, 0.0),
            now=now,
            poke_window_seconds=int(self.conf.get("poke_window_seconds") or 120),
        )
        if not boost:
            return

        probability = max(0.5, min(0.98, float(self.conf.get("boost_probability") or 0.98)))
        duration = max(30, int(self.conf.get("boost_duration_seconds") or 120))
        await probability_manager.ProbabilityManager.boost_probability(
            event.get_platform_name(), False, chat_id, probability, duration
        )
        logger.info(
            f"[{NAME}] 连续对话呼唤（{reason}，注意力={attention_score:.2f}，"
            f"近期互动={recent_interaction}）→ 概率提升至 {probability:.2f}/{duration}s："
            f"{text[:50] or '<图片>'}"
        )
