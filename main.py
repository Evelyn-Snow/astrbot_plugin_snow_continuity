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
from astrbot.core.message.components import Image, Poke

from .library import should_boost

NAME = "astrbot_plugin_snow_continuity"
GCP_PLUGIN = "astrbot_plugin_group_chat_plus"


class SnowContinuity(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        # (platform, chat_id, sender_id) -> 时间戳
        self._pokes: dict[tuple[str, str, str], float] = {}
        self._interactions: dict[tuple[str, str, str], float] = {}
        self._gcp_warned = False

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
        """回复生成开始：即使还没发出去，也算一次进行中的互动。"""
        if event.get_platform_name() == "webchat":
            return
        self._interactions[self._key(event)] = time.time()
        self._prune()

    @filter.after_message_sent()
    async def record_reply_sent(self, event: AstrMessageEvent):
        """回复已发送：刷新最近互动时间。"""
        if event.get_platform_name() == "webchat":
            return
        self._interactions[self._key(event)] = time.time()
        self._prune()

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
