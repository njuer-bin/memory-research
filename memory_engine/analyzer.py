from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .models import fingerprint, new_id
from .temporal_parser import normalize_temporal


@dataclass
class Fact:
    id: str
    user_id: str
    subject: str
    predicate: str
    object: str
    content: str
    timestamp: int
    fingerprint: str
    valid_from: int
    valid_to: Optional[int] = None
    status: str = "active"
    supersedes_id: Optional[str] = None
    temporal_text: str = ""
    source: str = "user"
    conflict_status: str = "none"
    conflict_group_id: Optional[str] = None
    source_raw_id: Optional[str] = None
    source_session_id: Optional[str] = None
    source_turn_id: Optional[str] = None


@dataclass
class Relation:
    id: str
    user_id: str
    subject: str
    predicate: str
    object: str
    content: str
    timestamp: int
    fingerprint: str
    source_raw_id: Optional[str] = None
    source_session_id: Optional[str] = None
    source_turn_id: Optional[str] = None


@dataclass
class Event:
    id: str
    user_id: str
    event: str
    content: str
    timestamp: int
    fingerprint: str
    event_start: Optional[int] = None
    event_end: Optional[int] = None
    temporal_text: str = ""
    source_raw_id: Optional[str] = None
    source_session_id: Optional[str] = None
    source_turn_id: Optional[str] = None


@dataclass
class Rule:
    id: str
    user_id: str
    rule: str
    content: str
    timestamp: int
    fingerprint: str
    source_raw_id: Optional[str] = None
    source_session_id: Optional[str] = None
    source_turn_id: Optional[str] = None


@dataclass
class Profile:
    user_id: str
    key: str
    value: str
    content: str
    timestamp: int
    source_raw_id: Optional[str] = None
    source_session_id: Optional[str] = None
    source_turn_id: Optional[str] = None


class MemoryAnalyzer:
    """Deterministic Parse-2 analyzer.

    覆盖：
    - atomic fact / profile
    - relation
    - rule / preference
    - event + temporal expression
    - 否定、纠正、迁移类事实
    - 一句话中的多事实抽取

    不依赖外部模型，后续可以把 analyze() 替换成 LLM extractor。
    """

    FACT_PATTERNS = [
        (re.compile(r"我(?:现在|目前|当前)?住在([^\n，。,.；;]+)"), "residence"),
        (re.compile(r"我(?:目前|现在)?在([^\n，。,.；;]+?)(?:工作|上班)"), "workplace"),
        (re.compile(r"我(?:的)?职业是([^\n，。,.；;]+)"), "occupation"),
        (re.compile(r"我最喜欢([^\n。；;，,]+)"), "favorite"),
        (re.compile(r"我喜欢([^\n。；;，,]+)"), "like"),
        (re.compile(r"我不喜欢([^\n。；;，,]+)"), "dislike"),
        (re.compile(r"我讨厌([^\n。；;，,]+)"), "dislike"),
        (re.compile(r"我的生日是([^\n，。,.；;]+)"), "birthday"),
        (re.compile(r"我的名字是([^\n，。,.；;]+)"), "name"),
        (re.compile(r"我叫([^\n，。,.；;]+)"), "name"),
        (re.compile(r"我常用的语言是([^\n，。,.；;]+)"), "language"),
        (re.compile(r"我来自([^\n，。,.；;]+)"), "origin"),
        (re.compile(r"我毕业于([^\n，。,.；;]+)"), "school"),
        # English benchmark coverage (e.g. LongMemEval).
        (re.compile(r"\bI\s+graduated\s+with\s+(?:a|an)\s+degree\s+in\s+([^\n.,;!?]+)", re.I), "degree"),
        (re.compile(r"\bI\s+graduated\s+from\s+([^\n.,;!?]+)", re.I), "school"),
        (re.compile(r"\bI\s+(?:currently\s+)?live\s+in\s+([^\n.,;!?]+)", re.I), "residence"),
        (re.compile(r"\bI\s+(?:currently\s+)?work\s+(?:at|for)\s+([^\n.,;!?]+)", re.I), "workplace"),
        (re.compile(r"\bI\s+(?:am|\'m)\s+(?:a|an)\s+([^\n.,;!?]+)", re.I), "occupation"),
        (re.compile(r"\bI\s+(?:really\s+)?like\s+([^\n.,;!?]+)", re.I), "like"),
        (re.compile(r"\bI\s+(?:really\s+)?(?:do not|don\'t)\s+like\s+([^\n.,;!?]+)", re.I), "dislike"),
        (re.compile(r"\bI\s+hate\s+([^\n.,;!?]+)", re.I), "dislike"),
        (re.compile(r"\bMy\s+birthday\s+is\s+([^\n.,;!?]+)", re.I), "birthday"),
        (re.compile(r"\bMy\s+name\s+is\s+([^\n.,;!?]+)", re.I), "name"),
        (re.compile(r"\bI\s+(?:speak|use)\s+([^\n.,;!?]+)", re.I), "language"),
        (re.compile(r"\bI\s+(?:am\s+)?from\s+([^\n.,;!?]+)", re.I), "origin"),
    ]

    REL_PATTERNS = [
        (re.compile(r"(?:我的|我)?(?:朋友|好友)\s*(?:是|叫|为)?\s*([A-Za-z0-9_\u4e00-\u9fff]{1,4})(?=推荐|介绍|、|，|。|\s|$)"), "friend"),
        (re.compile(r"(?:我的|我)?同事\s*(?:是|叫|为)?\s*([A-Za-z0-9_\u4e00-\u9fff]{1,4})(?=推荐|介绍|、|，|。|\s|$)"), "colleague"),
        (re.compile(r"(?:我的|我)?老板\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "boss"),
        (re.compile(r"(?:我的|我)?(?:妈妈|母亲)\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "mother"),
        (re.compile(r"(?:我的|我)?(?:爸爸|父亲)\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "father"),
        (re.compile(r"(?:我的|我)?妻子\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "wife"),
        (re.compile(r"(?:我的|我)?丈夫\s*([A-Za-z0-9_\u4e00-\u9fff]{1,20})"), "husband"),
        (re.compile(r"\bmy\s+friend\s+is\s+([A-Za-z][A-Za-z0-9_-]{1,40})", re.I), "friend"),
        (re.compile(r"\bmy\s+colleague\s+is\s+([A-Za-z][A-Za-z0-9_-]{1,40})", re.I), "colleague"),
        (re.compile(r"\bmy\s+(?:boss|manager)\s+is\s+([A-Za-z][A-Za-z0-9_-]{1,40})", re.I), "boss"),
        (re.compile(r"\bmy\s+(?:mother|mom)\s+is\s+([A-Za-z][A-Za-z0-9_-]{1,40})", re.I), "mother"),
        (re.compile(r"\bmy\s+(?:father|dad)\s+is\s+([A-Za-z][A-Za-z0-9_-]{1,40})", re.I), "father"),
        (re.compile(r"\bmy\s+(?:wife|husband)\s+is\s+([A-Za-z][A-Za-z0-9_-]{1,40})", re.I), "spouse"),
    ]

    RULE_PATTERNS = [
        re.compile(r"(?:以后|今后|从现在开始)[，,:： ]*(.*)"),
        re.compile(r"(?:请记住|记住)[，,:： ]*(.*)"),
        re.compile(r"(?:我的习惯是)[，,:： ]*(.*)"),
        re.compile(r"(?:我通常|我一般)(.*)"),
        re.compile(r"\bremember(?: that)?\s+(.+)", re.I),
        re.compile(r"\bfrom now on[, ]+(.+)", re.I),
        re.compile(r"\bi usually\s+(.+)", re.I),
        re.compile(r"\bi normally\s+(.+)", re.I),
    ]

    EVENT_WORDS = (
        "搬到", "搬家", "毕业", "入职", "离职", "结婚", "分手",
        "旅行", "去过", "参加", "开始", "结束", "购买", "买了",
        "完成", "搬去", "搬来", "加入", "辞职", "回到",
        "graduated", "moved", "married", "divorced", "started", "finished",
        "joined", "left", "bought", "purchased", "visited", "traveled",
    )

    CORRECTION_MARKERS = ("不是", "改成", "改为", "其实是", "更正为", "纠正一下")

    def analyze(self, user_id: str, content: str, timestamp: int, source: str = "user",
                source_raw_id: Optional[str] = None, source_session_id: Optional[str] = None,
                source_turn_id: Optional[str] = None):
        facts: list[Fact] = []
        relations: list[Relation] = []
        events: list[Event] = []
        rules: list[Rule] = []
        profiles: list[Profile] = []

        temporal = normalize_temporal(content, timestamp)
        temporal_text = temporal.text

        for pattern, predicate in self.FACT_PATTERNS:
            for m in pattern.finditer(content):
                current_predicate = predicate
                obj = m.group(1).strip(" ，,。；;")
                if not obj:
                    continue

                # “我不喜欢X”不应被 like 规则截断为“不喜欢X”
                if current_predicate == "like" and obj.startswith(("不", "讨厌")):
                    current_predicate = "dislike"
                if current_predicate == "like" and re.match(r"^(?:do not|don't)\s+", obj, re.I):
                    current_predicate = "dislike"

                fact_text = m.group(0).strip()
                fp = fingerprint(user_id, "fact", "user", current_predicate, obj)

                valid_from = temporal.start if temporal.start is not None else timestamp
                valid_to = temporal.end
                facts.append(Fact(
                    id=new_id("fact"),
                    user_id=user_id,
                    subject="user",
                    predicate=current_predicate,
                    object=obj,
                    content=fact_text,
                    timestamp=timestamp,
                    fingerprint=fp,
                    valid_from=valid_from,
                    valid_to=valid_to,
                    temporal_text=temporal_text,
                    source=source,
                    source_raw_id=source_raw_id, source_session_id=source_session_id,
                    source_turn_id=source_turn_id,
                ))
                profiles.append(Profile(
                    user_id=user_id, key=current_predicate, value=obj,
                    content=fact_text, timestamp=timestamp,
                    source_raw_id=source_raw_id, source_session_id=source_session_id,
                    source_turn_id=source_turn_id
                ))

        for pattern, predicate in self.REL_PATTERNS:
            for m in pattern.finditer(content):
                value = m.group(1).strip()
                if not value or value in {"推荐我", "介绍我", "告诉我", "说"}:
                    continue
                relations.append(Relation(
                    id=new_id("rel"),
                    user_id=user_id,
                    subject="user",
                    predicate=predicate,
                    object=value,
                    content=m.group(0).strip(),
                    timestamp=timestamp,
                    fingerprint=fingerprint(user_id, "rel", "user", predicate, value),
                    source_raw_id=source_raw_id, source_session_id=source_session_id,
                    source_turn_id=source_turn_id,
                ))

        for pattern in self.RULE_PATTERNS:
            for match in pattern.finditer(content):
                value = match.group(1).strip(" ，,。；;")
                if value:
                    rules.append(Rule(
                        id=new_id("rule"), user_id=user_id, rule=value,
                        content=match.group(0).strip(), timestamp=timestamp,
                        fingerprint=fingerprint(user_id, "rule", value),
                        source_raw_id=source_raw_id, source_session_id=source_session_id,
                        source_turn_id=source_turn_id
                    ))

        # 事件：一条消息只要包含事件词就保留原句作为证据，并结构化时间。
        for word in self.EVENT_WORDS:
            if word in content:
                events.append(Event(
                    id=new_id("event"), user_id=user_id, event=word,
                    content=content.strip(), timestamp=timestamp,
                    fingerprint=fingerprint(user_id, "event", content.strip(), word),
                    event_start=temporal.start,
                    event_end=temporal.end,
                    temporal_text=temporal_text,
                    source_raw_id=source_raw_id, source_session_id=source_session_id,
                    source_turn_id=source_turn_id,
                ))
                break

        return {
            "facts": facts,
            "relations": relations,
            "events": events,
            "rules": rules,
            "profiles": profiles,
            "temporal": temporal,
            "correction": any(x in content for x in self.CORRECTION_MARKERS),
        }
