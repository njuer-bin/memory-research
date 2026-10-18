from __future__ import annotations

import re
from dataclasses import dataclass

from .temporal_parser import normalize_temporal


@dataclass
class QueryPlan:
    original: str
    rewritten: str
    multi_hop: bool
    keywords: list[str]
    temporal: bool
    temporal_start: int | None = None
    temporal_end: int | None = None
    temporal_relation: str = "at"
    memory_type_hint: str | None = None
    relation_hint: bool = False
    expanded_query: str = ""
    predicate_hint: str | None = None
    intent_hint: str | None = None


class QueryAnalyzer:
    MULTI_HOP_MARKERS = (
        "谁推荐", "谁介绍", "朋友的", "同事的", "他的", "她的",
        "他们", "那个", "之前提到", "基于", "根据", "为什么",
        "和谁", "关系", "哪个朋友", "朋友推荐", "同事推荐",
    )

    TEMPORAL_MARKERS = (
        "现在", "目前", "当前", "以前", "之前", "后来", "之后",
        "最近", "当时", "历史", "过去", "去年", "前年", "今年",
        "曾经", "上个月", "本月", "昨天", "今天", "明天",
    )

    def analyze(self, query: str, forced_multi_hop=None, reference_ts=None) -> QueryPlan:
        q = query.strip()
        rewritten = self.rewrite(q)
        multi = any(x in q for x in self.MULTI_HOP_MARKERS)
        if forced_multi_hop is not None:
            multi = forced_multi_hop
        temporal = any(x in q for x in self.TEMPORAL_MARKERS)
        if reference_ts is None:
            import time
            reference_ts = int(time.time() * 1000)
        info = normalize_temporal(q, reference_ts)
        keywords = self.keywords(rewritten)
        memory_type_hint = self.infer_memory_type(q)
        relation_hint = any(
            marker in q for marker in (
                "朋友", "同事", "推荐", "介绍", "谁和", "关系", "和谁"
            )
        )
        expanded_query = self.expand_query(rewritten, memory_type_hint, relation_hint, info.relation)
        predicate_hint = self.infer_predicate(q)
        intent_hint = self.infer_intent(q, memory_type_hint)
        return QueryPlan(
            q, rewritten, multi, keywords, temporal,
            info.start, info.end, info.relation,
            memory_type_hint, relation_hint, expanded_query, predicate_hint, intent_hint
        )

    @staticmethod
    def infer_memory_type(q: str) -> str | None:
        if any(x in q for x in ("习惯", "通常", "一般", "规则", "请记住", "总是")):
            return "rule"
        if any(x in q for x in ("什么时候", "何时", "哪天", "哪一年", "参加了什么", "发生了什么")):
            return "event"
        if any(x in q for x in ("朋友", "同事", "推荐", "介绍", "谁和", "关系", "和谁")):
            return "relation"
        if any(x in q for x in ("喜欢", "爱好", "偏好", "不喜欢")):
            return "fact"
        if any(x in q for x in ("住哪里", "住哪", "居住地", "住过")):
            return "fact"
        return None

    @staticmethod
    def infer_predicate(q: str) -> str | None:
        if any(x in q for x in ("不喜欢", "讨厌", "不爱")):
            return "dislike"
        if any(x in q for x in ("喜欢", "偏好", "爱好", "喜爱")):
            return "like"
        if any(x in q for x in ("职业", "工作", "从事")):
            return "occupation"
        if any(x in q for x in ("住哪里", "住哪", "居住地", "住过", "住址")):
            return "residence"
        if "生日" in q or "出生" in q:
            return "birthday"
        if any(x in q for x in ("名字", "姓名", "叫")):
            return "name"
        return None

    @staticmethod
    def infer_intent(q: str, memory_type_hint: str | None) -> str | None:
        if memory_type_hint == "rule":
            return "habit"
        if memory_type_hint == "event":
            return "event"
        if memory_type_hint == "relation":
            return "relation"
        if memory_type_hint == "fact":
            return "fact"
        return None

    @staticmethod
    def expand_query(q: str, memory_type_hint: str | None, relation_hint: bool,
                      temporal_relation: str) -> str:
        """仅用于召回阶段的确定性语义扩展；不改变最终回答所依据的原始查询。"""
        terms = []
        if memory_type_hint == "rule":
            terms += ["习惯", "通常", "一般", "经常", "平时", "规则"]
        elif memory_type_hint == "fact":
            if any(x in q for x in ("偏好", "喜欢", "爱好", "喜爱", "不喜欢")):
                terms += ["偏好", "喜欢", "爱好", "喜爱"]
            if any(x in q for x in ("职业", "工作", "从事")):
                terms += ["职业", "工作", "从事"]
            if any(x in q for x in ("住哪里", "住哪", "居住地", "住过")):
                terms += ["居住地", "住处", "居住", "以前", "曾经", "之前"]
        elif memory_type_hint == "event":
            terms += ["事件", "参加", "发生", "经历"]
        if relation_hint:
            terms += ["朋友", "好友", "同事", "推荐", "介绍", "关系"]
        if temporal_relation == "before":
            terms += ["以前", "之前", "曾经", "历史"]
        elif temporal_relation == "after":
            terms += ["后来", "之后"]
        unique = list(dict.fromkeys(x for x in terms if x not in q))
        return q if not unique else q + " " + " ".join(unique)

    @staticmethod
    def rewrite(q: str) -> str:
        replacements = {
            "我现在住哪": "用户 当前 居住地",
            "我现在住哪里": "用户 当前 居住地",
            "我住哪里": "用户 当前 居住地",
            "我住哪": "用户 当前 居住地",
            "以前住哪里": "用户 历史 居住地",
            "之前住哪里": "用户 历史 居住地",
            "我喜欢什么": "用户 喜欢 偏好",
            "我的偏好是什么": "用户 喜欢 偏好",
            "我的偏好是": "用户 喜欢 偏好",
            "我的爱好": "用户 喜欢 偏好",
        }
        out = q
        for a, b in replacements.items():
            out = out.replace(a, b)
        return out

    @staticmethod
    def keywords(q: str) -> list[str]:
        chars = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", q.lower())
        grams = []
        for i in range(len(chars) - 1):
            if all("\u4e00" <= c <= "\u9fff" for c in chars[i:i + 2]):
                grams.append(chars[i] + chars[i + 1])
        return list(dict.fromkeys(chars + grams))
