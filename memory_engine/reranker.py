from __future__ import annotations

import logging
import os
import time

from .store import tokenize


logger = logging.getLogger(__name__)


class LightweightReranker:
    """
    无需下载模型的可解释 fallback。
    如果以后接入 BGE reranker，只需替换 score 方法。
    """

    QUERY_STOPWORDS = {
        "我", "我的", "你", "你的", "他", "她", "它", "我们",
        "是", "什么", "哪个", "哪些", "哪里", "哪", "怎么",
        "如何", "吗", "呢", "啊", "呀", "请问",
    }

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        return {
            token for token in tokenize(text)
            if token not in cls.QUERY_STOPWORDS
        }

    @staticmethod
    def _ngrams(text: str, min_n: int = 2, max_n: int = 4) -> set[str]:
        tokens = tokenize(text)
        if len(tokens) < min_n:
            return set()
        grams = set()
        for n in range(min_n, min(max_n, len(tokens)) + 1):
            grams.update(
                "".join(tokens[i:i + n])
                for i in range(len(tokens) - n + 1)
            )
        return grams

    def score(self, query: str, content: str) -> float:
        q = self._tokens(query)
        d = self._tokens(content)
        if not q or not d:
            return 0.0

        overlap = len(q & d) / len(q)

        # 中文短语重合对自然语言问题很重要，可以区分
        # “喜欢杭州”和“周末喜欢吃火锅”这类仅共享“喜欢”的候选。
        q_grams = self._ngrams(query)
        d_grams = self._ngrams(content)
        phrase_overlap = (
            len(q_grams & d_grams) / len(q_grams)
            if q_grams else 0.0
        )

        normalized_query = "".join(tokenize(query))
        normalized_content = "".join(tokenize(content))
        exact_phrase = (
            0.35
            if normalized_query and normalized_query in normalized_content
            else 0.0
        )

        return min(
            1.0,
            0.60 * overlap
            + 0.40 * phrase_overlap
            + exact_phrase,
        )

    def rerank(
            self,
            query,
            results,
            top_k,
            memory_type_hint=None,
            relation_hint=False,
            temporal_relation="at",
            expanded_query=None,
            predicate_hint=None,
            intent_hint=None,
    ):
        rescored = []
        t0 = time.perf_counter()

        for r in results:
            lexical = self.score(query, r["content"])
            expanded_lexical = self.score(
                expanded_query or query,
                r["content"],
            )
            lexical_signal = (
                    0.70 * lexical + 0.30 * expanded_lexical
            )

            base = float(r.get("score", 0.0))

            # 保留 RRF 分数的区分度，避免过早饱和。
            retrieval_signal = min(1.0, max(0.0, base * 20.0))

            mt = r.get("memory_type")
            metadata = r.get("metadata") or {}
            doc_predicate = metadata.get("predicate")

            structured = 0.0

            # 类型匹配
            if memory_type_hint and mt == memory_type_hint:
                structured += 0.15

            # 关系查询
            if relation_hint and mt == "relation":
                structured += 0.10

            # 时间查询
            if temporal_relation in ("before", "after"):
                if mt in ("fact", "event"):
                    structured += 0.05

            # 正负偏好方向
            if predicate_hint:
                if doc_predicate == predicate_hint:
                    structured += 0.15
                elif (
                        predicate_hint == "like"
                        and doc_predicate == "dislike"
                ):
                    structured -= 0.05
                elif (
                        predicate_hint == "dislike"
                        and doc_predicate == "like"
                ):
                    structured -= 0.05

            # 查询意图
            if intent_hint == "habit" and mt == "rule":
                structured += 0.15
            elif intent_hint == "event" and mt == "event":
                structured += 0.10
            elif intent_hint == "relation" and mt == "relation":
                structured += 0.10
            elif intent_hint == "fact" and mt == "fact":
                structured += 0.05

            # 当前有效事实
            if mt == "fact" and r.get("status") == "active":
                structured += 0.03

            structured = max(-0.1, min(0.5, structured))

            # RRF 负责高召回，reranker 更强调“真正回答问题的文本”。
            final = (
                    0.52 * lexical_signal
                    + 0.33 * retrieval_signal
                    + 0.15 * max(0.0, structured)
            )

            item = dict(r)
            item["score"] = round(final, 6)
            item["metadata"] = dict(metadata)
            item["metadata"]["rerank_score"] = round(
                lexical_signal, 6
            )
            item["metadata"]["expanded_rerank_score"] = round(
                expanded_lexical, 6
            )
            item["metadata"]["structured_bonus"] = round(
                structured, 6
            )
            item["metadata"]["retrieval_signal"] = round(
                retrieval_signal, 6
            )

            rescored.append(item)

        rescored.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        if os.getenv("MEMORY_PROFILE", "").strip() == "1":
            logger.info(
                "RERANK_PROFILE input=%d output=%d elapsed=%.2f",
                len(results),
                min(len(rescored), top_k),
                (time.perf_counter() - t0) * 1000,
            )

        return rescored[:top_k]