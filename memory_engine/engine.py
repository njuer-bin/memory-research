from __future__ import annotations

import logging
import os
import time
from typing import Any

from .analyzer import MemoryAnalyzer
from .evidence import EvidenceBuilder
from .governance import MemoryGovernance
from .graph_retriever import GraphRetriever
from .hybrid_retriever import HybridRetriever
from .models import new_id, now_ms
from .query_analyzer import QueryAnalyzer
from .reranker import LightweightReranker
from .store import SQLiteStore
from .vector import EmbeddingProvider
from .vector_index import MemoryVectorIndex


logger = logging.getLogger(__name__)


class MemoryEngine:
    def __init__(self, db_path="data/memory.db"):
        self.store = SQLiteStore(db_path)
        self.analyzer = MemoryAnalyzer()
        self.governance = MemoryGovernance(self.store)
        self.embedder = EmbeddingProvider()
        # SQLite 持久化 + 进程级向量索引：Search 热路径不再反复读取/解析 JSON 向量。
        self.vector_index = MemoryVectorIndex(self.store)
        self.hybrid = HybridRetriever(self.store, self.embedder, self.vector_index)
        self.graph = GraphRetriever(self.store)
        self.query_analyzer = QueryAnalyzer()
        self.reranker = LightweightReranker()
        self.evidence = EvidenceBuilder()
        self.last_search_trace: dict[str, Any] = {}

    def add(self, request):
        # Claim request_id atomically before doing any writes. This closes the
        # check-then-act race between concurrent duplicate Add requests.
        if not self.store.claim_request(request.request_id, request.user_id):
            return True

        try:
            return self._add_claimed(request)
        except Exception:
            # Do not permanently consume a request_id when the Add operation
            # fails before completion; the caller can retry safely.
            self.store.release_request(request.request_id)
            raise

    def _add_claimed(self, request):
        # 按 20 条消息或约 2000 词做确定性批次边界。
        batches = []
        current = []
        words = 0
        for msg in request.messages:
            n = len(msg.content.split())
            if current and (len(current) >= 20 or words + n > 2000):
                batches.append(current)
                current, words = [], 0
            current.append(msg)
            words += n
        if current:
            batches.append(current)

        turn_counter = 0
        for batch_idx, batch in enumerate(batches):
            for msg_idx, msg in enumerate(batch):
                ts = msg.timestamp or now_ms()
                raw_id = new_id("raw")
                self.store.insert_raw({
                    "id": raw_id,
                    "request_id": request.request_id,
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": ts,
                    "chunk_index": batch_idx,
                    "source_turn_id": f"{request.session_id}:turn_{turn_counter}",
                })

                # 原始记忆同时写入 SQLite 和进程级向量索引，保证 Add -> Search 立即可见。
                raw_vector = self.embedder.embed(msg.content)
                self.store.embed(raw_id, request.user_id, raw_vector)
                self.vector_index.add(request.user_id, raw_id, raw_vector)

                role = (msg.role or "user").strip().lower()
                source = "system" if role == "system" else ("assistant" if role in {"assistant", "model"} else "user")
                analyzed = self.analyzer.analyze(
                    request.user_id, msg.content, ts, source=source,
                    source_raw_id=raw_id,
                    source_session_id=request.session_id,
                    source_turn_id=f"{request.session_id}:turn_{turn_counter}",
                )

                for fact in analyzed["facts"]:
                    inserted, _ = self.governance.accept_fact(fact)
                    if inserted:
                        fact_vector = self.embedder.embed(fact.content)
                        self.store.embed(fact.id, request.user_id, fact_vector)
                        self.vector_index.add(request.user_id, fact.id, fact_vector)

                for rel in analyzed["relations"]:
                    self.store.insert_relation(rel)
                    rel_vector = self.embedder.embed(rel.content)
                    self.store.embed(rel.id, request.user_id, rel_vector)
                    self.vector_index.add(request.user_id, rel.id, rel_vector)

                for event in analyzed["events"]:
                    self.store.insert_event(event)
                    event_vector = self.embedder.embed(event.content)
                    self.store.embed(event.id, request.user_id, event_vector)
                    self.vector_index.add(request.user_id, event.id, event_vector)

                for rule in analyzed["rules"]:
                    self.store.insert_rule(rule)
                    rule_vector = self.embedder.embed(rule.content)
                    self.store.embed(rule.id, request.user_id, rule_vector)
                    self.vector_index.add(request.user_id, rule.id, rule_vector)

                for profile in analyzed["profiles"]:
                    self.store.upsert_profile(profile)
                turn_counter += 1

        # request_id was already atomically claimed before processing.
        return True

    def search(self, request):
        search_t0 = time.perf_counter()
        query = (request.query or request.question or "").strip()
        if not query:
            return []

        # 用用户已有最新记忆作为相对时间参考，避免服务当前时间与 benchmark 时间轴不一致。
        t0 = time.perf_counter()
        latest = self.store.all_raw(request.user_id)
        latest_ms = (time.perf_counter() - t0) * 1000
        reference_ts = latest[0]["timestamp"] if latest else now_ms()

        t0 = time.perf_counter()
        plan = self.query_analyzer.analyze(query, request.multi_hop, reference_ts)
        analyze_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        candidates = self.hybrid.candidates(
            user_id=request.user_id,
            query=plan.rewritten,
            top_k=max(30, request.top_k * 5),
            include_history=request.include_history or ("历史" in query or "以前" in query or "之前" in query),
            session_id=request.session_id,
            start_time=request.start_time,
            end_time=request.end_time,
            memory_types=request.memory_types,
            memory_type_hint=plan.memory_type_hint,
            temporal_relation=plan.temporal_relation,
            relation_hint=plan.relation_hint,
            sparse_query=plan.expanded_query or plan.rewritten,
            predicate_hint=plan.predicate_hint,
            intent_hint=plan.intent_hint,
        )
        hybrid_ms = (time.perf_counter() - t0) * 1000

        result = []
        for d, score in candidates:
            item = dict(d)
            item["score"] = score
            result.append(item)
        # 查询级时间约束：优先使用显式时间窗口；“以前/去年/上个月”等
        # 会由 QueryAnalyzer 归一化后应用到候选证据。
        if plan.temporal and (plan.temporal_start is not None or plan.temporal_end is not None):
            result = [
                r for r in result
                if (plan.temporal_start is None or r.get("valid_from", r.get("timestamp", 0)) >= plan.temporal_start)
                and (plan.temporal_end is None or r.get("valid_from", r.get("timestamp", 0)) <= plan.temporal_end)
            ]

        # P1：受控两轮检索。仅对多跳查询启用：第一轮先找锚点实体，
        # 第二轮用锚点做 BM25-only 扩展，避免再次调用 embedding。
        second_round_ms = 0.0
        if plan.multi_hop and result:
            entity_terms = []
            for seed in result[:5]:
                md = seed.get("metadata", {}) or {}
                for key in ("subject", "object", "value"):
                    value = md.get(key)
                    if value and value != "user" and value not in entity_terms:
                        entity_terms.append(str(value))
            if entity_terms:
                round2_query = plan.rewritten + " " + " ".join(entity_terms[:4])
                t0 = time.perf_counter()
                round2 = self.hybrid.candidates(
                    user_id=request.user_id,
                    query=round2_query,
                    top_k=max(15, request.top_k * 2),
                    include_history=request.include_history or ("历史" in query or "以前" in query or "之前" in query),
                    session_id=request.session_id,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    memory_types=request.memory_types,
                    memory_type_hint=plan.memory_type_hint,
                    temporal_relation=plan.temporal_relation,
                    relation_hint=plan.relation_hint,
                    sparse_query=round2_query,
                    predicate_hint=plan.predicate_hint,
                    intent_hint=plan.intent_hint,
                    use_dense=False,
                )
                result.extend({**dict(d), "score": score} for d, score in round2)
                second_round_ms = (time.perf_counter() - t0) * 1000

        # 仅多跳查询执行额外图扩展，避免所有查询都增加延迟。
        graph_ms = 0.0
        if plan.multi_hop:
            t0 = time.perf_counter()
            expanded = self.graph.expand(request.user_id, result[:5], limit=10)
            result.extend(expanded)
            graph_ms = (time.perf_counter() - t0) * 1000

        # 去重
        dedup = {}
        for r in result:
            key = (r["content"].strip(), r.get("memory_type"))
            if key not in dedup or r["score"] > dedup[key]["score"]:
                dedup[key] = r

        ranked = list(dedup.values())
        retrieval_candidates = list(ranked)
        t0 = time.perf_counter()
        pre_rerank = list(ranked)
        ranked = self.reranker.rerank(
            plan.rewritten,
            ranked,
            max(request.top_k * 3, request.top_k),
            memory_type_hint=plan.memory_type_hint,
            relation_hint=plan.relation_hint,
            temporal_relation=plan.temporal_relation,
            expanded_query=plan.expanded_query or plan.rewritten,
            predicate_hint=plan.predicate_hint,
            intent_hint=plan.intent_hint,
        )
        rerank_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        reranked = list(ranked)
        ranked = self.evidence.build(ranked, request.top_k)
        context_items = list(ranked)
        evidence_ms = (time.perf_counter() - t0) * 1000

        # 时间查询的结果顺序：当前有效事实优先；历史查询保留时间信息。
        if not request.include_history and plan.temporal:
            ranked.sort(
                key=lambda x: (
                    1 if x.get("status") == "active" else 0,
                    x.get("timestamp", 0),
                    x.get("score", 0.0),
                ),
                reverse=True,
            )

        total_ms = (time.perf_counter() - search_t0) * 1000
        self.last_search_trace = {
            "query": query,
            "retrieval_candidates": retrieval_candidates,
            "pre_rerank_candidates": pre_rerank,
            "reranked": reranked,
            "context": context_items,
            "timings_ms": {
                "latest": round(latest_ms, 3), "analyze": round(analyze_ms, 3),
                "hybrid": round(hybrid_ms, 3), "second_round": round(second_round_ms, 3),
                "graph": round(graph_ms, 3), "rerank": round(rerank_ms, 3),
                "evidence": round(evidence_ms, 3), "total": round(total_ms, 3),
            },
        }

        if os.getenv("MEMORY_PROFILE", "").strip() == "1":
            logger.info(
                "SEARCH_PROFILE query=%r total=%.2f latest=%.2f analyze=%.2f hybrid=%.2f "
                "second_round=%.2f graph=%.2f rerank=%.2f evidence=%.2f candidates=%d final=%d",
                query, total_ms, latest_ms, analyze_ms, hybrid_ms, second_round_ms, graph_ms,
                rerank_ms, evidence_ms, len(candidates), len(ranked),
            )
        return ranked[:request.top_k]
