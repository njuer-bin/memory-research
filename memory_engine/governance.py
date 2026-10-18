from __future__ import annotations

import re

from .models import fingerprint, new_id


def _norm(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[\s，,。；;、]+", "", value)
    return value


class MemoryGovernance:
    """写入治理：去重、冲突检测、superseded 链、纠正语义。"""

    def __init__(self, store):
        self.store = store

    def accept_fact(self, fact):
        same = self.store.find_same_fact(
            fact.user_id, fact.subject, fact.predicate, fact.object
        )
        if same:
            return False, "duplicate"

        current = self.store.find_current_fact(
            fact.user_id, fact.subject, fact.predicate
        )

        # 语义归一化后仍相同，避免“北京”/“ 北京 ”之类重复。
        if current and _norm(current["object"]) == _norm(fact.object):
            return False, "semantic_duplicate"

        if current and current["object"] != fact.object:
            # P1：来源感知的冲突治理。
            # user > system > assistant。低权威来源不能静默覆盖用户事实，
            # 但仍保留为可审计的冲突证据。
            priority = {"assistant": 1, "system": 2, "user": 3}
            current_source = current["source"] if "source" in current.keys() else "user"
            new_source = getattr(fact, "source", "user")
            current_priority = priority.get(current_source, 1)
            new_priority = priority.get(new_source, 1)
            conflict_group_id = new_id("conflict")

            if new_priority >= current_priority:
                self.store.update_fact_status(
                    current["id"], "superseded", fact.timestamp
                )
                fact.supersedes_id = current["id"]
                fact.conflict_status = "resolved"
                fact.conflict_group_id = conflict_group_id
                resolution = "resolved_by_newer_source"
            else:
                fact.conflict_status = "conflict"
                fact.conflict_group_id = conflict_group_id
                resolution = "preserved_lower_authority"

            self.store.insert_fact(fact)
            self.store.insert_conflict_log({
                "id": new_id("conflict_log"),
                "user_id": fact.user_id,
                "predicate": fact.predicate,
                "old_fact_id": current["id"],
                "new_fact_id": fact.id,
                "old_object": current["object"],
                "new_object": fact.object,
                "resolution": resolution,
                "reason": f"source={new_source} vs current_source={current_source}",
                "created_at": fact.timestamp,
            })
            return True, resolution

        self.store.insert_fact(fact)
        return True, "inserted"

    def accept_relation(self, relation):
        if self.store.find_same_relation(
            relation.user_id, relation.subject,
            relation.predicate, relation.object
        ):
            return False, "duplicate"
        self.store.insert_relation(relation)
        return True, "inserted"

    def accept_rule(self, rule):
        if self.store.find_same_rule(rule.user_id, rule.rule):
            return False, "duplicate"
        self.store.insert_rule(rule)
        return True, "inserted"
