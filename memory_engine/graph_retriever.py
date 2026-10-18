from __future__ import annotations


class GraphRetriever:
    def __init__(self, store):
        self.store = store

    def expand(self, user_id: str, seed_results: list[dict], limit=10):
        rels = self.store.relations(user_id)
        if not rels:
            return []

        seeds = set()
        for r in seed_results:
            md = r.get("metadata", {})
            for k in ("subject","object","value"):
                if md.get(k):
                    seeds.add(str(md[k]))
            content = r.get("content","")
            if content:
                seeds.add(content)

        out = []
        for rel in rels:
            if rel["subject"] in seeds or rel["object"] in seeds:
                out.append({
                    "id": rel["id"],
                    "content": rel["content"],
                    "role": "relation",
                    "timestamp": rel["timestamp"],
                    "user_id": user_id,
                    "session_id": "",
                    "score": 0.01,
                    "source": "graph_expansion",
                    "memory_type": "relation",
                    "status": "active",
                    "valid_from": rel["timestamp"],
                    "valid_to": None,
                    "metadata": {
                        "subject": rel["subject"],
                        "predicate": rel["predicate"],
                        "object": rel["object"],
                    },
                })
        return out[:limit]
