from __future__ import annotations

from .engine import MemoryEngine
from .reasoning import MemoryReasoner


class ReasoningMemoryEngine(MemoryEngine):
    """P7 production wrapper around the stable P6 engine.

    P6 remains the source of truth for retrieval.  This wrapper adds a small
    deterministic reasoning pass after retrieval so the AML endpoint can
    benefit from state, temporal, and causal evidence without replacing the
    proven P6 retrieval pipeline.
    """

    def search(self, request):
        rows = super().search(request)
        query = (request.query or request.question or "").strip()

        # Rebuild the lightweight plan only for the reasoning pass.  P6's
        # internal plan remains untouched, keeping this change isolated.
        latest = self.store.all_raw(request.user_id)
        reference_ts = latest[0]["timestamp"] if latest else None
        plan = self.query_analyzer.analyze(query, request.multi_hop, reference_ts)

        rows = MemoryReasoner.augment(
            query=query,
            plan=plan,
            ranked=rows,
            store=self.store,
            user_id=request.user_id,
        )

        # Re-apply the evidence-chain annotation after causal edges are added.
        # This is intentionally bounded to the returned P6 candidates plus a
        # small reasoning expansion; it does not re-run retrieval.
        if plan.multi_hop and len(rows) > 1:
            rows = self.evidence_chain.annotate(plan.rewritten, rows)

        return rows[: request.top_k]
