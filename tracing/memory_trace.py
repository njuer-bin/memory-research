"""Core trace schema for diagnosing where evidence is lost in long-term memory systems."""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceItem:
    evidence_id: str
    text: str = ""
    source: Optional[str] = None
    turn_id: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageTrace:
    found: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryTrace:
    question_id: str
    question: str
    gold_evidence: List[EvidenceItem] = field(default_factory=list)
    formation: StageTrace = field(default_factory=StageTrace)
    storage: StageTrace = field(default_factory=StageTrace)
    evolution: StageTrace = field(default_factory=StageTrace)
    retrieval: StageTrace = field(default_factory=StageTrace)
    rerank: StageTrace = field(default_factory=StageTrace)
    context: StageTrace = field(default_factory=StageTrace)
    answer: Dict[str, Any] = field(default_factory=dict)
    failure_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def gold_ids(self) -> set:
        return {x.evidence_id for x in self.gold_evidence}

    def first_loss_stage(self) -> Optional[str]:
        gold = self.gold_ids()
        if not gold:
            return None
        stages = (("formation", self.formation), ("storage", self.storage),
                  ("evolution", self.evolution), ("retrieval", self.retrieval),
                  ("rerank", self.rerank), ("context", self.context))
        for name, stage in stages:
            if not gold.issubset(set(stage.found)):
                return name
        return None
