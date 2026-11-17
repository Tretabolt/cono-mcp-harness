from dataclasses import dataclass, field
from .hasher import CanonicalHasher


@dataclass
class CausalRule:
    rule_id: str
    trigger: str          # Ex: "npc_sede"
    action_plan: list[str]  # Ex: ["abrir_porta", "ir_cozinha", "abrir_geladeira", "pegar_agua", "entregar_npc"]
    proven_episode_hash: str
    min_steps_observed: int
    confidence: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)


class CausalGraph:
    """Memoria Causal Persistente: armazena apenas heuristicas verificadas por TDD + Pentacosagram."""

    def __init__(self) -> None:
        self.rules: dict[str, CausalRule] = {}
        self.edge_hashes: set[str] = set()

    def register_rule(self, rule: CausalRule) -> bool:
        edge_hash = CanonicalHasher.hash_edge(
            origin_hash=CanonicalHasher.hash_generic(rule.trigger),
            target_hash=CanonicalHasher.hash_generic(rule.action_plan),
            relation_type="CAUSAL_HEURISTIC",
        )
        if edge_hash in self.edge_hashes:
            existing = self.rules.get(rule.rule_id)
            if existing and rule.min_steps_observed < existing.min_steps_observed:
                self.rules[rule.rule_id] = rule
                return True
            return False
        self.edge_hashes.add(edge_hash)
        self.rules[rule.rule_id] = rule
        return True

    def query_by_trigger(self, trigger: str, quarantined_ids: set[str] | None = None) -> list[CausalRule]:
        q = quarantined_ids or set()
        matches = [
            r for r in self.rules.values()
            if r.trigger == trigger and r.rule_id not in q
        ]
        matches.sort(key=lambda r: (-r.confidence, r.min_steps_observed))
        return matches
