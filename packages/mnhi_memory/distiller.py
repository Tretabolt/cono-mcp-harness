from typing import Callable
from .graph import CausalGraph, CausalRule
from .pentacosagram import Pentacosagram
from .heptagram import Heptagram


class Distiller:
    """So consolida heuristicas de episodios aprovados no TDD e com cadeia integra."""

    def __init__(self, graph: CausalGraph, auditor: Pentacosagram, heptagram: Heptagram) -> None:
        self.graph = graph
        self.auditor = auditor
        self.heptagram = heptagram

    def commit_if_verified(
        self,
        rule_id: str,
        trigger: str,
        action_plan: list[str],
        steps_used: int,
        tdd_passed: bool,
        regression_suite_fn: Callable[[], bool] | None = None,
    ) -> tuple[bool, str]:
        if not tdd_passed:
            self.heptagram.apply_penalty(rule_id)
            return False, "Recusado: episodio falhou na assercao TDD."

        if not self.auditor.verify_chain():
            return False, "Recusado: cadeia criptografica do Pentacosagram corrompida."

        if any(e.status == "ALUCINACAO_DETECTADA" for e in self.auditor.entries):
            self.heptagram.apply_penalty(rule_id)
            return False, "Recusado: alucinacao causal detectada durante o episodio."

        if regression_suite_fn is not None and not regression_suite_fn():
            return False, "Recusado: regressao detectada nos niveis anteriores."

        rule = CausalRule(
            rule_id=rule_id,
            trigger=trigger,
            action_plan=list(action_plan),
            proven_episode_hash=self.auditor.head_hash,
            min_steps_observed=steps_used,
            confidence=self.heptagram.get_credit(rule_id),
        )
        committed = self.graph.register_rule(rule)
        self.heptagram.apply_reward(rule_id)
        return True, "Heuristica consolidada no Grafo Causal." if committed else "Heuristica ja existente (sem melhoria)."
