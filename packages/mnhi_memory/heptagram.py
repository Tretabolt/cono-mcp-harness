from dataclasses import dataclass
from typing import Sequence
from .metrics import (
    jensen_shannon_divergence_bits,
    jsd_uniforme_para_tirania,
)


@dataclass
class VetoResult:
    vetoed: bool
    jsd_current: float
    jsd_proposed: float
    reason: str


class Heptagram:
    """Atuador Etico e Motor de Regras (J_E): aplica vetos de diversidade e penalidade causal."""

    def __init__(
        self,
        quarantine_floor: float = 0.05,
        penalty_factor: float = 0.5,
        reward_factor: float = 1.10,
        margem_jsd: float = 0.05,
    ) -> None:
        self.quarantine_floor = quarantine_floor
        self.penalty_factor = penalty_factor
        self.reward_factor = reward_factor
        self.margem_jsd = margem_jsd
        self.credits: dict[str, float] = {}
        self.quarantined: set[str] = set()

    def get_credit(self, unit_id: str) -> float:
        return self.credits.get(unit_id, 1.0)

    def is_quarantined(self, unit_id: str) -> bool:
        return unit_id in self.quarantined

    def apply_penalty(self, unit_id: str) -> float:
        """Penaliza peso de confianca/credito causal por evento de falha (0.5^k)."""
        curr = self.get_credit(unit_id)
        updated = curr * self.penalty_factor
        self.credits[unit_id] = updated
        if updated < self.quarantine_floor:
            self.quarantined.add(unit_id)
        return updated

    def apply_reward(self, unit_id: str) -> float:
        if self.is_quarantined(unit_id):
            return self.get_credit(unit_id)
        curr = self.get_credit(unit_id)
        updated = min(1.0, curr * self.reward_factor)
        self.credits[unit_id] = updated
        return updated

    def release_from_quarantine(self, unit_id: str, reset_credit: float = 0.25) -> None:
        self.quarantined.discard(unit_id)
        self.credits[unit_id] = reset_credit

    def evaluate_law_change(
        self,
        p_current: Sequence[float],
        p_proposed: Sequence[float],
        p_tyranny: Sequence[float],
    ) -> VetoResult:
        """Regra de Vetos: barra propostas que aproximam a Noosfera da Tirania Causal.

        Veto relativo (DEFINICOES.md, secao 5): veta se a JSD proposta ficar
        abaixo de JSD(uniforme, Tirania) - margem, ou se regredir (JSD cair
        em direcao a tirania). O piso absoluto antigo (0,15) vetava ate a
        distribuicao uniforme em N=2 (JSD = 0,1468) — por isso foi trocado.
        """
        jsd_curr = jensen_shannon_divergence_bits(p_current, p_tyranny)
        jsd_prop = jensen_shannon_divergence_bits(p_proposed, p_tyranny)
        piso_relativo = jsd_uniforme_para_tirania(len(p_tyranny)) - self.margem_jsd

        if jsd_prop < piso_relativo:
            return VetoResult(
                vetoed=True,
                jsd_current=jsd_curr,
                jsd_proposed=jsd_prop,
                reason=f"JSD proposta ({jsd_prop:.4f}) abaixo do piso relativo ({piso_relativo:.4f}).",
            )
        if jsd_prop < jsd_curr:
            return VetoResult(
                vetoed=True,
                jsd_current=jsd_curr,
                jsd_proposed=jsd_prop,
                reason=f"Regressao de diversidade: JSD caiu de {jsd_curr:.4f} para {jsd_prop:.4f}.",
            )
        return VetoResult(
            vetoed=False,
            jsd_current=jsd_curr,
            jsd_proposed=jsd_prop,
            reason="Diversidade informacional conservada.",
        )
