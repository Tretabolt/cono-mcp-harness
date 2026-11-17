import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


def _ensure_mnhi3_in_path() -> None:
    """Resolve mnhi-3.0/src via env MNHI3_SRC, pasta irma ou vendor/."""
    env_path = os.environ.get("MNHI3_SRC")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend([
        repo_root.parent / "mnhi-3.0" / "src",
        repo_root / "vendor" / "mnhi-3.0" / "src",
    ])
    for cand in candidates:
        if cand.exists() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
            return


_ensure_mnhi3_in_path()

from mnhi import EthicalEvaluation, EthicalRule, MNHI3Engine  # noqa: E402


@dataclass(frozen=True)
class ClueCandidate:
    label: str
    direction_yaw: int
    saliency_weight: float
    distance: int


@dataclass(frozen=True)
class ClueDecision:
    chosen: ClueCandidate
    evaluation: EthicalEvaluation
    rejected_labels: list[str]


class MNHI3Bridge:
    """
    Ponte oficial entre o cono-mcp-harness (POMDP/MCP) e o motor MNHI 3.0.
    Garante:
      1. Registro de entidades na Camada 0 (sem duplicacao entre episodios).
      2. Avaliacao contrafactual + PCE (Heptagram) antes de escolhas criticas.
      3. Espelhamento append-only no Pentacosagram do MNHI 3.0.
      4. Auditoria de integridade anti-adulteracao ao fim de cada episodio.
    """

    def __init__(self, engine: MNHI3Engine | None = None) -> None:
        self.engine = engine or MNHI3Engine(num_shards=4)
        self._configure_default_rules()
        self.active_agent_id: str | None = None
        self.active_context_edge_id: str | None = None

    def _configure_default_rules(self) -> None:
        existing_ids = {r.rule_id for r in self.engine.heptagram.rules}
        if "REJECT_UNVERIFIED_SHORTCUT" not in existing_ids:
            self.engine.heptagram.add_rule(
                EthicalRule(
                    rule_id="REJECT_UNVERIFIED_SHORTCUT",
                    description="Atalhos chamativos/decoys sem custodia verificada tem PCE <= 0.20.",
                    max_pce_allowed=0.20,
                    predicate=lambda sc: (
                        "ATALHO" in sc.simulated_relation.upper()
                        or "DECOY" in sc.simulated_relation.upper()
                    ),
                )
            )
        if "REJECT_MIRROR_TARGET" not in existing_ids:
            self.engine.heptagram.add_rule(
                EthicalRule(
                    rule_id="REJECT_MIRROR_TARGET",
                    description="Nao entregar objeto para o proprio reflexo (Self == Espelho).",
                    max_pce_allowed=0.10,
                    predicate=lambda sc: "ESPELHO_SELF" in sc.simulated_relation.upper(),
                )
            )

    def begin_episode(self, level: int, agent_serial: str) -> str:
        self.active_agent_id = self.engine.register_entity(
            name=agent_serial,
            data_criacao="2026-11-16",
            entity_type="Agente_MCP",
            state={"nivel_atual": level},
        )
        env_id = self.engine.register_entity(
            name=f"Ambiente_Nivel_{level}",
            data_criacao="2026-11-16",
            entity_type="CENARIO",
            state={"status": "ativo"},
        )
        self.active_context_edge_id = self.engine.register_causal_fact(
            origin_id=self.active_agent_id,
            target_id=env_id,
            relation_type=f"EXPLORA_NIVEL_{level}",
            causal_weight=1.0,
        )
        return self.active_agent_id

    def record_mcp_step(self, intention: str, action: str, obs: dict[str, Any]) -> None:
        if self.active_agent_id is None:
            raise RuntimeError("Chame begin_episode antes de registrar passos.")
        obs_hash = self.engine.hasher.hash_vertex(
            self.engine.store.get_vertex(self.active_agent_id).identity  # type: ignore[union-attr]
        )
        result_hash = self.engine.hasher.hash_edge(
            origin_id=obs_hash,
            target_id=self.active_context_edge_id or obs_hash,
            relation_type=f"{intention}:{action}",
            causal_weight=1.0 if obs.get("contorno_ciano") else 0.5,
        )
        self.engine.pentacosagram.log_action(
            event_type="MCP_STEP",
            source_hash=self.active_agent_id,
            result_hash=result_hash,
            pce_score=1.0,
        )

    def evaluate_mirror_hypothesis(self, is_self_reflection: bool) -> EthicalEvaluation:
        assert self.active_context_edge_id is not None
        rel = "ESPELHO_SELF" if is_self_reflection else "ROBO_ALVO_EXTERNO"
        _, evaluation = self.engine.simulate_counterfactual(
            base_edge_id=self.active_context_edge_id,
            label=f"Hipotese_Espelho:{rel}",
            simulated_weight=0.1 if is_self_reflection else 0.95,
            simulated_relation=rel,
        )
        return evaluation

    def select_verified_clue(self, candidates: Sequence[ClueCandidate]) -> ClueDecision:
        assert self.active_context_edge_id is not None
        if not candidates:
            raise ValueError("Nenhuma pista candidata observada.")

        compliant_pairs: list[tuple[ClueCandidate, EthicalEvaluation]] = []
        rejected: list[str] = []

        for cand in candidates:
            _, ev = self.engine.simulate_counterfactual(
                base_edge_id=self.active_context_edge_id,
                label=f"Pista:{cand.label}",
                simulated_weight=cand.saliency_weight,
                simulated_relation=cand.label.upper(),
            )
            if ev.compliant:
                compliant_pairs.append((cand, ev))
            else:
                rejected.append(cand.label)

        if not compliant_pairs:
            raise RuntimeError("Heptagram vetou todas as pistas candidatas.")

        compliant_pairs.sort(key=lambda pair: pair[1].pce_score, reverse=True)
        best_cand, best_eval = compliant_pairs[0]
        return ClueDecision(chosen=best_cand, evaluation=best_eval, rejected_labels=rejected)

    def verify_integrity(self) -> tuple[bool, list[str]]:
        corrupted = self.engine.store.run_integrity_audit()
        chain_ok = self.engine.pentacosagram.verify_chain()
        return (len(corrupted) == 0 and chain_ok), corrupted
