from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PulseTrial:
    trial_id: str
    stimulus_vector: tuple[float, ...]
    expected_decision: str  # "VAI" | "FICA"
    ideal_state_vec: tuple[float, float]
    max_latency_ticks: int = 10


class PulseOracle:
    """Verificador TDD para tarefa VAI/FICA (Go/No-Go) em redes de pulsos ou organoides."""

    @staticmethod
    def verify_trial(
        trial: PulseTrial,
        decision: str,
        latency_ticks: int,
        delta_spikes: Sequence[float],
    ) -> tuple[bool, str]:
        if decision not in ("VAI", "FICA"):
            return False, f"Decisao invalida: {decision}"
        if latency_ticks > trial.max_latency_ticks:
            return False, f"Latencia excedida: {latency_ticks} > {trial.max_latency_ticks}"
        if decision != trial.expected_decision:
            return False, f"Decisao incorreta: obteve {decision}, esperado {trial.expected_decision}"
        if not delta_spikes:
            return False, "Trajetoria de pulsos vazia."
        return True, "Ensaio VAI/FICA aprovado."
