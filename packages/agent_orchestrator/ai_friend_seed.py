from dataclasses import dataclass
from mnhi_memory.hasher import CanonicalHasher


@dataclass(frozen=True)
class AIFriendIdentity:
    """Identidade apolinea imutavel (Camada 0). Estados emocionais/sessao ficam fora do hash."""
    serial: str
    tipo_agente: str
    funcao_s: str
    hash_uif: str


def build_ai_friend_seed() -> AIFriendIdentity:
    serial = "persona-ai-friend-001"
    tipo_agente = "Agente_Cognitivo_Sincrasia"
    funcao_s = "Politica_Base:Empatica_Curiosa_Auditavel_Sem_Alucinacao"
    h = CanonicalHasher.hash_uif(serial=serial, tipo_agente=tipo_agente, funcao_s=funcao_s)
    return AIFriendIdentity(serial=serial, tipo_agente=tipo_agente, funcao_s=funcao_s, hash_uif=h)
