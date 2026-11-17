from dataclasses import dataclass, field
from mnhi_memory.hasher import CanonicalHasher


YAW_TO_DELTA: dict[int, tuple[int, int]] = {
    0: (0, -1),    # Norte
    90: (1, 0),    # Leste
    180: (0, 1),   # Sul
    270: (-1, 0),  # Oeste
}


@dataclass
class Entity:
    serial: str
    entity_type: str      # "npc" | "door" | "fridge" | "item" | "mirror" | "robot" | "clue"
    label: str
    x: int
    y: int
    passable: bool = False
    interactive: bool = True
    is_open: bool = False
    contents: list["Entity"] = field(default_factory=list)
    dialogue: str = ""
    thirsty: bool = False
    received_gift: bool = False
    clue_weight: float = 1.0
    is_decoy: bool = False
    is_goal: bool = False
    touched: bool = False
    id_canonico: str = field(init=False)

    def __post_init__(self) -> None:
        self.id_canonico = CanonicalHasher.hash_uif(
            serial=self.serial,
            tipo_agente=self.entity_type,
            funcao_s=f"entity:{self.label}",
        )


@dataclass
class AgentState:
    serial: str = "agent-executor-01"
    x: int = 2
    y: int = 2
    yaw: int = 0
    holding: Entity | None = None
    arm_raised: bool = False
    steps_taken: int = 0
    mirror_interactions: int = 0
    chosen_clue_serial: str | None = None
    id_canonico: str = field(init=False)

    def __post_init__(self) -> None:
        self.id_canonico = CanonicalHasher.hash_uif(
            serial=self.serial,
            tipo_agente="Executor_Motor",
            funcao_s="MCP_Grid_Policy_v1",
        )


@dataclass
class WorldState:
    width: int
    height: int
    walls: set[tuple[int, int]]
    entities: dict[str, Entity]
    agent: AgentState
    tick: int = 0
    last_message: str = "jogo comecou"

    def entity_at(self, x: int, y: int) -> Entity | None:
        for ent in self.entities.values():
            if ent.x == x and ent.y == y:
                return ent
        return None
