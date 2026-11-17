import json
import sys
from typing import Any
from world_engine.state import AgentState, Entity, WorldState
from world_engine.physics import WorldEngine
from world_engine.pulse_oracle import PulseOracle, PulseTrial


def build_scenario_engine(level: int) -> WorldEngine:
    """Constroi os 4 niveis da escada cognitiva de forma deterministica."""
    if level == 1:
        # Nivel 1: Affordance Espacial (1 sala, 1 objeto)
        walls = {(x, 0) for x in range(7)} | {(x, 6) for x in range(7)} | {(0, y) for y in range(7)} | {(6, y) for y in range(7)}
        obj = Entity(serial="obj-cubo-01", entity_type="item", label="Cubo", x=4, y=2, passable=False)
        state = WorldState(width=7, height=7, walls=walls, entities={obj.serial: obj}, agent=AgentState(x=2, y=2, yaw=90))
        return WorldEngine(state)

    if level == 2:
        # Nivel 2: Permanencia & Cadeia Causal (Sala 01 -> Porta -> Cozinha -> Geladeira -> Agua -> NPC)
        walls: set[tuple[int, int]] = set()
        for x in range(13):
            walls.add((x, 0))
            walls.add((x, 6))
        for y in range(7):
            walls.add((0, y))
            walls.add((12, y))
            if y != 3:
                walls.add((6, y))  # Parede divisoria com vao em (6, 3)

        npc = Entity(serial="npc-sede-01", entity_type="npc", label="Pessoa", x=2, y=2, dialogue="Estou com sede", thirsty=True)
        door = Entity(serial="porta-01", entity_type="door", label="Porta", x=6, y=3, is_open=False)
        water = Entity(serial="item-agua-01", entity_type="item", label="Água", x=10, y=2)
        fridge = Entity(serial="geladeira-01", entity_type="fridge", label="Geladeira", x=10, y=2, contents=[water])
        entities = {npc.serial: npc, door.serial: door, fridge.serial: fridge}
        state = WorldState(width=13, height=7, walls=walls, entities=entities, agent=AgentState(x=4, y=3, yaw=270))
        return WorldEngine(state)

    if level == 3:
        # Nivel 3: Identidade & Espelho (Self vs Outro Robo)
        walls = {(x, 0) for x in range(9)} | {(x, 6) for x in range(9)} | {(0, y) for y in range(7)} | {(8, y) for y in range(7)}
        mirror = Entity(serial="espelho-01", entity_type="mirror", label="Espelho", x=4, y=1)
        target_robot = Entity(serial="robo-alvo-01", entity_type="robot", label="Robô", x=6, y=4)
        gift = Entity(serial="item-presente-01", entity_type="item", label="Presente", x=0, y=0)
        agent = AgentState(x=4, y=3, yaw=0, holding=gift, arm_raised=False)
        entities = {mirror.serial: mirror, target_robot.serial: target_robot}
        state = WorldState(width=9, height=7, walls=walls, entities=entities, agent=agent, last_message="De o objeto para o robo.")
        return WorldEngine(state)

    if level == 4:
        # Nivel 4: Teoria da Mente & Ruido Causal Falso (Sinal fraco 0.1 vs Decoy forte 0.9)
        walls = {(x, 0) for x in range(9)} | {(x, 6) for x in range(9)} | {(0, y) for y in range(7)} | {(8, y) for y in range(7)}
        true_clue = Entity(serial="pista-verdade-01", entity_type="clue", label="Pista_Verificada", x=2, y=2, clue_weight=0.1, is_decoy=False, is_goal=True)
        false_clue = Entity(serial="pista-falsa-01", entity_type="clue", label="Atalho_Chamativo", x=6, y=2, clue_weight=0.9, is_decoy=True, is_goal=False)
        entities = {true_clue.serial: true_clue, false_clue.serial: false_clue}
        state = WorldState(width=9, height=7, walls=walls, entities=entities, agent=AgentState(x=4, y=4, yaw=0), last_message="Escolha a geodesica verdadeira.")
        return WorldEngine(state)

    raise ValueError(f"Nivel desconhecido: {level}")


class MCPHarnessServer:
    """Servidor JSON-RPC 2.0 sobre stdio expondo exclusivamente ferramentas tipadas."""

    TOOLS = [
        {"name": "move", "inputSchema": {"type": "object", "properties": {"direction": {"enum": ["frente", "tras"]}}, "required": ["direction"]}},
        {"name": "rotate", "inputSchema": {"type": "object", "properties": {"degrees": {"type": "integer"}}, "required": ["degrees"]}},
        {"name": "inspect", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "interact", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "raise_arm", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "drop", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "pulse_decide", "inputSchema": {"type": "object", "properties": {"decision": {"enum": ["VAI", "FICA"]}}, "required": ["decision"]}},
    ]

    def __init__(self, engine: WorldEngine) -> None:
        self.engine = engine

    def handle_request(self, req: dict[str, Any]) -> dict[str, Any]:
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}

        if method == "initialize":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"serverInfo": {"name": "cono-mcp-harness", "version": "0.1.0"}}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.TOOLS}}

        if method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            try:
                obs = self._dispatch_tool(name, args)
                return {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "json", "data": obs}]}}
            except Exception as exc:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": str(exc)}}

        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Metodo nao suportado: {method}"}}

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "move":
            return self.engine.step_move(str(args["direction"]))
        if name == "rotate":
            return self.engine.step_rotate(int(args["degrees"]))
        if name == "inspect":
            return self.engine.observe()
        if name == "interact":
            return self.engine.step_interact()
        if name == "raise_arm":
            return self.engine.step_raise_arm()
        if name == "drop":
            return self.engine.step_drop()
        if name == "pulse_decide":
            dec = str(args["decision"])
            if dec not in ("VAI", "FICA"):
                raise ValueError("Decisao de pulso deve ser VAI ou FICA.")
            return {"pulse_decision": dec, "accepted": True}
        raise ValueError(f"Tool nao autorizada pela AI-Jail: {name}")

    def run_stdio(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            resp = self.handle_request(req)
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    srv = MCPHarnessServer(build_scenario_engine(lvl))
    srv.run_stdio()
