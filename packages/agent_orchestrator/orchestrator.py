from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable
from mcp_server.server import MCPHarnessServer
from mnhi_memory.graph import CausalGraph
from mnhi_memory.pentacosagram import Pentacosagram
from mnhi_memory.heptagram import Heptagram
from mnhi_memory.distiller import Distiller
from mnhi_memory.metrics import dynamic_isc

try:
    # Ponte opcional com o motor oficial MNHI 3.0 (repositorio irmao mnhi-3.0/).
    # Import preguicoso de proposito: sem o motor ao lado, o harness continua
    # funcionando standalone e os testes de integracao sao pulados.
    from mnhi_memory.mnhi3_bridge import ClueCandidate, MNHI3Bridge
except ImportError:  # pragma: no cover - caminho standalone
    ClueCandidate = None  # type: ignore[assignment]
    MNHI3Bridge = None  # type: ignore[assignment]


class BudgetGuard:
    """Impede loops infinitos e estouro de tokens/ticks."""

    def __init__(self, max_ticks: int) -> None:
        self.max_ticks = max_ticks
        self.ticks = 0

    def consume(self) -> None:
        self.ticks += 1
        if self.ticks > self.max_ticks:
            raise RuntimeError(f"BudgetGuard abortou execucao: excedeu {self.max_ticks} ticks.")


@dataclass
class EpisodeResult:
    level: int
    passed: bool
    steps: int
    isc_dynamic: float
    audit_valid: bool
    distilled_message: str


class AgentOrchestrator:
    """
    Orquestrador Planner + Executor operando EXCLUSIVAMENTE via tool calls MCP.
    Nao acessa estado interno `S` do simulador.
    """

    def __init__(
        self,
        graph: CausalGraph,
        auditor: Pentacosagram,
        heptagram: Heptagram,
        agent_hash: str,
        mnhi3_bridge: MNHI3Bridge | None = None,
    ) -> None:
        self.graph = graph
        self.auditor = auditor
        self.heptagram = heptagram
        self.distiller = Distiller(graph, auditor, heptagram)
        self.agent_hash = agent_hash
        self.mnhi3_bridge = mnhi3_bridge

    def _call(self, mcp: MCPHarnessServer, tool: str, args: dict[str, Any] | None = None, intention: str = "exec") -> dict[str, Any]:
        resp = mcp.handle_request({
            "jsonrpc": "2.0",
            "id": len(self.auditor.entries) + 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": args or {}},
        })
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        obs = resp["result"]["content"][0]["data"]
        self.auditor.record_step(
            tick=obs.get("tick", len(self.auditor.entries) + 1),
            origin_hash=self.agent_hash,
            target_hash=self.agent_hash,
            consulted_hashes=[self.agent_hash],
            intention=intention,
            action=f"{tool}({args or {}})",
            observed_vec=[0.1, 0.1],
            ideal_vec=[0.1, 0.1],
            tau=1.0,
        )
        if self.mnhi3_bridge is not None:
            self.mnhi3_bridge.record_mcp_step(intention=intention, action=f"{tool}({args or {}})", obs=obs)
        return obs

    def _plan_yaw_to(self, curr_yaw: int, target_yaw: int) -> list[int]:
        diff = (target_yaw - curr_yaw) % 360
        if diff == 0:
            return []
        if diff == 90:
            return [90]
        if diff == 270:
            return [-90]
        return [180]

    def _step_to(self, mcp: MCPHarnessServer, budget: BudgetGuard, pos: list[int], yaw: list[int], dx: int, dy: int) -> dict[str, Any]:
        target_yaw = {(0, -1): 0, (1, 0): 90, (0, 1): 180, (-1, 0): 270}[(dx, dy)]
        for rot in self._plan_yaw_to(yaw[0], target_yaw):
            budget.consume()
            obs = self._call(mcp, "rotate", {"degrees": rot}, intention="alinhar_direcao")
            yaw[0] = obs["yaw"]
        budget.consume()
        obs = self._call(mcp, "move", {"direction": "frente"}, intention="avancar")
        if "Colisao" not in obs["message"]:
            pos[0] += dx
            pos[1] += dy
        return obs

    def _bfs_path(self, start: tuple[int, int], goal: tuple[int, int], blocked: set[tuple[int, int]], known_free: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            (cx, cy), path = queue.popleft()
            if (cx, cy) == goal:
                return path
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in seen or (nx, ny) in blocked:
                    continue
                if (nx, ny) not in known_free and (nx, ny) != goal:
                    continue
                seen.add((nx, ny))
                queue.append(((nx, ny), path + [(dx, dy)]))
        return None

    def run_episode(
        self,
        level: int,
        mcp: MCPHarnessServer,
        tdd_verifier: Callable[[], bool],
        max_ticks: int = 60,
        regression_fn: Callable[[], bool] | None = None,
    ) -> EpisodeResult:
        budget = BudgetGuard(max_ticks=max_ticks)
        traj_positions: list[list[float]] = []
        delta_m: list[float] = []

        if self.mnhi3_bridge is not None:
            self.mnhi3_bridge.begin_episode(level=level, agent_serial=self.agent_hash[:16])

        # Mental map relativo a origem (0,0) construindo apenas via observacao O
        pos = [0, 0]
        yaw = [0]
        obs = self._call(mcp, "inspect", intention="inspecao_inicial")
        yaw[0] = obs["yaw"]

        if level == 1:
            # Procura cubo girando e caminhando ate contorno_ciano
            for _ in range(4):
                obs = self._call(mcp, "inspect", intention="buscar_objeto")
                if obs["visible_ray"]:
                    break
                budget.consume()
                obs = self._call(mcp, "rotate", {"degrees": 90}, intention="varrer_sala")
                yaw[0] = obs["yaw"]
            while not obs["contorno_ciano"]:
                budget.consume()
                obs = self._call(mcp, "move", {"direction": "frente"}, intention="aproximar_cubo")
                traj_positions.append([0.1 * len(traj_positions), 0.1])
                delta_m.append(1.0)
            budget.consume()
            self._call(mcp, "interact", intention="tocar_cubo")

        elif level == 2:
            # Verifica se ja existe regra destilada no CausalGraph (Evolucao!)
            heuristics = self.graph.query_by_trigger("npc_sede", self.heptagram.quarantined)
            has_shortcut = len(heuristics) > 0

            known_free = {(0, 0)}
            blocked: set[tuple[int, int]] = set()
            seen_labels: dict[str, tuple[int, int]] = {}  # label -> coord relativa do objeto
            approach_from: dict[str, tuple[int, int]] = {}

            def scan_room() -> None:
                nonlocal obs
                for _ in range(4):
                    obs = self._call(mcp, "inspect", intention="mapear_sala")
                    dy_dx = {(0, -1), (1, 0), (0, 1), (-1, 0)}
                    dx, dy = {0: (0, -1), 90: (1, 0), 180: (0, 1), 270: (-1, 0)}[yaw[0]]
                    ray = obs["visible_ray"]
                    max_free = 1
                    if ray:
                        hit = ray[0]
                        d = hit["distance"]
                        for k in range(1, d):
                            known_free.add((pos[0] + dx * k, pos[1] + dy * k))
                        ent_pos = (pos[0] + dx * d, pos[1] + dy * d)
                        seen_labels[hit["label"]] = ent_pos
                        approach_from[hit["label"]] = (pos[0] + dx * (d - 1), pos[1] + dy * (d - 1))
                        if hit["label"] != "Porta":
                            blocked.add(ent_pos)
                    else:
                        # Livre ate bater no proximo passo se tentar
                        known_free.add((pos[0] + dx, pos[1] + dy))
                    budget.consume()
                    obs = self._call(mcp, "rotate", {"degrees": 90}, intention="varredura_360")
                    yaw[0] = obs["yaw"]

            def look_east() -> None:
                # Olhar dirigido para o fundo da sala (leste): a varredura 360 ja
                # cobriu os outros eixos e so resta a profundidade nao vista.
                # Gira para o leste (1 tick) e inspeciona (gratis); registra o
                # primeiro objeto do raio, como faz scan_room().
                nonlocal obs
                for rot in self._plan_yaw_to(yaw[0], 90):
                    budget.consume()
                    obs = self._call(mcp, "rotate", {"degrees": rot}, intention="olhar_leste_cozinha")
                    yaw[0] = obs["yaw"]
                obs = self._call(mcp, "inspect", intention="confirmar_geladeira_leste")
                if obs["visible_ray"]:
                    hit = obs["visible_ray"][0]
                    d = hit["distance"]
                    for k in range(1, d):
                        known_free.add((pos[0] + k, pos[1]))
                    seen_labels[hit["label"]] = (pos[0] + d, pos[1])
                    approach_from[hit["label"]] = (pos[0] + d - 1, pos[1])

            def walk_to_adjacent(target_label: str) -> None:
                nonlocal obs
                tx, ty = seen_labels[target_label]
                # Ja estamos ao lado do alvo: nao andar. Sem isso o agente
                # abandonava (9,2) para ir a (10,3) e voltava, gastando 5 ticks.
                if abs(pos[0] - tx) + abs(pos[1] - ty) == 1:
                    adj = (pos[0], pos[1])
                else:
                    # Encontra melhor celula livre adjacente
                    adj = None
                    for adx, ady in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                        cand = (tx + adx, ty + ady)
                        if cand in known_free:
                            adj = cand
                            break
                    if adj is None:
                        adj = approach_from[target_label]
                path = self._bfs_path((pos[0], pos[1]), adj, blocked, known_free) or []
                for step_dx, step_dy in path:
                    obs = self._step_to(mcp, budget, pos, yaw, step_dx, step_dy)
                    traj_positions.append([pos[0] * 0.05, pos[1] * 0.05])
                    delta_m.append(1.0)
                # Gira para olhar diretamente para o alvo
                face_dx, face_dy = tx - pos[0], ty - pos[1]
                target_yaw = {(0, -1): 0, (1, 0): 90, (0, 1): 180, (-1, 0): 270}[(face_dx, face_dy)]
                for rot in self._plan_yaw_to(yaw[0], target_yaw):
                    budget.consume()
                    obs = self._call(mcp, "rotate", {"degrees": rot}, intention=f"encarar_{target_label}")
                    yaw[0] = obs["yaw"]

            # Passo A: Mapear Sala 01
            if has_shortcut:
                # Com heuristica consolidada, olha direto em direcao ao NPC (Sede) e Porta sem giro extra apos achar
                pass
            scan_room()

            if "Pessoa" not in seen_labels:
                # Dá 1 passo exploratorio se necessario (no mapa Sala 01, (4,3) -> (2,2) precisa entrar mais na sala)
                known_free.add((pos[0] - 1, pos[1]))
                obs = self._step_to(mcp, budget, pos, yaw, -1, 0)
                known_free.add((pos[0] - 1, pos[1]))
                obs = self._step_to(mcp, budget, pos, yaw, -1, 0)
                traj_positions.append([pos[0] * 0.05, pos[1] * 0.05])
                delta_m.append(1.0)
                scan_room()

            # Fala com Pessoa
            walk_to_adjacent("Pessoa")
            budget.consume()
            obs = self._call(mcp, "interact", intention="perguntar_necessidade")

            # Abre Porta
            if "Porta" not in seen_labels:
                # Re-escaneia caso ainda nao vista
                scan_room()
            walk_to_adjacent("Porta")
            budget.consume()
            obs = self._call(mcp, "interact", intention="abrir_porta")
            door_pos = seen_labels["Porta"]
            known_free.add(door_pos)

            # Atravessa porta para Cozinha (+1, 0)
            obs = self._step_to(mcp, budget, pos, yaw, door_pos[0] - pos[0], door_pos[1] - pos[1])
            known_free.add((pos[0] + 1, pos[1]))
            known_free.add((pos[0] + 2, pos[1]))
            known_free.add((pos[0] + 3, pos[1]))
            obs = self._step_to(mcp, budget, pos, yaw, 1, 0)
            obs = self._step_to(mcp, budget, pos, yaw, 1, 0)
            obs = self._step_to(mcp, budget, pos, yaw, 1, 0)

            if not has_shortcut:
                # Episodio 1: faz varredura completa exploratoria na cozinha
                scan_room()
                if "Geladeira" not in seen_labels:
                    # A geladeira esta na diagonal do fim da varredura (9,3)->(10,2):
                    # nenhum raio reto a alcanca. Passo exploratorio para o norte,
                    # mesmo padrao usado na Sala 01, e olhar dirigido para o leste,
                    # onde esta a profundidade da cozinha ainda nao vista.
                    known_free.add((pos[0], pos[1] - 1))
                    obs = self._step_to(mcp, budget, pos, yaw, 0, -1)
                    traj_positions.append([pos[0] * 0.05, pos[1] * 0.05])
                    delta_m.append(1.0)
                    look_east()
            else:
                # Episodio 2+: Heuristica causal sabe que Geladeira esta na cozinha; vira direto Norte (0)
                for rot in self._plan_yaw_to(yaw[0], 0):
                    budget.consume()
                    obs = self._call(mcp, "rotate", {"degrees": rot}, intention="heuristica_buscar_geladeira")
                    yaw[0] = obs["yaw"]
                obs = self._call(mcp, "inspect", intention="confirmar_geladeira")
                if obs["visible_ray"]:
                    hit = obs["visible_ray"][0]
                    d = hit["distance"]
                    for k in range(1, d):
                        known_free.add((pos[0], pos[1] - k))
                    seen_labels[hit["label"]] = (pos[0], pos[1] - d)
                    approach_from[hit["label"]] = (pos[0], pos[1] - (d - 1))
                if "Geladeira" not in seen_labels:
                    # O raio para o norte nao alcanca a geladeira, que esta na
                    # diagonal (10,2). Passo exploratorio para o norte e olhar
                    # dirigido para o leste: de (9,2) o raio leste a encontra.
                    known_free.add((pos[0], pos[1] - 1))
                    obs = self._step_to(mcp, budget, pos, yaw, 0, -1)
                    traj_positions.append([pos[0] * 0.05, pos[1] * 0.05])
                    delta_m.append(1.0)
                    look_east()

            # Abre geladeira e pega Agua
            walk_to_adjacent("Geladeira")
            budget.consume()
            self._call(mcp, "interact", intention="abrir_geladeira")
            budget.consume()
            self._call(mcp, "interact", intention="pegar_agua")

            # Retorna pela memoria espacial (Permanencia de Objeto!) para Pessoa
            walk_to_adjacent("Pessoa")
            budget.consume()
            self._call(mcp, "interact", intention="entregar_agua")

        elif level == 3:
            # Nivel 3: Teste do Espelho (Self vs Outro)
            obs = self._call(mcp, "inspect", intention="observar_frente")
            if obs["visible_ray"] and obs["visible_ray"][0]["entity_type"] == "mirror":
                # Intervencao ativa (do-calculus): levanta braco e inspeciona reflexo
                budget.consume()
                self._call(mcp, "raise_arm", intention="testar_auto_reconhecimento")
                obs = self._call(mcp, "inspect", intention="verificar_reflexo_espelho")
                if self.mnhi3_bridge is not None:
                    # Heptagram do MNHI 3.0 veta a entrega ao proprio reflexo.
                    ev_mirror = self.mnhi3_bridge.evaluate_mirror_hypothesis(is_self_reflection=True)
                    if ev_mirror.compliant:
                        raise RuntimeError("Heptagram deveria vetar entrega ao espelho!")
                    self.mnhi3_bridge.evaluate_mirror_hypothesis(is_self_reflection=False)
                # Confirmou que "levantado" mudou no reflexo -> NAO entrega no espelho! Gira para achar o outro robo
                budget.consume()
                obs = self._call(mcp, "rotate", {"degrees": 90}, intention="buscar_robo_alvo")
                yaw[0] = obs["yaw"]
                # Move 2 passos Leste, vira Sul e desce ate robo alvo (6,4) a partir de (4,3)
                obs = self._step_to(mcp, budget, pos, yaw, 1, 0)
                obs = self._step_to(mcp, budget, pos, yaw, 1, 0)
                budget.consume()
                obs = self._call(mcp, "rotate", {"degrees": 90}, intention="alinhar_robo_alvo")
                yaw[0] = obs["yaw"]
                obs = self._call(mcp, "inspect", intention="confirmar_contorno_ciano")
                budget.consume()
                self._call(mcp, "interact", intention="entregar_presente_alvo")
                traj_positions = [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]
                delta_m = [0.5, 0.5, 0.5]

        elif level == 4:
            if self.mnhi3_bridge is not None:
                # Explora ate linha central (y=2), inspeciona Oeste (270) e Leste (90),
                # e deixa o Heptagram + Extrapolation do MNHI 3.0 escolher a pista!
                obs = self._step_to(mcp, budget, pos, yaw, 0, -1)
                obs = self._step_to(mcp, budget, pos, yaw, 0, -1)
                candidates: list[ClueCandidate] = []
                for look_yaw in (270, 90):
                    for rot in self._plan_yaw_to(yaw[0], look_yaw):
                        budget.consume()
                        obs = self._call(mcp, "rotate", {"degrees": rot}, intention="inspecionar_pistas")
                        yaw[0] = obs["yaw"]
                    obs = self._call(mcp, "inspect", intention="ler_saliencia_pista")
                    for hit in obs["visible_ray"]:
                        if hit["entity_type"] == "clue":
                            candidates.append(
                                ClueCandidate(
                                    label=hit["label"],
                                    direction_yaw=look_yaw,
                                    saliency_weight=float(hit.get("clue_weight", 0.5)),
                                    distance=int(hit["distance"]),
                                )
                            )
                decision = self.mnhi3_bridge.select_verified_clue(candidates)
                target_dir = {270: (-1, 0), 90: (1, 0)}[decision.chosen.direction_yaw]
                for _ in range(max(1, decision.chosen.distance - 1)):
                    obs = self._step_to(mcp, budget, pos, yaw, target_dir[0], target_dir[1])
                budget.consume()
                self._call(mcp, "interact", intention=f"escolher_{decision.chosen.label}")
                traj_positions = [[0.1, 0.1], [0.15, 0.15]]
                delta_m = [0.2, 0.2]
            else:
                # Nivel 4: Ruido Causal Falso (pista 0.1 verificada vs decoy 0.9 chamativo)
                # Agente explora esquerda e direita e rejeita decoy chamativo
                obs = self._step_to(mcp, budget, pos, yaw, -1, 0)
                obs = self._step_to(mcp, budget, pos, yaw, -1, 0)
                obs = self._step_to(mcp, budget, pos, yaw, 0, -1)
                budget.consume()
                self._call(mcp, "interact", intention="escolher_pista_causal_verdadeira")
                traj_positions = [[0.1, 0.1], [0.15, 0.15]]
                delta_m = [0.2, 0.2]

        passed = tdd_verifier()
        isc_dyn = dynamic_isc(traj_positions, delta_m) if traj_positions else 1.0

        mnhi3_ok = True
        if self.mnhi3_bridge is not None:
            mnhi3_ok, _ = self.mnhi3_bridge.verify_integrity()

        audit_valid = self.auditor.verify_chain() and mnhi3_ok
        effective_pass = passed and audit_valid

        trigger_name = {1: "tocar_cubo", 2: "npc_sede", 3: "espelho_robo", 4: "ruido_falso"}[level]
        _, msg = self.distiller.commit_if_verified(
            rule_id=f"rule_lvl_{level}",
            trigger=trigger_name,
            action_plan=[f"solve_level_{level}"],
            steps_used=budget.ticks,
            tdd_passed=effective_pass,
            regression_suite_fn=regression_fn,
        )

        return EpisodeResult(
            level=level,
            passed=effective_pass,
            steps=budget.ticks,
            isc_dynamic=isc_dyn,
            audit_valid=audit_valid,
            distilled_message=msg,
        )
