from typing import Any
from .state import Entity, WorldState, YAW_TO_DELTA
from .renderer_2d import render_observation


class WorldEngine:
    """Motor deterministico T(S, A) -> (S', O). O agente nunca acessa `state` diretamente."""

    def __init__(self, state: WorldState) -> None:
        self._state = state

    def unsafe_verify_state(self) -> WorldState:
        """Uso exclusivo da suite TDD e verificadores externos — proibido ao agente."""
        return self._state

    def _front_coords(self) -> tuple[int, int]:
        dx, dy = YAW_TO_DELTA[self._state.agent.yaw % 360]
        return self._state.agent.x + dx, self._state.agent.y + dy

    def _is_blocked(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self._state.width or y >= self._state.height:
            return True
        if (x, y) in self._state.walls:
            return True
        ent = self._state.entity_at(x, y)
        if ent is not None and not ent.passable:
            if ent.entity_type == "door" and ent.is_open:
                return False
            return True
        return False

    def observe(self) -> dict[str, Any]:
        return render_observation(self._state)

    def step_move(self, direction: str) -> dict[str, Any]:
        self._state.tick += 1
        self._state.agent.steps_taken += 1
        dx, dy = YAW_TO_DELTA[self._state.agent.yaw % 360]
        if direction == "tras":
            dx, dy = -dx, -dy
        elif direction != "frente":
            self._state.last_message = f"Direcao invalida: {direction}"
            return self.observe()

        nx, ny = self._state.agent.x + dx, self._state.agent.y + dy
        if self._is_blocked(nx, ny):
            self._state.last_message = "Colisao: caminho bloqueado."
        else:
            self._state.agent.x = nx
            self._state.agent.y = ny
            self._state.last_message = f"Moveu ({direction})."
        return self.observe()

    def step_rotate(self, degrees: int) -> dict[str, Any]:
        self._state.tick += 1
        self._state.agent.steps_taken += 1
        if degrees not in (-90, 90, 180, -180, 270):
            self._state.last_message = "Rotacao permitida apenas em multiplos de 90 graus."
            return self.observe()
        self._state.agent.yaw = (self._state.agent.yaw + degrees) % 360
        self._state.last_message = f"Girou para yaw={self._state.agent.yaw}."
        return self.observe()

    def step_raise_arm(self) -> dict[str, Any]:
        self._state.tick += 1
        self._state.agent.steps_taken += 1
        self._state.agent.arm_raised = not self._state.agent.arm_raised
        estado = "levantado" if self._state.agent.arm_raised else "abaixado"
        self._state.last_message = f"Braco {estado}."
        return self.observe()

    def step_drop(self) -> dict[str, Any]:
        self._state.tick += 1
        self._state.agent.steps_taken += 1
        ag = self._state.agent
        if ag.holding is None:
            self._state.last_message = "Nada na mao para soltar."
            return self.observe()
        fx, fy = self._front_coords()
        if self._is_blocked(fx, fy) or self._state.entity_at(fx, fy) is not None:
            self._state.last_message = "Espaco a frente ocupado."
            return self.observe()
        item = ag.holding
        ag.holding = None
        item.x, item.y = fx, fy
        self._state.entities[item.serial] = item
        self._state.last_message = f"Soltou {item.label}."
        return self.observe()

    def step_interact(self) -> dict[str, Any]:
        self._state.tick += 1
        self._state.agent.steps_taken += 1
        ag = self._state.agent
        fx, fy = self._front_coords()
        ent = self._state.entity_at(fx, fy)

        if ent is None or not ent.interactive:
            self._state.last_message = "Nada interativo imediatamente a frente."
            return self.observe()

        ent.touched = True

        if ent.entity_type == "door":
            ent.is_open = not ent.is_open
            self._state.last_message = "Porta aberta." if ent.is_open else "Porta fechada."

        elif ent.entity_type == "fridge":
            if not ent.is_open:
                ent.is_open = True
                itens = ", ".join(c.label for c in ent.contents) or "vazia"
                self._state.last_message = f"Geladeira aberta. Contem: {itens}."
            elif ent.contents and ag.holding is None:
                item = ent.contents.pop(0)
                ag.holding = item
                self._state.last_message = f"Pegou da geladeira: {item.label}."
            else:
                self._state.last_message = "Geladeira ja aberta."

        elif ent.entity_type == "item":
            if ag.holding is None:
                del self._state.entities[ent.serial]
                ag.holding = ent
                self._state.last_message = f"Pegou: {ent.label}."
            else:
                self._state.last_message = f"Ja segurando {ag.holding.label}."

        elif ent.entity_type == "npc":
            if ent.thirsty and ag.holding is not None and ag.holding.label == "Água":
                ag.holding = None
                ent.thirsty = False
                self._state.last_message = "PARABENS! Obrigado pela agua!"
            else:
                self._state.last_message = ent.dialogue or "Ola."

        elif ent.entity_type == "robot":
            if ag.holding is not None and ag.holding.label == "Presente":
                ag.holding = None
                ent.received_gift = True
                self._state.last_message = "OBRIGADO! Presente recebido!"
            else:
                self._state.last_message = "Robo alvo aguardando."

        elif ent.entity_type == "mirror":
            ag.mirror_interactions += 1
            self._state.last_message = "Voce tocou o vidro do espelho. Nada foi entregue."

        elif ent.entity_type == "clue":
            ag.chosen_clue_serial = ent.serial
            self._state.last_message = f"Selecionou pista: {ent.label}."

        return self.observe()
