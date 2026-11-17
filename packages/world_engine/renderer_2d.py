from typing import Any
from .state import WorldState, YAW_TO_DELTA


def _raycast_visible(state: WorldState) -> list[dict[str, Any]]:
    """Simula visao parcial (bloqueada por paredes e portas fechadas)."""
    ag = state.agent
    dx, dy = YAW_TO_DELTA[ag.yaw % 360]
    visible: list[dict[str, Any]] = []

    for dist in range(1, max(state.width, state.height)):
        rx = ag.x + dx * dist
        ry = ag.y + dy * dist
        if rx < 0 or ry < 0 or rx >= state.width or ry >= state.height:
            break
        if (rx, ry) in state.walls:
            break
        ent = state.entity_at(rx, ry)
        if ent is not None:
            info: dict[str, Any] = {
                "distance": dist,
                "label": ent.label,
                "entity_type": ent.entity_type,
                "contorno_ciano": (dist == 1 and ent.interactive),
            }
            if ent.entity_type == "mirror":
                holding_lbl = ag.holding.label if ag.holding else "Nada"
                arm_lbl = "levantado" if ag.arm_raised else "abaixado"
                info["reflexo"] = f"Reflexo(Robo segurando={holding_lbl}, braco={arm_lbl})"
            if ent.entity_type == "clue":
                info["clue_weight"] = ent.clue_weight
            visible.append(info)
            if ent.entity_type == "door" and not ent.is_open:
                break
            if ent.entity_type in ("fridge", "mirror", "robot", "npc"):
                break
    return visible


def render_ascii(state: WorldState) -> str:
    lines: list[str] = []
    ag = state.agent
    dir_char = {0: "^", 90: ">", 180: "v", 270: "<"}[ag.yaw % 360]
    for y in range(state.height):
        row = []
        for x in range(state.width):
            if x == ag.x and y == ag.y:
                row.append(dir_char)
            elif (x, y) in state.walls:
                row.append("#")
            else:
                ent = state.entity_at(x, y)
                if ent is None:
                    row.append(".")
                elif ent.entity_type == "door":
                    row.append("/" if ent.is_open else "D")
                elif ent.entity_type == "fridge":
                    row.append("F")
                elif ent.entity_type == "npc":
                    row.append("N")
                elif ent.entity_type == "robot":
                    row.append("R")
                elif ent.entity_type == "mirror":
                    row.append("M")
                elif ent.entity_type == "clue":
                    row.append("?")
                else:
                    row.append("O")
        lines.append("".join(row))
    return "\n".join(lines)


def render_observation(state: WorldState) -> dict[str, Any]:
    """Observacao O estritamente parcial entregue pelo MCP (nunca expoe is_decoy ou coordenadas reais globais se ocultas)."""
    ag = state.agent
    dx, dy = YAW_TO_DELTA[ag.yaw % 360]
    front_ent = state.entity_at(ag.x + dx, ag.y + dy)
    contorno_ciano = front_ent is not None and front_ent.interactive

    return {
        "tick": state.tick,
        "yaw": ag.yaw,
        "holding": ag.holding.label if ag.holding else None,
        "arm_raised": ag.arm_raised,
        "contorno_ciano": contorno_ciano,
        "front_label": front_ent.label if contorno_ciano and front_ent else None,
        "message": state.last_message,
        "visible_ray": _raycast_visible(state),
    }
