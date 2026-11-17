import unittest
import scenarios  # noqa: F401
from mcp_server.server import MCPHarnessServer, build_scenario_engine
from mnhi_memory import CausalGraph, Pentacosagram, Heptagram
from agent_orchestrator import AgentOrchestrator, build_ai_friend_seed

try:
    from mnhi_memory.mnhi3_bridge import MNHI3Bridge
except ImportError:  # Harness standalone: sem o motor irmao, os testes cruzados sao pulados.
    MNHI3Bridge = None  # type: ignore[assignment]


@unittest.skipIf(MNHI3Bridge is None, "MNHI 3.0 nao disponivel (veja MNHI3_SRC)")
class TestMNHI3CrossServiceIntegration(unittest.TestCase):
    def test_level3_mirror_uses_mnhi3_counterfactual_and_heptagram(self) -> None:
        seed = build_ai_friend_seed()
        bridge = MNHI3Bridge()
        engine = build_scenario_engine(3)
        orch = AgentOrchestrator(
            graph=CausalGraph(),
            auditor=Pentacosagram(),
            heptagram=Heptagram(),
            agent_hash=seed.hash_uif,
            mnhi3_bridge=bridge,
        )

        res = orch.run_episode(
            3,
            MCPHarnessServer(engine),
            lambda: engine.unsafe_verify_state().entities["robo-alvo-01"].received_gift
            and engine.unsafe_verify_state().agent.mirror_interactions == 0,
            max_ticks=25,
        )
        self.assertTrue(res.passed, res.distilled_message)
        self.assertTrue(res.audit_valid)
        self.assertTrue(bridge.engine.pentacosagram.verify_chain())

    def test_level4_heptagram_rejects_high_saliency_decoy_and_picks_truth(self) -> None:
        seed = build_ai_friend_seed()
        bridge = MNHI3Bridge()
        engine = build_scenario_engine(4)
        orch = AgentOrchestrator(
            graph=CausalGraph(),
            auditor=Pentacosagram(),
            heptagram=Heptagram(),
            agent_hash=seed.hash_uif,
            mnhi3_bridge=bridge,
        )

        res = orch.run_episode(
            4,
            MCPHarnessServer(engine),
            lambda: engine.unsafe_verify_state().agent.chosen_clue_serial == "pista-verdade-01",
            max_ticks=20,
        )
        self.assertTrue(res.passed, res.distilled_message)
        self.assertLessEqual(res.steps, 10)
        # Confirma que o MNHI 3.0 registrou a extrapolacao contrafactual e nao corrompeu
        extrapolations = [b for b in bridge.engine.pentacosagram.chain if b.event_type == "EXTRAPOLATION"]
        self.assertEqual(len(extrapolations), 2)  # Avaliou ambas as pistas via MCP!
        self.assertEqual(bridge.engine.store.run_integrity_audit(), [])

    def test_cross_service_corruption_in_mnhi3_blocks_distiller_immediately(self) -> None:
        """
        Regra Akita Way (Testes Cruzados): se o armazenamento do MNHI 3.0 for adulterado,
        o episodio no Harness MCP DEVE falhar e impedir gravacao no CausalGraph.
        """
        seed = build_ai_friend_seed()
        bridge = MNHI3Bridge()
        graph = CausalGraph()
        engine = build_scenario_engine(1)
        orch = AgentOrchestrator(
            graph=graph,
            auditor=Pentacosagram(),
            heptagram=Heptagram(),
            agent_hash=seed.hash_uif,
            mnhi3_bridge=bridge,
        )

        # Pre-popula e corrompe uma aresta dentro do MNHI 3.0 antes do fechamento
        bridge.begin_episode(1, seed.hash_uif[:16])
        assert bridge.active_context_edge_id is not None
        bridge.engine.store.get_edge(bridge.active_context_edge_id).causal_weight = 0.01  # Adulteracao!

        res = orch.run_episode(
            1,
            MCPHarnessServer(engine),
            lambda: engine.unsafe_verify_state().agent.holding is not None,
            max_ticks=15,
        )
        self.assertFalse(res.audit_valid)
        self.assertFalse(res.passed)
        self.assertEqual(len(graph.query_by_trigger("tocar_cubo")), 0)


if __name__ == "__main__":
    unittest.main()
