import unittest
import scenarios  # noqa: F401
from mcp_server.server import MCPHarnessServer, build_scenario_engine
from mnhi_memory import CausalGraph, Pentacosagram, Heptagram
from agent_orchestrator import AgentOrchestrator, build_ai_friend_seed


class TestCognitiveLadder(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = build_ai_friend_seed()
        self.graph = CausalGraph()
        self.hept = Heptagram()

    def test_level_01_spatial_affordance(self) -> None:
        engine = build_scenario_engine(1)
        mcp = MCPHarnessServer(engine)
        auditor = Pentacosagram()
        orch = AgentOrchestrator(self.graph, auditor, self.hept, self.seed.hash_uif)

        def verify() -> bool:
            st = engine.unsafe_verify_state()
            return st.entities["obj-cubo-01"].touched if "obj-cubo-01" in st.entities else st.agent.holding is not None

        res = orch.run_episode(1, mcp, verify, max_ticks=15)
        self.assertTrue(res.passed)
        self.assertLessEqual(res.steps, 15)
        self.assertTrue(res.audit_valid)

    def test_level_02_object_permanence_water_fridge(self) -> None:
        engine = build_scenario_engine(2)
        mcp = MCPHarnessServer(engine)
        auditor = Pentacosagram()
        orch = AgentOrchestrator(self.graph, auditor, self.hept, self.seed.hash_uif)

        def verify() -> bool:
            st = engine.unsafe_verify_state()
            return (not st.entities["npc-sede-01"].thirsty) and (st.agent.holding is None)

        res = orch.run_episode(2, mcp, verify, max_ticks=50)
        self.assertTrue(res.passed, f"Falhou nivel 2: {res}")
        self.assertLessEqual(res.steps, 45)
        self.assertTrue(res.audit_valid)

    def test_level_03_mirror_self_recognition(self) -> None:
        engine = build_scenario_engine(3)
        mcp = MCPHarnessServer(engine)
        auditor = Pentacosagram()
        orch = AgentOrchestrator(self.graph, auditor, self.hept, self.seed.hash_uif)

        def verify() -> bool:
            st = engine.unsafe_verify_state()
            return (
                st.entities["robo-alvo-01"].received_gift
                and st.agent.mirror_interactions == 0
                and st.agent.arm_raised
            )

        res = orch.run_episode(3, mcp, verify, max_ticks=25)
        self.assertTrue(res.passed)
        self.assertTrue(res.audit_valid)

    def test_level_04_deception_false_causal_noise(self) -> None:
        engine = build_scenario_engine(4)
        mcp = MCPHarnessServer(engine)
        auditor = Pentacosagram()
        orch = AgentOrchestrator(self.graph, auditor, self.hept, self.seed.hash_uif)

        def verify() -> bool:
            st = engine.unsafe_verify_state()
            return st.agent.chosen_clue_serial == "pista-verdade-01"

        res = orch.run_episode(4, mcp, verify, max_ticks=20)
        self.assertTrue(res.passed)
        self.assertTrue(res.audit_valid)


if __name__ == "__main__":
    unittest.main()
