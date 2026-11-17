import unittest
import scenarios  # noqa: F401
from mcp_server.server import MCPHarnessServer, build_scenario_engine
from mnhi_memory import CausalGraph, Pentacosagram, Heptagram, Distiller, dynamic_isc
from world_engine.pulse_oracle import PulseOracle, PulseTrial
from agent_orchestrator import AgentOrchestrator, build_ai_friend_seed


class TestEvolutionAndPulseOracle(unittest.TestCase):
    def test_cross_episode_evolution_reduces_steps_and_preserves_regression(self) -> None:
        seed = build_ai_friend_seed()
        graph = CausalGraph()
        hept = Heptagram()

        def regression_level_1() -> bool:
            e1 = build_scenario_engine(1)
            o1 = AgentOrchestrator(graph, Pentacosagram(), hept, seed.hash_uif)
            r1 = o1.run_episode(
                1,
                MCPHarnessServer(e1),
                lambda: e1.unsafe_verify_state().agent.holding is not None,
                max_ticks=15,
            )
            return r1.passed

        # Episodio 1 (Zero-shot exploratorio no Nivel 2)
        eng1 = build_scenario_engine(2)
        orch1 = AgentOrchestrator(graph, Pentacosagram(), hept, seed.hash_uif)
        res1 = orch1.run_episode(
            2,
            MCPHarnessServer(eng1),
            lambda: not eng1.unsafe_verify_state().entities["npc-sede-01"].thirsty,
            max_ticks=50,
            regression_fn=regression_level_1,
        )
        self.assertTrue(res1.passed)
        self.assertEqual(len(graph.query_by_trigger("npc_sede")), 1)

        # Episodio 2 (Usa heuristica destilada do Grafo Causal -> precisa gastar menos passos!)
        eng2 = build_scenario_engine(2)
        orch2 = AgentOrchestrator(graph, Pentacosagram(), hept, seed.hash_uif)
        res2 = orch2.run_episode(
            2,
            MCPHarnessServer(eng2),
            lambda: not eng2.unsafe_verify_state().entities["npc-sede-01"].thirsty,
            max_ticks=50,
            regression_fn=regression_level_1,
        )
        self.assertTrue(res2.passed)
        self.assertLess(res2.steps, res1.steps)

    def test_distiller_rejects_failed_tdd_and_penalizes_credit(self) -> None:
        graph = CausalGraph()
        auditor = Pentacosagram()
        hept = Heptagram()
        dist = Distiller(graph, auditor, hept)

        ok, _ = dist.commit_if_verified("bad_rule", "trigger_x", ["loop"], steps_used=50, tdd_passed=False)
        self.assertFalse(ok)
        self.assertEqual(hept.get_credit("bad_rule"), 0.5)
        self.assertEqual(len(graph.query_by_trigger("trigger_x")), 0)

    def test_pulse_network_vai_fica_tdd_and_dynamic_isc(self) -> None:
        trial = PulseTrial(
            trial_id="go-nogo-01",
            stimulus_vector=(0.8, 0.2),
            expected_decision="VAI",
            ideal_state_vec=(0.2, 0.05),
            max_latency_ticks=5,
        )
        # Delta_M na rede de pulsos = variacao da taxa de disparo por tick dentro do ensaio
        delta_spikes_stable = [0.25, 0.25, 0.25, 0.25]
        # Trajetoria estavel: pontos distintos e proximos (o terceiro ponto nao
        # pode repetir o primeiro — ponto duplicado infla H_fase e derruba o ISC).
        spike_positions = [[0.2, 0.05], [0.21, 0.05], [0.22, 0.05]]

        passed, msg = PulseOracle.verify_trial(trial, decision="VAI", latency_ticks=4, delta_spikes=delta_spikes_stable)
        self.assertTrue(passed, msg)

        isc_traj = dynamic_isc(spike_positions, delta_spikes_stable)
        self.assertGreater(isc_traj, 0.85)


if __name__ == "__main__":
    unittest.main()
