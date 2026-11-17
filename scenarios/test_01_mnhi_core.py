import unittest
import scenarios  # noqa: F401 - garante packages no sys.path
from mnhi_memory import (
    CanonicalHasher,
    ConsistentHashRing,
    MerkleTree,
    Pentacosagram,
    Heptagram,
    laplacian_eigenvalues_norm,
    spectral_variance,
    topological_isc,
    jensen_shannon_divergence_bits,
)


class TestMNHICore(unittest.TestCase):
    def test_identity_vs_state_invariance(self) -> None:
        h1 = CanonicalHasher.hash_uif("uif-01", "Executor", "S_v1")
        h2 = CanonicalHasher.hash_uif("uif-01", "Executor", "S_v1")
        h3 = CanonicalHasher.hash_uif("uif-01", "Executor", "S_v2")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)

    def test_consistent_hashing_sharding_deterministic(self) -> None:
        ring = ConsistentHashRing(["shard_0", "shard_1", "shard_2", "shard_3"])
        h = CanonicalHasher.hash_uif("uif-07", "Sensor", "R1")
        self.assertEqual(ring.get_shard(h), ring.get_shard(h))
        self.assertIn(ring.get_shard(h), {"shard_0", "shard_1", "shard_2", "shard_3"})

    def test_laplacian_symbol_emergence_k3_vs_k5(self) -> None:
        # K3: Var(lambda) = 0.5 -> rejeitado para simbolo estavel (Y = 0.25)
        k3 = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        eigs_k3 = laplacian_eigenvalues_norm(k3)
        self.assertAlmostEqual(spectral_variance(eigs_k3), 0.5, places=4)

        # K5: ISC_top = 1.0 e Var(lambda) = 0.25 -> aprovado!
        k5 = [[0 if i == j else 1 for j in range(5)] for i in range(5)]
        eigs_k5 = laplacian_eigenvalues_norm(k5)
        var_k5 = spectral_variance(eigs_k5)
        isc_k5 = topological_isc(10, 5)
        self.assertGreaterEqual(isc_k5, 0.85)
        self.assertLessEqual(var_k5, 0.25)

    def test_merkle_inclusion_and_pentacosagram_tamper_detection(self) -> None:
        h_true = CanonicalHasher.hash_edge("A", "C", "VERDADEIRO")
        h_noise = CanonicalHasher.hash_edge("B", "C", "RUIDO")
        tree = MerkleTree([h_true, h_noise])
        proof = tree.get_proof(0)
        self.assertTrue(MerkleTree.verify_proof(h_true, proof, tree.root))

        auditor = Pentacosagram()
        e1 = auditor.record_step(
            tick=1,
            origin_hash="A",
            target_hash="C",
            consulted_hashes=[h_true, h_noise],
            intention="seguir_verdade",
            action="move",
            observed_vec=[0.2, 0.0],
            ideal_vec=[0.1, 0.0],
            tau=1.0,
            required_true_hash=h_true,
        )
        self.assertEqual(e1.status, "VALIDO")
        self.assertTrue(auditor.verify_chain())

        # Forca alucinacao (D_Hyper > tau)
        e2 = auditor.record_step(
            tick=2,
            origin_hash="A",
            target_hash="C",
            consulted_hashes=[h_noise],  # excluiu verdade e saltou longe
            intention="alucinar",
            action="2x_speed",
            observed_vec=[0.85, 0.0],
            ideal_vec=[0.1, 0.0],
            tau=1.0,
            required_true_hash=h_true,
        )
        self.assertEqual(e2.status, "ALUCINACAO_DETECTADA")

        # Adulteracao retroativa deve quebrar verify_chain()
        auditor.entries[0].action = "adulterado"
        self.assertFalse(auditor.verify_chain())

    def test_heptagram_jsd_veto_and_quarantine(self) -> None:
        hept = Heptagram(jsd_min_safe=0.15)
        p_curr = [0.4, 0.3, 0.2, 0.1]
        p_prop_tirania = [0.6, 0.2, 0.1, 0.1]
        p_tirania = [0.9, 0.033, 0.033, 0.034]

        jsd_curr = jensen_shannon_divergence_bits(p_curr, p_tirania)
        jsd_prop = jensen_shannon_divergence_bits(p_prop_tirania, p_tirania)
        self.assertGreater(jsd_curr, 0.15)
        self.assertLess(jsd_prop, 0.15)

        veto = hept.evaluate_law_change(p_curr, p_prop_tirania, p_tirania)
        self.assertTrue(veto.vetoed)

        # Penalidade exponencial 0.5^k ate quarentena (< 0.05)
        for _ in range(5):
            hept.apply_penalty("heuristica_ruim")
        self.assertTrue(hept.is_quarantined("heuristica_ruim"))


if __name__ == "__main__":
    unittest.main()
