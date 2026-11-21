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
    var_ponderada_normalizada,
    can_emerge_symbol,
    jensen_shannon_divergence_bits,
    jsd_uniforme_para_tirania,
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
        # Regra canônica (DEFINICOES.md, seções 2-3): ISC >= 0,85 E VarPn <= 0,36.
        # A Var(λ) antiga fica só como diagnóstico: ela rejeitava o K3 (0,5)
        # por artefato — sob VarPn o K3 uniforme é Símbolo (0,25 <= 0,36).
        k3 = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        eigs_k3 = laplacian_eigenvalues_norm(k3)
        self.assertAlmostEqual(spectral_variance(eigs_k3), 0.5, places=4)
        self.assertAlmostEqual(var_ponderada_normalizada(k3), 0.25, places=4)
        self.assertTrue(can_emerge_symbol(1.0, 0.25)[0])

        # K5: ISC_top = 1.0 e VarPn = 0,0625 -> aprovado!
        k5 = [[0 if i == j else 1 for j in range(5)] for i in range(5)]
        eigs_k5 = laplacian_eigenvalues_norm(k5)
        var_k5 = spectral_variance(eigs_k5)
        isc_k5 = topological_isc(10, 5)
        self.assertGreaterEqual(isc_k5, 0.85)
        self.assertLessEqual(var_k5, 0.25)
        self.assertAlmostEqual(var_ponderada_normalizada(k5), 0.0625, places=4)
        self.assertTrue(can_emerge_symbol(isc_k5, 0.0625)[0])

    def test_varpn_k6_reponderado(self) -> None:
        # O caso que matou a Var(λ) antiga: um rastro x10 no K6 levava
        # Var 0,20 -> 0,2882 e matava o Símbolo. Sob VarPn ele sobrevive
        # (0,175) e só morre com x100 (0,764) — valores do gym, 18-19/11.
        def k_completo(n, pesado=None):
            adj = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j:
                        adj[i][j] = 1.0
            if pesado:
                (a, b), w = pesado
                adj[a][b] = adj[b][a] = w
            return adj

        self.assertAlmostEqual(
            var_ponderada_normalizada(k_completo(6)), 0.04, places=4
        )
        self.assertAlmostEqual(
            var_ponderada_normalizada(k_completo(6, ((0, 1), 10.0))),
            0.175,
            places=3,
        )
        self.assertTrue(
            can_emerge_symbol(1.0, 0.175)[0]
        )
        self.assertAlmostEqual(
            var_ponderada_normalizada(k_completo(6, ((0, 1), 100.0))),
            0.764,
            places=3,
        )
        self.assertFalse(
            can_emerge_symbol(1.0, 0.764)[0]
        )

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

    def test_h_causal_blocks_tyranny_and_allows_meta_symbol_emergence(self) -> None:
        from mnhi_memory.metrics import causal_entropy_bits, can_emerge_meta_symbol
        from mnhi_memory import CanonicalHasher, MerkleTree

        # Cenario 1: Tirania Causal (1 shard domina) -> H_causal ~ 0.63 bits -> BLOQUEADO
        tyranny = [0.90, 0.033, 0.033, 0.034]
        self.assertAlmostEqual(causal_entropy_bits(tyranny), 0.6275, places=3)
        emerged_bad, _, _ = can_emerge_meta_symbol(tyranny, all_symbols_valid=True)
        self.assertFalse(emerged_bad)

        # Cenario 2: Ordem na Diversidade (4 shards coesos e equilibrados) -> APROVADO
        balanced = [0.90, 0.88, 0.85, 0.89]
        emerged_ok, isc_g, h_c = can_emerge_meta_symbol(balanced, all_symbols_valid=True)
        self.assertTrue(emerged_ok)
        self.assertGreaterEqual(isc_g, 0.80)
        self.assertGreaterEqual(h_c, 1.0)

        # Agora sim forja o Hash_Meta_Simbolo com raiz Merkle + H_causal auditado
        tree = MerkleTree(["s1", "s2", "s3", "s4"])
        meta_hash = CanonicalHasher.hash_meta_symbol(tree.root, "NOOSFERA_V1", h_c)
        self.assertEqual(len(meta_hash), 64)

    def test_heptagram_jsd_veto_and_quarantine(self) -> None:
        # Veto JSD relativo (DEFINICOES.md, secao 5): veta se a JSD proposta
        # ficar abaixo de JSD(uniforme, Tirania) - 0,05, ou se regredir.
        # N=4: piso relativo = 0,3436 - 0,05 = 0,2936.
        hept = Heptagram()
        p_curr = [0.4, 0.3, 0.2, 0.1]
        p_prop_tirania = [0.6, 0.2, 0.1, 0.1]
        p_tirania = [0.9, 0.033, 0.033, 0.034]

        jsd_curr = jensen_shannon_divergence_bits(p_curr, p_tirania)
        jsd_prop = jensen_shannon_divergence_bits(p_prop_tirania, p_tirania)
        piso_relativo = jsd_uniforme_para_tirania(4) - 0.05
        self.assertAlmostEqual(piso_relativo, 0.2936, places=3)
        self.assertLess(jsd_prop, piso_relativo)
        self.assertLess(jsd_prop, jsd_curr)

        veto = hept.evaluate_law_change(p_curr, p_prop_tirania, p_tirania)
        self.assertTrue(veto.vetoed)

        # Penalidade exponencial 0.5^k ate quarentena (< 0.05)
        for _ in range(5):
            hept.apply_penalty("heuristica_ruim")
        self.assertTrue(hept.is_quarantined("heuristica_ruim"))


if __name__ == "__main__":
    unittest.main()
