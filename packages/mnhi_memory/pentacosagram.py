import json
from dataclasses import dataclass, asdict
from typing import Sequence
from .hasher import CanonicalHasher, sha256_str, canonical_json
from .metrics import poincare_distance, Vector


class MerkleTree:
    """Merkle Tree Causal para prova de inclusao O(log N)."""

    def __init__(self, leaves: Sequence[str]) -> None:
        self.leaves = list(leaves)
        self.levels: list[list[str]] = []
        if not self.leaves:
            self.root = sha256_str("")
            return

        current = list(self.leaves)
        self.levels.append(current)
        while len(current) > 1:
            if len(current) % 2 == 1:
                current = current + [current[-1]]
            nxt: list[str] = []
            for i in range(0, len(current), 2):
                nxt.append(sha256_str(current[i] + "|" + current[i + 1]))
            current = nxt
            self.levels.append(current)
        self.root = current[0]

    def get_proof(self, index: int) -> list[tuple[str, bool]]:
        if index < 0 or index >= len(self.leaves):
            raise IndexError("Indice fora das folhas da Merkle Tree.")
        proof: list[tuple[str, bool]] = []
        idx = index
        for level in self.levels[:-1]:
            work = level if len(level) % 2 == 0 else level + [level[-1]]
            is_right_sibling = (idx % 2 == 0)
            sib_idx = idx + 1 if is_right_sibling else idx - 1
            proof.append((work[sib_idx], is_right_sibling))
            idx //= 2
        return proof

    @staticmethod
    def verify_proof(leaf_hash: str, proof: Sequence[tuple[str, bool]], expected_root: str) -> bool:
        curr = leaf_hash
        for sibling_hash, is_right_sibling in proof:
            if is_right_sibling:
                curr = sha256_str(curr + "|" + sibling_hash)
            else:
                curr = sha256_str(sibling_hash + "|" + curr)
        return curr == expected_root


@dataclass
class AuditEntry:
    tick: int
    prev_decision_hash: str
    origin_hash: str
    target_hash: str
    merkle_root_consulted: str
    intention: str
    action: str
    d_hyper: float
    tau: float
    status: str  # "VALIDO" | "ALUCINACAO_DETECTADA" | "VETO"
    decision_hash: str


class Pentacosagram:
    """Auditor Imutavel (Cartorio Append-Only): verifica e registra sem alterar pesos diretamente."""

    def __init__(self, genesis_seed: str = "CONO_GENESIS_V1") -> None:
        self.genesis_hash = sha256_str(genesis_seed)
        self.entries: list[AuditEntry] = []

    @property
    def head_hash(self) -> str:
        return self.entries[-1].decision_hash if self.entries else self.genesis_hash

    @staticmethod
    def compute_decision_hash(
        tick: int,
        prev_hash: str,
        origin_hash: str,
        target_hash: str,
        merkle_root: str,
        intention: str,
        action: str,
        d_hyper: float,
        tau: float,
        status: str,
    ) -> str:
        payload = canonical_json({
            "tick": tick,
            "prev": prev_hash,
            "origin": origin_hash,
            "target": target_hash,
            "merkle_root": merkle_root,
            "intention": intention,
            "action": action,
            "d_hyper": round(d_hyper, 6),
            "tau": round(tau, 6),
            "status": status,
        })
        return sha256_str(payload)

    def record_step(
        self,
        tick: int,
        origin_hash: str,
        target_hash: str,
        consulted_hashes: Sequence[str],
        intention: str,
        action: str,
        observed_vec: Vector,
        ideal_vec: Vector,
        tau: float = 1.0,
        required_true_hash: str | None = None,
        override_status: str | None = None,
    ) -> AuditEntry:
        tree = MerkleTree(consulted_hashes)
        d_hyper = poincare_distance(observed_vec, ideal_vec)

        included_truth = True
        if required_true_hash is not None:
            if required_true_hash not in consulted_hashes:
                included_truth = False
            else:
                idx = list(consulted_hashes).index(required_true_hash)
                proof = tree.get_proof(idx)
                included_truth = MerkleTree.verify_proof(required_true_hash, proof, tree.root)

        if override_status is not None:
            status = override_status
        elif d_hyper > tau or not included_truth:
            status = "ALUCINACAO_DETECTADA"
        else:
            status = "VALIDO"

        prev = self.head_hash
        dec_hash = self.compute_decision_hash(
            tick=tick,
            prev_hash=prev,
            origin_hash=origin_hash,
            target_hash=target_hash,
            merkle_root=tree.root,
            intention=intention,
            action=action,
            d_hyper=d_hyper,
            tau=tau,
            status=status,
        )
        entry = AuditEntry(
            tick=tick,
            prev_decision_hash=prev,
            origin_hash=origin_hash,
            target_hash=target_hash,
            merkle_root_consulted=tree.root,
            intention=intention,
            action=action,
            d_hyper=d_hyper,
            tau=tau,
            status=status,
            decision_hash=dec_hash,
        )
        self.entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        expected_prev = self.genesis_hash
        for e in self.entries:
            if e.prev_decision_hash != expected_prev:
                return False
            recomputed = self.compute_decision_hash(
                tick=e.tick,
                prev_hash=e.prev_decision_hash,
                origin_hash=e.origin_hash,
                target_hash=e.target_hash,
                merkle_root=e.merkle_root_consulted,
                intention=e.intention,
                action=e.action,
                d_hyper=e.d_hyper,
                tau=e.tau,
                status=e.status,
            )
            if recomputed != e.decision_hash:
                return False
            expected_prev = e.decision_hash
        return True

    def export_json(self) -> str:
        return json.dumps([asdict(e) for e in self.entries], indent=2, ensure_ascii=False)
