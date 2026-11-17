import bisect
import hashlib
import json
from typing import Any, Iterable, Sequence


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_str(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CanonicalHasher:
    """Camada 0 do MNHI 3.0: separacao estrita entre Identidade Imutavel e Estado Mutavel."""

    @staticmethod
    def hash_uif(serial: str, tipo_agente: str, funcao_s: str) -> str:
        if not serial or not tipo_agente or not funcao_s:
            raise ValueError("serial, tipo_agente e funcao_s sao obrigatorios e imutaveis.")
        payload = canonical_json({
            "kind": "UIF",
            "serial": serial,
            "tipo_agente": tipo_agente,
            "funcao_s": funcao_s,
        })
        return sha256_str(payload)

    @staticmethod
    def hash_edge(origin_hash: str, target_hash: str, relation_type: str) -> str:
        if not origin_hash or not target_hash or not relation_type:
            raise ValueError("origin_hash, target_hash e relation_type sao obrigatorios.")
        payload = canonical_json({
            "kind": "EDGE",
            "origin": origin_hash,
            "target": target_hash,
            "relation": relation_type,
        })
        return sha256_str(payload)

    @staticmethod
    def hash_symbol(serial: str, eigenvalues: Sequence[float], precision: int = 6) -> str:
        rounded_sorted = sorted(round(float(x), precision) for x in eigenvalues)
        payload = canonical_json({
            "kind": "SYMBOL",
            "serial": serial,
            "spectrum": rounded_sorted,
        })
        return sha256_str(payload)

    @staticmethod
    def hash_meta_symbol(merkle_root: str, meta_type: str, h_causal: float, precision: int = 6) -> str:
        payload = canonical_json({
            "kind": "META_SYMBOL",
            "merkle_root": merkle_root,
            "meta_type": meta_type,
            "h_causal": round(float(h_causal), precision),
        })
        return sha256_str(payload)

    @staticmethod
    def hash_generic(obj: Any) -> str:
        return sha256_str(canonical_json(obj))


class ConsistentHashRing:
    """Sharding canonico O(log M) usando os primeiros bits do SHA-256."""

    def __init__(self, shard_ids: Iterable[str], replicas: int = 16, prefix_bits: int = 32) -> None:
        self.prefix_bits = prefix_bits
        self.ring_mod = 1 << prefix_bits
        self._ring: list[tuple[int, str]] = []
        self._shards: set[str] = set()
        for sid in shard_ids:
            self.add_shard(sid, replicas=replicas)

    def _key_to_slot(self, hex_hash: str) -> int:
        hex_chars = max(1, self.prefix_bits // 4)
        return int(hex_hash[:hex_chars], 16) % self.ring_mod

    def add_shard(self, shard_id: str, replicas: int = 16) -> None:
        if shard_id in self._shards:
            return
        self._shards.add(shard_id)
        for r in range(replicas):
            vhash = sha256_str(f"{shard_id}#vnode{r}")
            slot = self._key_to_slot(vhash)
            bisect.insort(self._ring, (slot, shard_id))

    def get_shard(self, canonical_hash: str) -> str:
        if not self._ring:
            raise RuntimeError("Anel de sharding vazio.")
        slot = self._key_to_slot(canonical_hash)
        idx = bisect.bisect_left(self._ring, (slot, ""))
        if idx == len(self._ring):
            idx = 0
        return self._ring[idx][1]
