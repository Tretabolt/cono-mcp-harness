import math
from typing import Sequence

Vector = Sequence[float]


def norm_sq(v: Vector) -> float:
    return sum(x * x for x in v)


def clamp_to_disk(v: Vector, max_r: float = 0.999999) -> list[float]:
    n = math.sqrt(norm_sq(v))
    if n >= max_r and n > 0:
        scale = max_r / n
        return [x * scale for x in v]
    return [float(x) for x in v]


def poincare_distance(u: Vector, v: Vector) -> float:
    if len(u) != len(v):
        raise ValueError("Vetores precisam ter a mesma dimensao.")
    uc = clamp_to_disk(u)
    vc = clamp_to_disk(v)
    diff_sq = sum((a - b) ** 2 for a, b in zip(uc, vc))
    denom = (1.0 - norm_sq(uc)) * (1.0 - norm_sq(vc))
    if denom <= 1e-15:
        denom = 1e-15
    arg = 1.0 + (2.0 * diff_sq) / denom
    if arg < 1.0:
        arg = 1.0
    return math.acosh(arg)


def shannon_entropy_bits(values: Sequence[float]) -> float:
    total = sum(v for v in values if v > 0.0)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for v in values:
        if v > 0.0:
            p = v / total
            entropy -= p * math.log2(p)
    return entropy


def spectral_entropy_bits(magnitudes: Sequence[float], num_bins: int = 5) -> float:
    if not magnitudes:
        return 0.0
    lo = min(magnitudes)
    hi = max(magnitudes)
    if math.isclose(lo, hi, rel_tol=1e-9, abs_tol=1e-12):
        return 0.0
    bins = [0.0] * num_bins
    span = hi - lo
    for m in magnitudes:
        idx = int(((m - lo) / span) * (num_bins - 1))
        bins[idx] += 1.0
    return shannon_entropy_bits(bins)


def dynamic_isc(positions: Sequence[Vector], delta_m: Sequence[float]) -> float:
    """ISC_din = 1 / (1 + H_fase + H_espectral)."""
    dists: list[float] = []
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(poincare_distance(positions[i], positions[j]))

    if dists:
        mean_d = sum(dists) / len(dists)
        if mean_d > 1e-12:
            var_d = sum((d - mean_d) ** 2 for d in dists) / len(dists)
            h_phase = var_d / (mean_d * mean_d)
        else:
            h_phase = 0.0
    else:
        h_phase = 0.0

    h_spec = spectral_entropy_bits(delta_m)
    return 1.0 / (1.0 + h_phase + h_spec)


def topological_isc(num_valid_edges: int, num_nodes: int, directed: bool = False) -> float:
    if num_nodes < 2:
        return 0.0
    max_edges = num_nodes * (num_nodes - 1) if directed else (num_nodes * (num_nodes - 1)) // 2
    if max_edges <= 0:
        return 0.0
    return min(1.0, max(0.0, num_valid_edges / max_edges))


def laplacian_eigenvalues_norm(adj: Sequence[Sequence[float]], max_iter: int = 200) -> list[float]:
    """Calcula autovalores da Laplaciana Normalizada L_norm = I - D^(-1/2) A D^(-1/2) via Jacobi."""
    n = len(adj)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    deg = [sum(row) for row in adj]
    inv_sqrt = [1.0 / math.sqrt(d) if d > 1e-12 else 0.0 for d in deg]

    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = (1.0 if deg[i] > 1e-12 else 0.0) - inv_sqrt[i] * adj[i][j] * inv_sqrt[j]
            else:
                mat[i][j] = -inv_sqrt[i] * adj[i][j] * inv_sqrt[j]

    for _ in range(max_iter * n):
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(mat[i][j]) > max_val:
                    max_val = abs(mat[i][j])
                    p, q = i, j
        if max_val < 1e-10:
            break

        app = mat[p][p]
        aqq = mat[q][q]
        apq = mat[p][q]
        if math.isclose(app, aqq, abs_tol=1e-12):
            theta = math.pi / 4.0
        else:
            theta = 0.5 * math.atan2(2.0 * apq, aqq - app)
        c = math.cos(theta)
        s = math.sin(theta)

        for i in range(n):
            if i != p and i != q:
                aip = mat[i][p]
                aiq = mat[i][q]
                mat[i][p] = c * aip - s * aiq
                mat[p][i] = mat[i][p]
                mat[i][q] = s * aip + c * aiq
                mat[q][i] = mat[i][q]

        mat[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        mat[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        mat[p][q] = 0.0
        mat[q][p] = 0.0

    eigs = sorted(max(0.0, mat[i][i]) for i in range(n))
    return eigs


def spectral_variance(eigenvalues: Sequence[float]) -> float:
    if not eigenvalues:
        return 0.0
    mean = sum(eigenvalues) / len(eigenvalues)
    return sum((x - mean) ** 2 for x in eigenvalues) / len(eigenvalues)


def syntergic_factor(isc_local: float, var_lambda: float, eps: float = 1e-6) -> float:
    return isc_local / (var_lambda + eps)


def causal_entropy_bits(shard_iscs: Sequence[float]) -> float:
    """
    H_causal: Entropia de Shannon (base 2) sobre a distribuicao normalizada
    dos ISCs/pesos entre N shards. Mede diversidade contra Tirania Causal.

    H alto = atividade causal distribuída com diversidade (ordem sem
    tirania). H baixo = um único shard monopoliza o peso causal
    (Tirania Causal, H -> 0). Não confunda com o erro antigo (H <= 0,1
    como teto): o portão correto é um PISO, H >= log2(N)/2.
    """
    if not shard_iscs:
        return 0.0
    return shannon_entropy_bits(shard_iscs)


def can_emerge_meta_symbol(
    shard_iscs: Sequence[float],
    all_symbols_valid: bool,
    min_global_isc: float = 0.80,
) -> tuple[bool, float, float]:
    """
    Regra de Emergencia do Meta-Simbolo (Self Global):
      1. Todos os simbolos componentes validos;
      2. ISC_Global (media) >= 0.80 (integracao);
      3. H_causal >= log2(N) / 2 (diversidade minima: ordem sem tirania).
    """
    n = len(shard_iscs)
    if n < 2 or not all_symbols_valid:
        return False, 0.0, 0.0
    isc_global = sum(shard_iscs) / n
    h_causal = causal_entropy_bits(shard_iscs)
    min_h = math.log2(n) / 2.0
    emerged = (isc_global >= min_global_isc) and (h_causal >= min_h)
    return emerged, isc_global, h_causal


def jensen_shannon_divergence_bits(p: Sequence[float], q: Sequence[float]) -> float:
    if len(p) != len(q) or len(p) == 0:
        raise ValueError("Distribuicoes precisam ter o mesmo tamanho > 0.")
    sp = sum(p)
    sq = sum(q)
    if sp <= 0.0 or sq <= 0.0:
        raise ValueError("Soma das distribuicoes deve ser positiva.")

    pn = [x / sp for x in p]
    qn = [x / sq for x in q]
    m = [0.5 * (a + b) for a, b in zip(pn, qn)]

    def kl2(a: Sequence[float], b: Sequence[float]) -> float:
        total = 0.0
        for x, y in zip(a, b):
            if x > 0.0 and y > 0.0:
                total += x * math.log2(x / y)
        return total

    jsd = 0.5 * kl2(pn, m) + 0.5 * kl2(qn, m)
    return min(1.0, max(0.0, jsd))
