"""Pair-copula families and vine construction for correlated sampling.

Per-pair correlation specifications are modelled as a *1-truncated R-vine*,
which is equivalent to a Markov-tree copula: the specified pairs form the first
(and only) tree, and any two fields not joined by an edge are conditionally
independent given the path between them. This lets every pair keep its own
copula family and strength, which a single joint copula cannot do.

Everything here is pure and stateless. Functions take uniform arrays ``u`` and
``w`` -- drawn by the caller so the RNG draw order stays centralised in the
sampler -- and return uniform arrays. ``hinv`` is the inverse conditional CDF
(inverse h-function) of a bivariate copula; sampling one tree edge is
``u_child = hinv(family, w_child, u_parent, param)``. All five families are
exchangeable (symmetric in their two arguments), so the conditioning direction
does not matter.

The ``corr`` value in a spec is interpreted as the (latent) correlation for the
Gaussian and Student-t families and as the target Kendall's tau for the
Archimedean families (Clayton, Gumbel, Frank), matching the historical
semantics of the sampler.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, optimize, stats

from .distributions import CopulaType

# Degrees of freedom for the Student-t copula (matches the full-matrix path).
STUDENT_T_DF = 4

# Clip uniforms away from the open-interval boundaries before inverse-CDF calls.
_EPS = 1e-10

_POSITIVE_ONLY = {CopulaType.CLAYTON, CopulaType.GUMBEL}


# ---------------------------------------------------------------------------
# Kendall's tau -> copula parameter
# ---------------------------------------------------------------------------


def _debye1_integrand(t: float) -> float:
    # t / (e^t - 1); the removable singularity at 0 is 1, guard large t.
    if -1e-8 < t < 1e-8:
        return 1.0
    if t > 700.0:
        return 0.0
    return float(t / np.expm1(t))


def _frank_theta_from_tau(tau: float) -> float:
    """Invert Kendall's tau -> Frank theta (theta shares tau's sign)."""
    if abs(tau) < 1e-6:
        return 0.0

    def debye1(theta: float) -> float:
        value, _ = integrate.quad(_debye1_integrand, 0.0, theta)
        return float(value / theta)

    def tau_of(theta: float) -> float:
        return 1.0 + 4.0 / theta * (debye1(theta) - 1.0)

    lo, hi = (1e-6, 745.0) if tau > 0 else (-745.0, -1e-6)
    try:
        return float(optimize.brentq(lambda th: tau_of(th) - tau, lo, hi))
    except Exception:
        return max(tau * 10.0, 0.1) if tau > 0 else min(tau * 10.0, -0.1)


def edge_param(family: str, corr: float) -> float:
    """Copula parameter for a tree edge from the spec's correlation value.

    Gaussian/Student-t use ``corr`` directly as the latent correlation; the
    Archimedean families map ``corr`` (a target Kendall's tau) to their
    single parameter theta.
    """
    if family in (CopulaType.GAUSSIAN, CopulaType.STUDENT_T):
        return corr
    if family == CopulaType.CLAYTON:
        # tau -> 1 sends theta -> inf (comonotonicity); clamp just below 1 so a
        # target of exactly 1.0 yields a large-but-finite theta instead of
        # dividing by zero.
        corr = min(corr, 1.0 - 1e-6)
        return 2.0 * corr / (1.0 - corr)
    if family == CopulaType.GUMBEL:
        corr = min(corr, 1.0 - 1e-6)
        return 1.0 / (1.0 - corr)
    if family == CopulaType.FRANK:
        return _frank_theta_from_tau(corr)
    raise ValueError(f"Unknown copula family {family!r}")


# ---------------------------------------------------------------------------
# h-functions and inverse h-functions (conditional CDF / quantile)
# ---------------------------------------------------------------------------


def _hinv_gaussian(w: NDArray[Any], u: NDArray[Any], rho: float) -> NDArray[Any]:
    if abs(rho) < 1e-12:
        return w
    x = stats.norm.ppf(u)
    y = stats.norm.ppf(w)
    return np.asarray(stats.norm.cdf(rho * x + np.sqrt(1.0 - rho**2) * y))


def _h_gaussian(v: NDArray[Any], u: NDArray[Any], rho: float) -> NDArray[Any]:
    if abs(rho) < 1e-12:
        return v
    return np.asarray(
        stats.norm.cdf((stats.norm.ppf(v) - rho * stats.norm.ppf(u)) / np.sqrt(1.0 - rho**2))
    )


def _hinv_student_t(
    w: NDArray[Any], u: NDArray[Any], rho: float, df: int = STUDENT_T_DF
) -> NDArray[Any]:
    # Conditional of a bivariate t is a location-scale t with df + 1 dof.
    x = stats.t.ppf(u, df)
    y = stats.t.ppf(w, df + 1)
    scale = np.sqrt((df + x**2) * (1.0 - rho**2) / (df + 1))
    return np.asarray(stats.t.cdf(rho * x + y * scale, df))


def _h_student_t(
    v: NDArray[Any], u: NDArray[Any], rho: float, df: int = STUDENT_T_DF
) -> NDArray[Any]:
    x = stats.t.ppf(u, df)
    z = stats.t.ppf(v, df)
    scale = np.sqrt((df + x**2) * (1.0 - rho**2) / (df + 1))
    return np.asarray(stats.t.cdf((z - rho * x) / scale, df + 1))


def _hinv_clayton(w: NDArray[Any], u: NDArray[Any], theta: float) -> NDArray[Any]:
    if theta <= 1e-10:
        return w
    return np.asarray(
        (u ** (-theta) * (w ** (-theta / (theta + 1.0)) - 1.0) + 1.0) ** (-1.0 / theta)
    )


def _h_clayton(v: NDArray[Any], u: NDArray[Any], theta: float) -> NDArray[Any]:
    if theta <= 1e-10:
        return v
    return np.asarray(
        u ** (-theta - 1.0) * (u ** (-theta) + v ** (-theta) - 1.0) ** (-1.0 / theta - 1.0)
    )


def _hinv_frank(w: NDArray[Any], u: NDArray[Any], theta: float) -> NDArray[Any]:
    if abs(theta) < 1e-10:
        return w
    em_theta = np.expm1(-theta)  # e^{-theta} - 1
    em_u = np.expm1(-theta * u)  # e^{-theta u} - 1
    return np.asarray(-1.0 / theta * np.log1p(w * em_theta / (1.0 + em_u * (1.0 - w))))


def _h_frank(v: NDArray[Any], u: NDArray[Any], theta: float) -> NDArray[Any]:
    if abs(theta) < 1e-10:
        return v
    em_theta = np.expm1(-theta)
    em_u = np.expm1(-theta * u)
    em_v = np.expm1(-theta * v)
    return np.asarray(np.exp(-theta * u) * em_v / (em_theta + em_u * em_v))


def _h_gumbel(v: NDArray[Any], u: NDArray[Any], theta: float) -> NDArray[Any]:
    lu = -np.log(u)
    lv = -np.log(v)
    a = lu**theta + lv**theta
    c = np.exp(-(a ** (1.0 / theta)))
    return np.asarray(c * (1.0 / u) * lu ** (theta - 1.0) * a ** (1.0 / theta - 1.0))


def _hinv_gumbel(w: NDArray[Any], u: NDArray[Any], theta: float) -> NDArray[Any]:
    # No closed form; h(.|u) is monotone increasing in v, so bisect.
    if theta <= 1.0 + 1e-10:
        return w
    lo = np.full_like(w, _EPS)
    hi = np.full_like(w, 1.0 - _EPS)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        below = _h_gumbel(mid, u, theta) < w
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return np.asarray(0.5 * (lo + hi))


_CopulaFn = Callable[[NDArray[Any], NDArray[Any], float], NDArray[Any]]

_HINV: dict[str, _CopulaFn] = {
    CopulaType.GAUSSIAN: _hinv_gaussian,
    CopulaType.STUDENT_T: _hinv_student_t,
    CopulaType.CLAYTON: _hinv_clayton,
    CopulaType.FRANK: _hinv_frank,
    CopulaType.GUMBEL: _hinv_gumbel,
}

_H: dict[str, _CopulaFn] = {
    CopulaType.GAUSSIAN: _h_gaussian,
    CopulaType.STUDENT_T: _h_student_t,
    CopulaType.CLAYTON: _h_clayton,
    CopulaType.FRANK: _h_frank,
    CopulaType.GUMBEL: _h_gumbel,
}


def hinv(family: str, w: NDArray[Any], u: NDArray[Any], param: float) -> NDArray[Any]:
    """Inverse conditional CDF: draw ``u2`` such that ``h(u2 | u) == w``."""
    w = np.clip(np.asarray(w, dtype=float), _EPS, 1.0 - _EPS)
    u = np.clip(np.asarray(u, dtype=float), _EPS, 1.0 - _EPS)
    try:
        fn = _HINV[family]
    except KeyError:
        raise ValueError(f"Unknown copula family {family!r}") from None
    return np.clip(fn(w, u, param), _EPS, 1.0 - _EPS)


def h(family: str, v: NDArray[Any], u: NDArray[Any], param: float) -> NDArray[Any]:
    """Conditional CDF ``h(v | u) = dC(u, v)/du``. Used for testing hinv."""
    v = np.clip(np.asarray(v, dtype=float), _EPS, 1.0 - _EPS)
    u = np.clip(np.asarray(u, dtype=float), _EPS, 1.0 - _EPS)
    try:
        fn = _H[family]
    except KeyError:
        raise ValueError(f"Unknown copula family {family!r}") from None
    return np.clip(fn(v, u, param), _EPS, 1.0 - _EPS)


# ---------------------------------------------------------------------------
# Vine (Markov-tree) construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VineEdge:
    """One tree edge: ``child`` is drawn conditionally on the already-drawn ``parent``."""

    parent: str
    child: str
    family: str
    param: float


@dataclass(frozen=True)
class VineTree:
    """A connected Markov tree; ``edges`` are ordered parent-before-child."""

    root: str
    edges: tuple[VineEdge, ...]


def build_vine(pairs: list[tuple[str, str, float, str]]) -> list[VineTree]:
    """Build a forest of Markov trees from ``(field1, field2, corr, family)`` pairs.

    Each connected component becomes one tree. Raises if a pair is specified
    twice, if the pairs form a cycle (a 1-truncated vine cannot place a pair
    outside the first tree without conditional-copula parameters the spec does
    not provide), or if a positive-only Archimedean family (Clayton, Gumbel) is
    given a negative correlation.
    """
    parent_of: dict[str, str] = {}

    def find(x: str) -> str:
        parent_of.setdefault(x, x)
        root = x
        while parent_of[root] != root:
            root = parent_of[root]
        while parent_of[x] != root:  # path compression
            parent_of[x], x = root, parent_of[x]
        return root

    seen: set[frozenset[str]] = set()
    adj: dict[str, list[tuple[str, str, float]]] = {}
    nodes: set[str] = set()

    for f1, f2, corr, family in pairs:
        key = frozenset((f1, f2))
        if key in seen:
            raise ValueError(
                f"Correlation pair ({f1}, {f2}) is specified more than once; "
                "each pair may appear at most once."
            )
        seen.add(key)
        if family in _POSITIVE_ONLY and corr < 0:
            raise ValueError(
                f"{family.capitalize()} copula models positive dependence only, but "
                f"pair ({f1}, {f2}) has correlation {corr}. Use a gaussian or frank "
                "copula, or a positive correlation."
            )
        if find(f1) == find(f2):
            raise ValueError(
                f"Correlation pair ({f1}, {f2}) closes a cycle; a vine copula requires "
                "the specified pairs to form a forest (each pair must connect two "
                "previously unconnected fields). Remove a pair from the cycle."
            )
        parent_of[find(f1)] = find(f2)
        param = edge_param(family, corr)
        adj.setdefault(f1, []).append((f2, family, param))
        adj.setdefault(f2, []).append((f1, family, param))
        nodes.update((f1, f2))

    trees: list[VineTree] = []
    visited: set[str] = set()
    for start in sorted(nodes):
        if start in visited:
            continue
        edges: list[VineEdge] = []
        visited.add(start)
        stack = [start]
        while stack:
            node = stack.pop()
            for nbr, family, param in sorted(adj[node]):
                if nbr not in visited:
                    visited.add(nbr)
                    edges.append(VineEdge(node, nbr, family, param))
                    stack.append(nbr)
        trees.append(VineTree(start, tuple(edges)))
    return trees
