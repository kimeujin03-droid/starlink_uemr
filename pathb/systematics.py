from __future__ import annotations

from typing import Literal

import numpy as np

SystematicsMode = Literal["none", "reflection", "coupling", "both"]


def _freq_ripple(freqs_hz: np.ndarray, eps: float, tau_ns: float, phi: float) -> np.ndarray:
    tau_s = float(tau_ns) * 1e-9
    return np.asarray(eps, dtype=float) * np.exp(
        2j * np.pi * np.asarray(freqs_hz, dtype=float) * tau_s + 1j * float(phi)
    )


def apply_cable_reflection(
    vis_tf: np.ndarray,
    freqs_hz: np.ndarray,
    eps: float = 0.03,
    tau_ns: float = 600.0,
    phi: float = 0.0,
) -> np.ndarray:
    """Apply a weak multiplicative cable-reflection ripple."""
    ripple = 1.0 + _freq_ripple(freqs_hz, eps=eps, tau_ns=tau_ns, phi=phi)
    return np.asarray(vis_tf, dtype=complex) * ripple[None, :]


def apply_delayed_coupling(
    vis_tf: np.ndarray,
    freqs_hz: np.ndarray,
    eps: float = 0.01,
    tau_ns: float = 800.0,
    phi: float = 0.0,
) -> np.ndarray:
    """Apply a self-coupled delayed echo as a first-order leakage proxy."""
    echo = _freq_ripple(freqs_hz, eps=eps, tau_ns=tau_ns, phi=phi)
    return np.asarray(vis_tf, dtype=complex) * (1.0 + echo[None, :])


def apply_systematics(
    vis_tf: np.ndarray,
    freqs_hz: np.ndarray,
    mode: SystematicsMode = "none",
    reflection_eps: float = 0.03,
    reflection_tau_ns: float = 600.0,
    reflection_phi: float = 0.0,
    coupling_eps: float = 0.01,
    coupling_tau_ns: float = 800.0,
    coupling_phi: float = 0.0,
) -> np.ndarray:
    """Apply the selected phenomenological high-delay perturbation."""
    mode = str(mode).lower()
    if mode == "none":
        return np.asarray(vis_tf, dtype=complex).copy()
    if mode == "reflection":
        return apply_cable_reflection(
            vis_tf,
            freqs_hz,
            eps=reflection_eps,
            tau_ns=reflection_tau_ns,
            phi=reflection_phi,
        )
    if mode == "coupling":
        return apply_delayed_coupling(
            vis_tf,
            freqs_hz,
            eps=coupling_eps,
            tau_ns=coupling_tau_ns,
            phi=coupling_phi,
        )
    if mode == "both":
        x = apply_cable_reflection(
            vis_tf,
            freqs_hz,
            eps=reflection_eps,
            tau_ns=reflection_tau_ns,
            phi=reflection_phi,
        )
        return apply_delayed_coupling(
            x,
            freqs_hz,
            eps=coupling_eps,
            tau_ns=coupling_tau_ns,
            phi=coupling_phi,
        )
    raise ValueError(f"Unknown systematics mode: {mode}")
