import numpy as np
import sympy as sp
from typing import List, Tuple, Dict, Any
import sys
import os

# Add the parent directory to sys.path to import lgr_math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lgr_math import find_breakaway_points, build_routh_hurwitz, simulate_root_locus


def evaluate_lgr(
    NG: List[float], DG: List[float], NH: List[float], DH: List[float]
) -> Dict[str, Any]:
    """
    Executes the mathematical logic of the 12 steps for testing purposes.
    """
    res = {}

    # Passo 1 & 2: Função de Transferência de Malha Aberta
    N_coeffs = np.polymul(NG, NH)
    D_coeffs = np.polymul(DG, DH)
    res["N_coeffs"] = N_coeffs
    res["D_coeffs"] = D_coeffs

    # Passo 3: Polos e Zeros
    poles = np.roots(D_coeffs)
    zeros = np.roots(N_coeffs)
    nP = len(poles)
    nZ = len(zeros)
    res["poles"] = poles
    res["zeros"] = zeros
    res["nP"] = nP
    res["nZ"] = nZ

    # Passo 4: Eixo Real
    real_poles = [np.real(p) for p in poles if abs(np.imag(p)) < 1e-5]
    real_zeros = [np.real(z) for z in zeros if abs(np.imag(z)) < 1e-5]
    real_critical = sorted(real_poles + real_zeros, reverse=True)
    res["real_critical"] = real_critical

    def on_lgr(x):
        return sum(1 for c in real_critical if c > x) % 2 == 1

    res["is_on_real_lgr"] = on_lgr

    # Passo 5: Ramos
    res["branches"] = max(nP, nZ)

    # Passo 6: Simetria (Always symmetric for real coeffs, implicitly true)

    # Passo 7: Assíntotas
    if nP > nZ:
        sigma_A = (sum(poles) - sum(zeros)) / (nP - nZ)
        angles_A = [(2 * q + 1) * 180 / (nP - nZ) for q in range(nP - nZ)]
    else:
        sigma_A = None
        angles_A = []
    res["sigma_A"] = sigma_A
    res["angles_A"] = angles_A

    # Passo 8: Pontos de Saída/Entrada
    valid_breakaway = find_breakaway_points(list(N_coeffs), list(D_coeffs))
    res["breakaway"] = valid_breakaway

    # Passo 9: Cruzamento com eixo imaginário
    if len(D_coeffs) > 1:  # at least degree 1
        try:
            routh = build_routh_hurwitz(list(N_coeffs), list(D_coeffs))
            res["routh"] = routh
        except Exception:
            res["routh"] = None
    else:
        res["routh"] = None

    # Passo 10: Ângulos de Partida e Chegada
    departure_angles = {}
    arrival_angles = {}
    complex_poles = [p for p in poles if abs(np.imag(p)) > 1e-5]
    complex_zeros = [z for z in zeros if abs(np.imag(z)) > 1e-5]

    for cp in complex_poles:
        m = sum(1 for p in poles if np.isclose(cp, p))
        sum_p = sum(
            np.degrees(np.angle(cp - p)) for p in poles if not np.isclose(cp, p)
        )
        sum_z = sum(np.degrees(np.angle(cp - z)) for z in zeros)
        # return the first branch for simplicity in testing
        angle_dep = ((1) * 180 - sum_p + sum_z) / m
        angle_norm = (angle_dep + 180) % 360 - 180
        departure_angles[np.round(cp, 4)] = angle_norm

    for cz in complex_zeros:
        m = sum(1 for z in zeros if np.isclose(cz, z))
        sum_z = sum(
            np.degrees(np.angle(cz - z)) for z in zeros if not np.isclose(cz, z)
        )
        sum_p = sum(np.degrees(np.angle(cz - p)) for p in poles)
        angle_arr = ((1) * 180 - sum_z + sum_p) / m
        angle_norm = (angle_arr + 180) % 360 - 180
        arrival_angles[np.round(cz, 4)] = angle_norm

    res["departure_angles"] = departure_angles
    res["arrival_angles"] = arrival_angles

    # Passo 11 & 12: Critério de ângulo e ganho K (we will provide a helper function for any s0)
    def test_point(s0):
        angles_z = [np.degrees(np.angle(s0 - z)) for z in zeros]
        angles_p = [np.degrees(np.angle(s0 - p)) for p in poles]
        total_angle = sum(angles_z) - sum(angles_p)
        normalized_angle = total_angle % 360
        if normalized_angle < 0:
            normalized_angle += 360
        is_lgr = np.isclose(normalized_angle, 180, atol=5.0)

        dist_z = [abs(s0 - z) for z in zeros]
        dist_p = [abs(s0 - p) for p in poles]
        # revert mut 7
        K_val = (np.prod(dist_p) if dist_p else 1.0) / (
            np.prod(dist_z) if dist_z else 1.0
        )

        return {
            "total_angle": total_angle,
            "normalized_angle": normalized_angle,
            "is_lgr": is_lgr,
            "K": K_val,
        }

    res["test_point"] = test_point

    return res
