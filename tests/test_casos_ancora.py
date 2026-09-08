import numpy as np
import pytest
from lgr_helper import evaluate_lgr


def test_caso_1_2_polos_reais_sem_zero():
    """
    Caso 1 — 2 polos reais, sem zero
    G(s)H(s) = K / [s(s+4)]
    """
    res = evaluate_lgr(NG=[1], DG=[1, 4, 0], NH=[1], DH=[1])

    # Polos: 0 e -4. Zeros: nenhum.
    assert res["nP"] == 2
    assert res["nZ"] == 0
    assert set(np.round(np.real(res["poles"]), 4)) == {0.0, -4.0}

    # Segmento real do LGR: apenas (-4, 0)
    assert res["is_on_real_lgr"](-2) == True
    assert res["is_on_real_lgr"](-5) == False
    assert res["is_on_real_lgr"](1) == False

    # Número de ramos: 2
    assert res["branches"] == 2

    # Assíntotas: 2, centroide = -2, ângulos = ±90°
    assert len(res["angles_A"]) == 2
    assert np.isclose(np.real(res["sigma_A"]), -2.0)
    assert set(np.round(res["angles_A"], 1)) == {90.0, 270.0}

    # Ponto de saída: s = -2, com K = 4
    breakaway = res["breakaway"]
    assert len(breakaway) == 1
    s_val, k_val = breakaway[0]
    assert np.isclose(s_val, -2.0)
    assert np.isclose(k_val, 4.0)

    # Cruzamento com eixo jw: não existe para K > 0 finito
    if res["routh"] and "crossings" in res["routh"]:
        assert len(res["routh"]["crossings"]) == 0


def test_caso_2_2_polos_1_zero():
    """
    Caso 2 — 2 polos reais + 1 zero real
    G(s)H(s) = K(s+2) / [s(s+4)]
    """
    res = evaluate_lgr(NG=[1, 2], DG=[1, 4, 0], NH=[1], DH=[1])

    # Polos: 0 e -4. Zero: -2.
    assert set(np.round(np.real(res["poles"]), 4)) == {0.0, -4.0}
    assert set(np.round(np.real(res["zeros"]), 4)) == {-2.0}

    # Segmentos reais do LGR: (-2, 0) e (-∞, -4)
    assert res["is_on_real_lgr"](-1) == True
    assert res["is_on_real_lgr"](-3) == False
    assert res["is_on_real_lgr"](-5) == True

    # Número de ramos: 2
    assert res["branches"] == 2

    # Assíntotas: 1, centroide = -2, ângulo = 180°
    assert len(res["angles_A"]) == 1
    assert np.isclose(np.real(res["sigma_A"]), -2.0)
    assert np.isclose(res["angles_A"][0], 180.0)

    # Ponto de saída/entrada: não existe sobre o eixo real
    assert len(res["breakaway"]) == 0

    # Cruzamento com eixo jw: não existe
    if res["routh"] and "crossings" in res["routh"]:
        assert len(res["routh"]["crossings"]) == 0

    # Ponto de teste: s0 = -1 -> deve fechar 180° e dar K=3
    pt_res = res["test_point"](-1.0 + 0j)
    assert np.isclose(pt_res["normalized_angle"], 180.0)
    assert pt_res["is_lgr"] == True
    assert np.isclose(pt_res["K"], 3.0)


def test_caso_3_3_polos_reais_com_cruzamento():
    """
    Caso 3 — 3 polos reais, sem zero
    G(s)H(s) = K / [s(s+2)(s+4)]
    DG(s) = s^3 + 6s^2 + 8s
    """
    res = evaluate_lgr(NG=[1], DG=[1, 6, 8, 0], NH=[1], DH=[1])

    # Segmentos reais do LGR: (-2, 0) e (-∞, -4)
    assert res["is_on_real_lgr"](-1) == True
    assert res["is_on_real_lgr"](-3) == False
    assert res["is_on_real_lgr"](-5) == True

    # Assíntotas: 3, centroide = -2, ângulos = 60, 180, 300
    assert len(res["angles_A"]) == 3
    assert np.isclose(np.real(res["sigma_A"]), -2.0)
    assert set(np.round(res["angles_A"], 1)) == {60.0, 180.0, 300.0}

    # Ponto de saída: s ≈ -0.845, com K ≈ 3.08
    breakaway = res["breakaway"]
    assert len(breakaway) == 1
    s_val, k_val = breakaway[0]
    assert np.isclose(s_val, -0.845, atol=1e-2)
    assert np.isclose(k_val, 3.08, atol=1e-2)

    # Cruzamento com eixo jw: existe K crítico = 48, w = ±2.828
    routh = res["routh"]
    assert routh is not None
    cross_data = routh.get("crossings_data", [])
    assert len(cross_data) > 0

    # Find the one corresponding to critical K
    found_cross = False
    for cd in cross_data:
        if np.isclose(cd["k_crit"], 48.0, atol=1e-2):
            omegas = cd["omegas"]
            if any(np.isclose(w, 2.828, atol=1e-2) for w in omegas):
                found_cross = True
                break
    assert found_cross == True
