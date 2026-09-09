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

    # Verifica passos da tabela de Routh
    assert "routh_steps" in routh
    assert len(routh["routh_steps"]) >= 2

    # Find the one corresponding to critical K and verify proofs
    found_cross = False
    for cd in cross_data:
        if np.isclose(cd["k_crit"], 48.0, atol=1e-2):
            omegas = cd["omegas"]
            if any(np.isclose(w, 2.828, atol=1e-2) for w in omegas):
                found_cross = True
                # Verifica comprovação por substituição direta P(jw) = 0
                direct_proofs = cd.get("direct_proofs", [])
                assert len(direct_proofs) > 0
                for dp in direct_proofs:
                    assert np.isclose(dp["real_sum"], 0.0, atol=1e-3)
                    assert np.isclose(dp["imag_sum"], 0.0, atol=1e-3)

                # Verifica comprovação por critério vetorial (fase = ±180° e ganho = 48)
                vector_proofs = cd.get("vector_proofs", [])
                assert len(vector_proofs) > 0
                for vp in vector_proofs:
                    assert np.isclose(abs(vp["phase_norm"]), 180.0, atol=1e-2)
                    assert np.isclose(vp["k_calc"], 48.0, atol=1e-2)
                break
    assert found_cross == True


def test_passo_10_angulos_partida_chegada():
    from lgr_math import calculate_departure_arrival_angles

    # Caso A: Sistema com polos complexos conjugados
    # G(s) = K(s+3) / [s(s^2 + 2s + 2)] => polos: 0, -1±1j, zero: -3
    poles_a = np.roots([1, 2, 2, 0])
    zeros_a = np.array([-3.0])
    res_a = calculate_departure_arrival_angles(poles_a, zeros_a)

    assert res_a["has_complex"] == True
    assert len(res_a["pole_details"]) == 2
    assert len(res_a["zero_details"]) == 0

    # Polo superior (-1 + 1j): theta_p ≈ -18.43°
    p_sup = [pd for pd in res_a["pole_details"] if np.imag(pd["pole"]) > 0][0]
    first_branch = p_sup["branches"][0]
    assert np.isclose(first_branch["norm"], -18.435, atol=1e-2)
    assert first_branch["proof"]["is_valid"] == True
    assert np.isclose(abs(first_branch["proof"]["phase_norm"]), 180.0, atol=1e-1)
    assert first_branch["proof"]["k_test"] > 0

    # Polo inferior (-1 - 1j): theta_p ≈ +18.43° (simétrico)
    p_inf = [pd for pd in res_a["pole_details"] if np.imag(pd["pole"]) < 0][0]
    assert np.isclose(p_inf["branches"][0]["norm"], 18.435, atol=1e-2)
    assert p_inf["branches"][0]["proof"]["is_valid"] == True

    # Caso B: Sistema com zeros complexos conjugados
    # G(s) = K(s^2 + 2s + 5) / [s(s+1)(s+2)(s+3)] => zeros: -1±2j
    poles_b = np.array([0.0, -1.0, -2.0, -3.0])
    zeros_b = np.roots([1, 2, 5])
    res_b = calculate_departure_arrival_angles(poles_b, zeros_b)

    assert res_b["has_complex"] == True
    assert len(res_b["zero_details"]) == 2
    z_sup = [zd for zd in res_b["zero_details"] if np.imag(zd["zero"]) > 0][0]
    assert np.isclose(z_sup["branches"][0]["norm"], 45.0, atol=1e-2)
    assert z_sup["branches"][0]["proof"]["is_valid"] == True
    assert np.isclose(abs(z_sup["branches"][0]["proof"]["phase_norm"]), 180.0, atol=1e-1)

    # Caso C: Sistema puramente real
    poles_c = np.array([0.0, -4.0])
    zeros_c = np.array([])
    res_c = calculate_departure_arrival_angles(poles_c, zeros_c)
    assert res_c["has_complex"] == False


def test_passo_11_12_criterio_angulo_e_calculo_k():
    from lgr_math import evaluate_test_point_details

    # 1. Ponto real sobre o LGR: s0 = -1 para G(s)H(s) = K(s+2)/[s(s+4)]
    poles = np.array([0.0, -4.0])
    zeros = np.array([-2.0])
    D_coeffs = [1, 4, 0]
    N_coeffs = [1, 2]

    det1 = evaluate_test_point_details(-1.0 + 0j, poles, zeros, D_coeffs, N_coeffs)
    assert det1["is_lgr"] == True
    assert np.isclose(det1["normalized_angle"], 180.0)
    assert len(det1["vecs_p"]) == 2
    assert len(det1["vecs_z"]) == 1
    assert np.isclose(det1["K"], 3.0)
    assert np.isclose(det1["D_s0"], -3.0)
    assert np.isclose(det1["N_s0"], 1.0)
    assert np.isclose(det1["P_s0"], 0.0, atol=1e-5)
    assert np.isclose(det1["residual"], 0.0, atol=1e-5)
    assert det1["proof_verified"] == True

    # 2. Ponto complexo sobre o LGR: s0 = -2 + 2j para G(s)H(s) = K/[s(s+4)]
    poles2 = np.array([0.0, -4.0])
    zeros2 = np.array([])
    D2 = [1, 4, 0]
    N2 = [1]
    det2 = evaluate_test_point_details(-2.0 + 2.0j, poles2, zeros2, D2, N2)
    assert det2["is_lgr"] == True
    assert np.isclose(det2["normalized_angle"], 180.0)
    assert np.isclose(det2["K"], 8.0)
    assert np.isclose(det2["P_s0"], 0.0, atol=1e-5)
    assert det2["proof_verified"] == True

    # 3. Ponto fora do LGR: s0 = -3 para G(s)H(s) = K(s+2)/[s(s+4)]
    det3 = evaluate_test_point_details(-3.0 + 0j, poles, zeros, D_coeffs, N_coeffs)
    assert det3["is_lgr"] == False
    assert np.isclose(det3["normalized_angle"], 0.0)
    assert np.isclose(det3["angular_deficiency_signed"], 180.0)
    assert np.isclose(det3["residual"], 6.0)
    assert np.isclose(det3["K_req"], -3.0)  # requer ganho negativo

    # 4. Ponto coincidente com polo
    det_pole = evaluate_test_point_details(0.0 + 0j, poles, zeros, D_coeffs, N_coeffs)
    assert det_pole["is_pole"] == True
    assert det_pole["is_lgr"] == True
    assert np.isclose(det_pole["K"], 0.0)

    # 5. Ponto coincidente com zero
    det_zero = evaluate_test_point_details(-2.0 + 0j, poles, zeros, D_coeffs, N_coeffs)
    assert det_zero["is_zero"] == True
    assert det_zero["is_lgr"] == True
    assert np.isinf(det_zero["K"])

