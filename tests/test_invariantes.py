import numpy as np
import pytest
from lgr_helper import evaluate_lgr


# --- Geração de Sistemas (Categoria B) ---
def generate_systems():
    systems = []

    def add(cat, ng, dg, nh=[1], dh=[1]):
        systems.append({"cat": cat, "ng": ng, "dg": dg, "nh": nh, "dh": dh})

    # 1. Sem zeros finitos, com 1, 2, 3 e 4 polos reais distintos
    add(1, [1], [1, 2])  # 1 polo
    add(1, [1], [1, 3, 2])  # 2 polos
    add(1, [1], [1, 6, 11, 6])  # 3 polos
    add(1, [1], [1, 10, 35, 50, 24])  # 4 polos

    # 2. Com zeros finitos, nZ < nP
    add(2, [1, 2], [1, 6, 11, 6])

    # 3. nZ == nP (sem assíntotas)
    add(3, [1, 4, 3], [1, 6, 8])

    # 4. Polos complexos conjugados
    add(4, [1], [1, 2, 5])  # polos -1 ± 2j

    # 5. Zeros complexos conjugados
    add(5, [1, 2, 5], [1, 6, 11, 6, 0])

    # 6. Polos repetidos
    add(6, [1], [1, 4, 4])  # (s+2)^2
    add(6, [1], [1, 0, 0, 0])  # s^3

    # 7. Com e sem polo na origem
    add(7, [1], [1, 3, 2, 0])  # com polo na origem

    # 8. Polo no semiplano direito
    add(8, [1], [1, -2, -3])  # (s-3)(s+1)

    # 9. H(s) não unitário
    add(9, [1], [1, 2], nh=[1], dh=[1, 10])

    # 10. Sem ponto de saída real
    add(10, [1, 2], [1, 4, 0])

    # 11. Múltiplos pontos de saída
    add(11, [1], [1, 10, 35, 50, 24])

    # 12. Estável para todo K>0
    add(12, [1], [1, 3, 2])

    # 13. Cruza eixo jw
    add(13, [1], [1, 6, 8, 0])

    return systems


SYSTEMS = generate_systems()


@pytest.mark.parametrize("sys_data", SYSTEMS)
def test_invariante_1_polos_zeros(sys_data):
    # Polos/zeros do passo 3 = raízes de DG*DH e NG*NH
    res = evaluate_lgr(sys_data["ng"], sys_data["dg"], sys_data["nh"], sys_data["dh"])

    exp_poles = np.roots(np.polymul(sys_data["dg"], sys_data["dh"]))
    exp_zeros = np.roots(np.polymul(sys_data["ng"], sys_data["nh"]))

    assert len(res["poles"]) == len(exp_poles)
    assert len(res["zeros"]) == len(exp_zeros)


@pytest.mark.parametrize("sys_data", SYSTEMS)
def test_invariante_2_criterio_angulo_segmento(sys_data):
    res = evaluate_lgr(sys_data["ng"], sys_data["dg"], sys_data["nh"], sys_data["dh"])
    rc = res["real_critical"]

    for i in range(len(rc) - 1):
        midpoint = (rc[i] + rc[i + 1]) / 2.0
        is_lgr_geometric = res["is_on_real_lgr"](midpoint)
        pt_res = res["test_point"](midpoint + 0j)

        # Só testa se não tiver erro de divisão por zero ou exatamente em cima da raiz
        if is_lgr_geometric:
            assert pt_res["is_lgr"] == True


@pytest.mark.parametrize("sys_data", SYSTEMS)
def test_invariante_3_ponto_saida_raiz_dupla(sys_data):
    res = evaluate_lgr(sys_data["ng"], sys_data["dg"], sys_data["nh"], sys_data["dh"])
    for s_val, k_val in res["breakaway"]:
        # P(s) = D(s) + K*N(s)
        D_s = np.poly1d(res["D_coeffs"])
        N_s = np.poly1d(res["N_coeffs"])
        P_s = D_s + k_val * N_s
        dP_s = np.polyder(P_s)

        assert np.isclose(P_s(s_val), 0.0, atol=1e-3)
        assert np.isclose(dP_s(s_val), 0.0, atol=1e-3)


@pytest.mark.parametrize("sys_data", SYSTEMS)
def test_invariante_4_K_fecha_laco(sys_data):
    res = evaluate_lgr(sys_data["ng"], sys_data["dg"], sys_data["nh"], sys_data["dh"])

    # Test point on LGR
    rc = res["real_critical"]
    s0 = None
    for i in range(len(rc) - 1):
        midpoint = (rc[i] + rc[i + 1]) / 2.0
        if res["is_on_real_lgr"](midpoint):
            s0 = midpoint
            break

    if s0 is not None:
        pt_res = res["test_point"](s0 + 0j)
        k_val = pt_res["K"]

        D_s = np.poly1d(res["D_coeffs"])
        N_s = np.poly1d(res["N_coeffs"])
        P_s = D_s + k_val * N_s
        assert np.isclose(P_s(s0), 0.0, atol=1e-3)


@pytest.mark.parametrize("sys_data", SYSTEMS)
def test_invariante_5_cruzamento_jw(sys_data):
    res = evaluate_lgr(sys_data["ng"], sys_data["dg"], sys_data["nh"], sys_data["dh"])
    routh = res["routh"]
    if routh and "crossings_data" in routh:
        for cd in routh["crossings_data"]:
            k_crit = cd["k_crit"]
            omegas = cd["omegas"]
            if k_crit > 0:
                for w in omegas:
                    s_val = w * 1j
                    D_s = np.poly1d(res["D_coeffs"])
                    N_s = np.poly1d(res["N_coeffs"])
                    P_s = D_s + k_crit * N_s
                    assert np.isclose(P_s(s_val), 0.0, atol=1e-3)


@pytest.mark.parametrize("sys_data", SYSTEMS)
def test_invariante_6_angulo_partida_chegada(sys_data):
    res = evaluate_lgr(sys_data["ng"], sys_data["dg"], sys_data["nh"], sys_data["dh"])
    eps = 1e-3

    for cp, angle in res["departure_angles"].items():
        s_test = cp + eps * np.exp(1j * np.radians(angle))
        pt_res = res["test_point"](s_test)
        assert pt_res["is_lgr"] == True

    for cz, angle in res["arrival_angles"].items():
        s_test = cz + eps * np.exp(1j * np.radians(angle))
        pt_res = res["test_point"](s_test)
        assert pt_res["is_lgr"] == True


@pytest.mark.parametrize("sys_data", SYSTEMS)
def test_invariante_8_contagem_ramos(sys_data):
    res = evaluate_lgr(sys_data["ng"], sys_data["dg"], sys_data["nh"], sys_data["dh"])
    assert res["branches"] == max(res["nP"], res["nZ"])


@pytest.mark.parametrize("sys_data", SYSTEMS)
def test_invariante_9_simetria(sys_data):
    res = evaluate_lgr(sys_data["ng"], sys_data["dg"], sys_data["nh"], sys_data["dh"])

    # Testar que raizes de polinômios com coefs reais são simétricas
    D_s = np.poly1d(res["D_coeffs"])
    N_s = np.poly1d(res["N_coeffs"])

    for K in [1.0, 10.0, 100.0]:
        P_s = D_s + K * N_s
        roots = np.roots(P_s)

        for r in roots:
            if abs(np.imag(r)) > 1e-5:
                conj = np.conj(r)
                # Ensure conj exists in roots
                found = any(np.isclose(conj, rr, atol=1e-3) for rr in roots)
                assert found

@pytest.mark.parametrize("sys_data", SYSTEMS)
def test_invariante_7_assintotas(sys_data):
    res = evaluate_lgr(sys_data["ng"], sys_data["dg"], sys_data["nh"], sys_data["dh"])
    
    if res["nP"] > res["nZ"]:
        K_large = 1e6
        D_s = np.poly1d(res["D_coeffs"])
        N_s = np.poly1d(res["N_coeffs"])
        P_s = D_s + K_large * N_s
        roots_large = np.roots(P_s)
        
        # Filtrar as raízes que convergiram para os zeros (que não vão pro infinito)
        roots_to_inf = []
        for r in roots_large:
            is_close_to_zero = False
            for z in res["zeros"]:
                if np.abs(r - z) < 1.0: # Com K=1e6, se a distância for menor que 1, tá indo pro zero
                    is_close_to_zero = True
                    break
            if not is_close_to_zero:
                roots_to_inf.append(r)
                
        # O número de raízes indo pro infinito deve ser nP - nZ
        assert len(roots_to_inf) == res["nP"] - res["nZ"]
        
        # Verificar o ângulo de cada raiz em relação ao centroide
        sigma_A = res["sigma_A"]
        calculated_angles = []
        for r in roots_to_inf:
            angle = np.degrees(np.angle(r - sigma_A))
            if angle < 0:
                angle += 360
            calculated_angles.append(angle)
            
        # Comparar com os ângulos das assíntotas
        expected_angles = res["angles_A"]
        
        # Para cada raiz, garantir que existe UMA assíntota próxima
        for calc_ang in calculated_angles:
            diffs = [min(abs(calc_ang - exp_ang), 360 - abs(calc_ang - exp_ang)) for exp_ang in expected_angles]
            assert min(diffs) < 2.0 # tolerância de 2 graus

@pytest.mark.parametrize("sys_data", SYSTEMS)
def test_invariante_5_cruzamento_jw_mut5(sys_data):
    res = evaluate_lgr(sys_data["ng"], sys_data["dg"], sys_data["nh"], sys_data["dh"])
    routh = res["routh"]
    if routh and "crossings_data" in routh:
        for cd in routh["crossings_data"]:
            if cd["k_crit"] > 0 and len(cd["omegas"]) > 0:
                D_s = np.poly1d(res["D_coeffs"])
                N_s = np.poly1d(res["N_coeffs"])
                P_s = D_s + cd["k_crit"] * N_s
                rts = np.roots(P_s)
                
                # Check that for each calculated w, w*1j is actually a true root
                for w in cd["omegas"]:
                    found = any(np.isclose(w*1j, r, atol=1e-3) or np.isclose(-w*1j, r, atol=1e-3) for r in rts)
                    assert found, f"w={w} is not a true root"
