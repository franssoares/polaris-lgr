import pytest
import numpy as np
from lgr_helper import evaluate_lgr

# In Python, we can test that the helper or math functions handle bad data predictably
# or raise appropriate exceptions without crashing the interpreter.


def test_nZ_maior_que_nP():
    # nZ > nP (mais zeros que polos)
    # nZ = 2 (s^2 + 3s + 2), nP = 1 (s + 1)
    # app.py should handle this mathematically or reject it.
    # Our lgr_math functions should not crash hard, they might return no asymptotes.
    res = evaluate_lgr(NG=[1, 3, 2], DG=[1, 1], NH=[1], DH=[1])
    assert res["nP"] == 1
    assert res["nZ"] == 2
    assert res["sigma_A"] is None
    assert len(res["angles_A"]) == 0
    assert res["branches"] == 2


def test_coeficientes_zero():
    # DG = [0]
    # np.roots([0]) is empty.
    # Should not crash Python, but should handle empty poles.
    res = evaluate_lgr(NG=[1], DG=[0], NH=[1], DH=[1])
    assert res["nP"] == 0
    assert len(res["poles"]) == 0
    assert res["sigma_A"] is None


def test_cancelamento_polo_zero():
    # Categoria 15: Cancelamento exato
    # G(s) = (s+3) / [(s+1)(s+3)]
    # NG = [1, 3], DG = [1, 4, 3]
    res = evaluate_lgr(NG=[1, 3], DG=[1, 4, 3], NH=[1], DH=[1])

    # Mathematical behaviour: the pole at -3 and zero at -3 coincide.
    # They will be detected as a pole and a zero.
    assert set(np.round(np.real(res["poles"]), 4)) == {-1.0, -3.0}
    assert set(np.round(np.real(res["zeros"]), 4)) == {-3.0}

    # Routh might behave weirdly, but shouldn't crash
    # Breakaway points: derivative of K will have roots, but they might cancel.
    assert "breakaway" in res
