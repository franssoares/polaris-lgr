import re
content = open('lgr_math.py', encoding='utf-8').read()

helpers = '''from fractions import Fraction

def format_frac(val, tol=1e-5):
    if abs(val - round(val)) < tol:
        return f"{int(round(val))}"
    frac = Fraction(float(val)).limit_denominator(1000)
    if frac.denominator == 1:
        return f"{frac.numerator}"
    if frac.numerator < 0:
        return f"-{abs(frac.numerator)}/{frac.denominator}"
    return f"{frac.numerator}/{frac.denominator}"

'''

content = content.replace('import numpy as np\n', 'import numpy as np\n' + helpers)

content = re.sub(r'\{c_abs:g\}', '{format_frac(c_abs)}', content)
content = re.sub(r'\{r_val:g\}', '{format_frac(r_val)}', content)
content = re.sub(r'\{-r_val:g\}', '{format_frac(-r_val)}', content)
content = re.sub(r'\{real_part:g\}', '{format_frac(real_part)}', content)
content = re.sub(r'\{abs\(imag_part\):g\}', '{format_frac(abs(imag_part))}', content)

open('lgr_math.py', 'w', encoding='utf-8').write(content)
