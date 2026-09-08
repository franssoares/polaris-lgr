import re
import numpy as np

content = open('app.py', encoding='utf-8').read()

helpers = '''from fractions import Fraction

def format_frac(val, tol=1e-5):
    if abs(val - round(val)) < tol:
        return f"{int(round(val))}"
    frac = Fraction(float(val)).limit_denominator(1000)
    if frac.denominator == 1:
        return f"{frac.numerator}"
    if frac.numerator < 0:
        return rf"-\\frac{{{abs(frac.numerator)}}}{{{frac.denominator}}}"
    return rf"\\frac{{{frac.numerator}}}{{{frac.denominator}}}"

def format_complex_frac(val, tol=1e-5):
    r = float(np.real(val))
    i = float(np.imag(val))
    
    if abs(i) < tol:
        return format_frac(r, tol)
    
    i_frac = Fraction(abs(i)).limit_denominator(1000)
    if i_frac.numerator == 1 and i_frac.denominator == 1:
        i_str = "j"
    elif i_frac.denominator == 1:
        i_str = f"{i_frac.numerator}j"
    else:
        i_str = rf"\\frac{{{i_frac.numerator}}}{{{i_frac.denominator}}}j"
        
    sign = "+" if i > 0 else "-"
    
    if abs(r) < tol:
        return f"{'-' if i < 0 else ''}{i_str}"
        
    return rf"{format_frac(r, tol)} {sign} {i_str}"

def main():'''

content = content.replace('def main():', helpers)

content = re.sub(r'\{np\.real\(sigma_A\):\.3g\}', '{format_frac(np.real(sigma_A))}', content)
content = re.sub(r'\{cross_y:\.3g\}j', '{format_frac(cross_y)}j', content)
content = re.sub(r'\{cross_y:\.3g\}', '{format_frac(cross_y)}', content)
content = re.sub(r'\{k_val:\.3g\}', '{format_frac(k_val)}', content)
content = re.sub(r'\{k_crit:\.3g\}', '{format_frac(k_crit)}', content)
content = re.sub(r'\{w:\.3g\}', '{format_frac(w)}', content)
content = re.sub(r'\{np\.real\(cp\):\.3g\}', '{format_frac(np.real(cp))}', content)
content = re.sub(r'\{abs\(np\.imag\(cp\)\):\.3g\}', '{format_frac(abs(np.imag(cp)))}', content)
content = re.sub(r'\{np\.real\(cz\):\.3g\}', '{format_frac(np.real(cz))}', content)
content = re.sub(r'\{abs\(np\.imag\(cz\)\):\.3g\}', '{format_frac(abs(np.imag(cz)))}', content)
content = re.sub(r'\{np\.real\(s0\):\.3g\}', '{format_frac(np.real(s0))}', content)
content = re.sub(r'\{abs\(np\.imag\(s0\)\):\.3g\}', '{format_frac(abs(np.imag(s0)))}', content)
content = re.sub(r'\{np\.real\(c\):\.3g\}', '{format_frac(np.real(c))}', content)
content = re.sub(r'\{abs\(np\.imag\(c\)\):\.3g\}', '{format_frac(abs(np.imag(c)))}', content)
content = re.sub(r'\{dist_z\[i\]:\.3g\}', '{format_frac(dist_z[i])}', content)
content = re.sub(r'\{dist_p\[i\]:\.3g\}', '{format_frac(dist_p[i])}', content)
content = re.sub(r'\{d:\.3g\}', '{format_frac(d)}', content)
content = re.sub(r'\{K_val:\.3g\}', '{format_frac(K_val)}', content)
content = re.sub(r'\{np\.real\(p\):g\}', '{format_frac(np.real(p))}', content)
content = re.sub(r'\{np\.real\(z\):g\}', '{format_frac(np.real(z))}', content)

content = re.sub(r'\{angle:\.1f\}', '{format_frac(angle)}', content)
content = re.sub(r'\{angle_norm:\.1f\}', '{format_frac(angle_norm)}', content)
content = re.sub(r'\{sum_p:\.1f\}', '{format_frac(sum_p)}', content)
content = re.sub(r'\{sum_z:\.1f\}', '{format_frac(sum_z)}', content)
content = re.sub(r'\{a:\.1f\}', '{format_frac(a)}', content)
content = re.sub(r'\{angles_z\[i\]:\.1f\}', '{format_frac(angles_z[i])}', content)
content = re.sub(r'\{angles_p\[i\]:\.1f\}', '{format_frac(angles_p[i])}', content)
content = re.sub(r'\{total_angle:\.1f\}', '{format_frac(total_angle)}', content)
content = re.sub(r'\{normalized_angle:\.1f\}', '{format_frac(normalized_angle)}', content)

content = re.sub(r'\{d_val:g\}', '{format_frac(d_val)}', content)
content = re.sub(r'\{abs_n:g\}', '{format_frac(abs_n)}', content)
content = re.sub(r'\{K_scale:g\}', '{format_frac(K_scale)}', content)
content = re.sub(r'\{np\.real\(r_rounded\):g\}', '{format_frac(np.real(r_rounded))}', content)
content = re.sub(r'\{abs\(np\.imag\(r_rounded\)\):g\}', '{format_frac(abs(np.imag(r_rounded)))}', content)
content = re.sub(r'\{end:g\}', '{format_frac(end)}', content)
content = re.sub(r'\{start:g\}', '{format_frac(start)}', content)

# Remove the complex old string building loops
content = re.sub(
    r'sum_p_str = " \+ "\.join\([\s\S]*?for p in poles\s*\]\s*\)',
    'sum_p_str = " + ".join([f"({format_complex_frac(p)})" for p in poles])',
    content
)
content = re.sub(
    r'sum_z_str = " \+ "\.join\([\s\S]*?for z in zeros\s*\]\s*\)',
    'sum_z_str = " + ".join([f"({format_complex_frac(z)})" for z in zeros])',
    content
)

open('app.py', 'w', encoding='utf-8').write(content)
