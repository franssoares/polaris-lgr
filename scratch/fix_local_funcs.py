import sys

def modify():
    with open('app.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    format_root_lines = [
        'def format_root(r):\n',
        '    r_rounded = np.round(r, 4)\n',
        '    if abs(np.imag(r_rounded)) < 1e-5:\n',
        '        return f"{format_frac(np.real(r_rounded))}"\n',
        '    else:\n',
        '        sign = "+" if np.imag(r_rounded) > 0 else "-"\n',
        '        return f"{format_frac(np.real(r_rounded))} {sign} {format_frac(abs(np.imag(r_rounded)))}j"\n',
        '\n'
    ]

    get_block_latex_lines = [
        'def get_block_latex(n_lat, d_lat, is_g=False):\n',
        '    if is_g:\n',
        '        if n_lat == "1" and d_lat == "1": return "1"\n',
        '        if d_lat == "1": return n_lat\n',
        '        return r"\\frac{" + n_lat + r"}{" + d_lat + r"}"\n',
        '    else:\n',
        '        if n_lat == "1" and d_lat == "1": return ""\n',
        '        if d_lat == "1": return r"\\cdot " + n_lat\n',
        '        return r"\\cdot \\frac{" + n_lat + r"}{" + d_lat + r"}"\n',
        '\n'
    ]

    format_eval_poly_lines = [
        'def format_eval_poly(poly_coeffs, s_val):\n',
        '    terms = []\n',
        '    deg = len(poly_coeffs) - 1\n',
        '    for i, c in enumerate(poly_coeffs):\n',
        '        if c == 0: continue\n',
        '        power = deg - i\n',
        '        c_str = format_frac(c)\n',
        '        if power == 0:\n',
        '            terms.append(c_str)\n',
        '        else:\n',
        '            if power == 1:\n',
        '                terms.append(f"({c_str})({format_complex_frac(s_val)})")\n',
        '            else:\n',
        '                terms.append(f"({c_str})({format_complex_frac(s_val)})^{{{power}}}")\n',
        '    if not terms: return "0"\n',
        '    return " + ".join(terms).replace("+ -", "- ")\n',
        '\n'
    ]

    # Insert at the top of the file before main()
    for i, l in enumerate(lines):
        if l.startswith('def main():'):
            lines = lines[:i] + format_root_lines + get_block_latex_lines + format_eval_poly_lines + lines[i:]
            break

    # Find and delete local definitions
    final_lines = []
    skip = False
    skip_indent = 0
    for l in lines:
        if skip:
            if len(l) - len(l.lstrip()) <= skip_indent and l.strip() != '':
                skip = False
            else:
                continue

        if l.strip().startswith('def format_root(r):') or \
           l.strip().startswith('def get_block_latex(n_lat, d_lat, is_g=False):') or \
           l.strip().startswith('def format_eval_poly(poly_coeffs, s_val):'):
            skip = True
            skip_indent = len(l) - len(l.lstrip())
            continue
            
        final_lines.append(l)

    with open('app.py', 'w', encoding='utf-8') as f:
        f.writelines(final_lines)

modify()
