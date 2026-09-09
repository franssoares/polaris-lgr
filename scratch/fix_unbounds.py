import sys

def modify():
    with open('app.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the function get_char_poly_latex
    start_func = -1
    end_func = -1
    for i, l in enumerate(lines):
        if l.strip().startswith('def get_char_poly_latex(D_coeffs, N_coeffs):'):
            start_func = i
        if start_func != -1 and i > start_func and l.strip() == 'return res':
            end_func = i + 1
            break
            
    if start_func != -1 and end_func != -1:
        func_lines = lines[start_func:end_func]
        # unindent by 16 spaces
        new_func_lines = []
        for fl in func_lines:
            if fl.startswith('                '):
                new_func_lines.append(fl[16:])
            else:
                new_func_lines.append(fl)
                
        # Remove from original location
        del lines[start_func:end_func]
        
        # Insert before main()
        for i, l in enumerate(lines):
            if l.startswith('def main():'):
                lines = lines[:i] + new_func_lines + ['\n'] + lines[i:]
                break

    # Find and remove `import sympy as sp` inside functions
    final_lines = []
    for l in lines:
        if l.strip() == 'import sympy as sp' and not l.startswith('import sympy as sp'):
            # It's an indented import
            pass
        else:
            final_lines.append(l)

    with open('app.py', 'w', encoding='utf-8') as f:
        f.writelines(final_lines)

modify()
