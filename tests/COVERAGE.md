# Matriz de Cobertura de Testes

A tabela abaixo cruza as **15 categorias de sistemas** com os cenários gerados em `generate_systems()` dentro do arquivo `test_invariantes.py` e os casos extras em `test_entradas_invalidas.py`.

| Categoria | Descrição | Sistema Testado (Array/Função) | Status |
| :--- | :--- | :--- | :---: |
| **1** | Sem zeros finitos, com 1, 2, 3 e 4 polos reais distintos | `sys_data0` a `sys_data3` | :white_check_mark: |
| **2** | Com zeros finitos, `nZ < nP` | `sys_data4` | :white_check_mark: |
| **3** | `nZ == nP` (sem assíntotas) | `sys_data5` | :white_check_mark: |
| **4** | Polos complexos conjugados | `sys_data6` | :white_check_mark: |
| **5** | Zeros complexos conjugados | `sys_data7` | :white_check_mark: |
| **6** | Polos repetidos / multiplicidade > 1 | `sys_data8` e `sys_data9` | :white_check_mark: |
| **7** | Com e sem polo na origem | `sys_data10` (com) e `sys_data1` (sem) | :white_check_mark: |
| **8** | Polo no semiplano direito (S.P.D.) | `sys_data11` | :white_check_mark: |
| **9** | Realimentação `H(s)` não unitária | `sys_data12` | :white_check_mark: |
| **10** | Sem ponto de saída/entrada no eixo real | `sys_data13` | :white_check_mark: |
| **11** | Múltiplos pontos de saída/entrada reais | `sys_data14` | :white_check_mark: |
| **12** | Estável para todo $K>0$ (sem cruzar o eixo $j\omega$) | `sys_data15` | :white_check_mark: |
| **13** | Cruzamento com o eixo $j\omega$ para $K$ finito | `sys_data16` | :white_check_mark: |
| **14** | Entradas inválidas (campos em branco, $nZ > nP$) | Tratados em `test_entradas_invalidas.py` | :white_check_mark: |
| **15** | Cancelamento exato de polo-zero | `test_cancelamento_polo_zero` | :white_check_mark: |

Todas as linhas possuem cobertura documentada. Nenhuma categoria está descoberta.
