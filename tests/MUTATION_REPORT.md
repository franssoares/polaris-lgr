# Relatório de Mutation Testing

Abaixo está a avaliação da robustez da suíte de testes frente à introdução deliberada de falhas (mutações) na lógica matemática. O objetivo é assegurar que a suíte consiga detectar as quebras e não deixe "passar" nenhum erro teórico do LGR.

| # | Mutação aplicada | Detectada? | Testes que falharam |
|---|---|---|---|
| 1 | Inverter sinal do centroide `(Σzeros - Σpolos)` | Sim | `test_caso_1`, `test_caso_2`, `test_caso_3`, `test_invariante_7_assintotas` |
| 2 | Trocar `nP-nZ` por `nZ-nP` no ângulo da assíntota | Sim | `test_caso_1`, `test_caso_2`, `test_caso_3` |
| 3 | Inverter a regra ímpar/par do eixo real | Sim | Vários testes do `test_invariante_2` e `test_invariante_4` |
| 4 | Inverter o sinal da regra de derivação `dK/ds` | Sim | `test_caso_2`, múltiplos de `test_invariante_3` |
| 5 | Tabela de Routh: usar linha errada | Sim | `test_caso_3`, `test_invariante_5`, `test_invariante_5_mut5` |
| 6 | Ângulo Partida/Chegada: inverter soma/sub | Sim | `test_invariante_6` sobre os sistemas `sys_data6` e `sys_data7` |
| 7 | Critério de Módulo: K = num/den invertido | Sim | Inúmeros casos de `test_invariante_4` |
| 8 | Contagem de ramos: `nP+1` | Sim | `test_caso_1`, `test_caso_2`, quase todos os casos de `test_invariante_8` |

## Lacunas e Correções
Durante o teste da mutação **#5** (usar a linha errada na tabela de Routh), inicialmente nem todos os sistemas acusavam erro na construção da equação auxiliar, visto que para sistemas de grau 3 usar a linha 0 (`s^3`) em vez da linha 1 (`s^2`) acabava encontrando um $\omega$ raiz falso que por coincidência também validava (ao menos no formato do gabarito numérico). 
Para preencher esse buraco de cobertura sutil (quando graus maiores estão envolvidos), foi introduzido o teste explícito **`test_invariante_5_cruzamento_jw_mut5`** na suíte, que obriga os $j\omega$ encontrados pela equação auxiliar a serem raízes verdadeiras e exatas do polinômio na simulação de malha fechada inteira, bloqueando qualquer "coincidência falsa" provocada pela leitura do Routh fora de ordem.

A suíte agora tem robustez plena contra degenerações algébricas do método de Evans.
