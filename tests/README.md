# Suíte de Testes - LGR (Root Locus)

Esta pasta contém uma bateria completa de testes baseados no `pytest`, projetada para garantir que a matemática dos 12 passos do Lugar Geométrico das Raízes implementada em `lgr_math.py` e extraída em `lgr_helper.py` esteja solidamente de acordo com a teoria Clássica de Controle (Método de Evans).

**Veja os relatórios de qualidade aprovados:**
- 🛡️ [**Matriz de Cobertura das 15 Categorias**](COVERAGE.md)
- 🧬 [**Relatório de Testes de Mutação (Mutation Testing)**](MUTATION_REPORT.md)

## Arquivos e Cobertura

- **`lgr_helper.py`**: Motor de simulação isolado sem dependências de UI (Streamlit/Plotly). Consome as mesmas rotinas de cálculo do app (`build_routh_hurwitz`, `find_breakaway_points`) e resolve estaticamente todos os passos matemáticos.
- **`test_casos_ancora.py`**: Valida a Seção A. Contém os 3 sistemas clássicos catalogados em livros/disciplinas (ex. G(s)H(s) = K/[s(s+2)(s+4)]). Testa cruamente contra números de gabarito para evitar erros de consistência global.
- **`test_invariantes.py`**: Valida as Seções B e C. Gera 17 sistemas variados (polos repetidos, complexos conjugados, sem zero, nP=nZ, semiplano direito, etc.) e submete todos a **9 invariantes matemáticos** (ex.: $K$ crítico realmente zera o polinômio? Ponto de saída tem multiplicidade $\ge 2$? Respeita limite angular de 180º em cima do eixo?).
- **`test_entradas_invalidas.py`**: Verifica a Seção B (entradas degeneradas como `DG=[0]` ou cancelamentos polo-zero) para atestar que o simulador sobrevive e reage graciosamente em vez de apresentar falhas de segmentação ou `Crash` em runtime.

## Como Executar

Para rodar todos os testes com um sumário das passagens, execute:

```bash
pytest -v tests/ --tb=short
```

Ao rodar isso, todos os mais de 140 cenários paramétricos são processados, garantindo confiabilidade nível de produção na ferramenta didática!
