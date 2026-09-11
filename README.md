# Polaris LGR — Análise Interativa do Lugar Geométrico das Raízes

Aplicação web interativa para análise e síntese de sistemas de controle em malha fechada via método do Lugar Geométrico das Raízes (Root Locus de Evans). O objetivo do projeto é aliar rigor matemático analítico com recursos visuais interativos para fins didáticos e de engenharia.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Funcionalidades e Passos de Evans](#funcionalidades-e-passos-de-evans)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Requisitos e Instalação](#requisitos-e-instalação)
- [Executando a Aplicação](#executando-a-aplicação)
- [Suíte de Testes e Validação](#suíte-de-testes-e-validação)
- [Integração Contínua (CI)](#integração-contínua-ci)
- [Licença](#licença)

---

## Visão Geral

Diferente de simuladores convencionais que operam como caixas-pretas fornecendo apenas a curva final, o **Polaris LGR** expõe toda a dedução matemática e comprovação analítica intermediária da teoria clássica de controle, demonstrando cada etapa com fórmulas em LaTeX, tabelas de coeficientes e gráficos vetoriais no plano complexo $s$.

A aplicação suporta sistemas de ordem arbitrária com realimentação unitária ou não-unitária $H(s)$, e disponibiliza dois modos de operação:
- **Modo Completo:** Exposição didática detalhada de cada um dos 12 passos de Evans, com deduções e comprovações analíticas formais (prova real).
- **Modo Simplificado:** Visualização direta dos itens essenciais para provas e relatórios técnicos (Item a: Esboço do LGR; Item b: Teste de Ponto e Cálculo de Ganho K).

---

## Funcionalidades e Passos de Evans

1. **Passo 1 — Equação Característica:** Obtenção analítica de $1 + K \cdot P(s) = 0$ a partir de $G(s)$ e $H(s)$.
2. **Passo 2 — Forma Fatorada:** Identificação dos polos $p_i$ e zeros $z_j$ em malha aberta.
3. **Passo 3 — Singularidades no Plano s:** Mapeamento visual das raízes no plano complexo.
4. **Passo 4 — Segmentos no Eixo Real:** Aplicação do critério de paridade à esquerda e animação de varredura.
5. **Passo 5 — Número de Ramos:** Contagem formal de lugares separados $LS = \max(n_P, n_Z)$.
6. **Passo 6 — Simetria:** Validação geométrica em relação ao eixo real.
7. **Passo 7 — Assíntotas:** Determinação do centroide $\sigma_A$, ângulos radiais $\theta_k$ e cruzamentos no plano imaginário.
8. **Passo 8 — Pontos de Saída e Entrada (Breakaway/Break-in):** Resolução analítica de $dK/ds = 0$ via regra do quociente, validação de paridade e classificação pela segunda derivada $d^2K/ds^2$.
9. **Passo 9 — Cruzamento com Eixo Imaginário:** Construção formal da Tabela de Routh-Hurwitz, isolamento do ganho crítico $K_{crítico}$ e frequência de oscilação $\omega$ via equação auxiliar $A(s) = 0$.
10. **Passo 10 — Ângulos de Partida e Chegada:** Cálculo tangencial para polos e zeros complexos conjugados.
11. **Passo 11 — Critério de Ângulo:** Validação de pertinência para ponto arbitrário $s_0$, tabela de fasores e cálculo de deficiência angular $\Delta\theta$.
12. **Passo 12 — Condição de Módulo e Prova Real:** Avaliação do ganho estático $K$ pelo produto das distâncias e substituição direta $P(s_0) = D(s_0) + K \cdot N(s_0) = 0$.
13. **Simulação Dinâmica:** Trajetória animada dos polos em malha fechada controlada por controle deslizante interativo.

---

## Arquitetura do Projeto

O código é estruturado de acordo com o princípio de separação de responsabilidades:

- `app.py`: Camada de orquestração do Streamlit, gerenciamento de estado (`st.session_state`), cache reativo (`@st.cache_data`) e seleção de predefinições.
- `lgr_math.py`: Núcleo matemático puro e desacoplado. Executa cálculos analíticos, simbólicos (SymPy) e numéricos (NumPy) sem dependência de UI.
- `lgr_plots.py`: Fábrica de gráficos e figuras interativas do Plotly.
- `lgr_steps.py`: Componentes de apresentação pedagógica e renderização dos 12 passos do LGR.
- `tests/`: Suíte completa de testes automatizados com pytest (testes unitários, invariantes teóricos, casos âncora e testes de integração de UI).

---

## Requisitos e Instalação

Requer Python 3.9 ou superior.

1. Clone o repositório:
```bash
git clone https://github.com/franssoares/polaris-lgr.git
cd polaris-lgr
```

2. Crie e ative um ambiente virtual (recomendado):
```bash
python -m venv venv
# No Linux/macOS:
source venv/bin/activate
# No Windows:
venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

---

## Executando a Aplicação

Inicie o servidor local do Streamlit:
```bash
streamlit run app.py
```

Acesse o endereço exibido no terminal (por padrão `http://localhost:8501`).

---

## Suíte de Testes e Validação

O projeto conta com mais de 190 testes automatizados que garantem a exatidão teórica de todos os cálculos de controle:

- **Casos Âncora (`test_casos_ancora.py`):** Validação contra problemas clássicos com gabarito analítico fechado.
- **Invariantes Teóricos (`test_invariantes.py`):** Testes parametrizados que verificam teoremas de controle (fechamento de laço, simetria, número de ramos, invariância de assíntotas).
- **Entradas Inválidas (`test_entradas_invalidas.py`):** Verificação de tratamento de erros, entradas degeneradas e cancelamento polo-zero.
- **Testes Unitários (`test_lgr_math_unit.py`):** Testes diretos para formatação, parsing, Routh-Hurwitz e derivadas.
- **Testes de UI (`test_ui.py`):** Execução sem falhas da interface do Streamlit usando `AppTest`.

Para rodar todos os testes:
```bash
pytest tests/ -v
```

---

## Integração Contínua (CI)

O repositório inclui um pipeline de integração contínua configurado via GitHub Actions em `.github/workflows/ci.yml`, testando automaticamente cada push e pull request nas versões 3.10, 3.11 e 3.12 do Python.

---

## Licença

Distribuído sob a licença [MIT](LICENSE). Consulte o arquivo `LICENSE` para mais detalhes.
