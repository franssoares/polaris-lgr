# LGR - 12 Passos: Análise do Lugar Geométrico das Raízes

Uma aplicação web interativa desenvolvida em Python e Streamlit que constrói e disseca o Lugar Geométrico das Raízes (Root Locus) passo a passo, utilizando o método clássico de Evans.

## Contexto

Este projeto foi desenvolvido como parte da disciplina de Sistemas de Controle. O objetivo principal é servir como uma ferramenta didática avançada para estudantes de engenharia: em vez de apenas jogar a função de transferência em um simulador caixa-preta (como o MATLAB/Simulink) e receber o gráfico final, o app **educa e demonstra a teoria matemática por trás de cada etapa** da construção do LGR.

## Funcionalidades

O aplicativo recebe funções de transferência $G(s)H(s)$ como entrada fracionária polinomial e cobre detalhadamente os famosos 12 Passos de Evans:

* **Passos 1 e 2:** Montagem da Malha Aberta e Equação Característica.
* **Passo 3:** Extração gráfica de Polos e Zeros.
* **Passo 4:** Determinação dos segmentos válidos no Eixo Real.
* **Passos 5, 6 e 7:** Contagem de ramos, verificação de simetria e cálculo detalhado das Assíntotas (com Centroide e Ângulos radiais).
* **Passo 8:** Pontos de Saída/Entrada (Breakaway/Break-in) derivados a partir de $\frac{dK}{ds} = 0$.
* **Passo 9:** Cruzamento com o Eixo Imaginário (j$\omega$) utilizando o critério de Routh-Hurwitz passo a passo.
* **Passo 10:** Ângulos de Partida e Chegada para polos e zeros complexos.
* **Passos 11 e 12:** Ponto de Teste arbitrário com validação pelo Critério de Ângulo ($\pm 180^\circ$) e cálculo explícito do Ganho $K$ pelo Critério de Módulo.
* **Gráfico Final Interativo:** Animação precisa dos polos percorrendo as trajetórias do LGR conforme o ganho $K$ é incrementado dinamicamente via slider.

## Como Rodar Localmente

Certifique-se de ter o Python 3.9 ou superior instalado em sua máquina.

1. Clone o repositório.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o servidor do Streamlit:
   ```bash
   streamlit run app.py
   ```


## Licença

Este projeto está licenciado sob a [MIT License](LICENSE). Você é livre para utilizar, modificar e distribuir o código para fins acadêmicos e comerciais, desde que mantenha o aviso de direitos autorais original.
