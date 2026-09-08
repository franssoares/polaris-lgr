# Contribuindo para o LGR - 12 Passos

Agradecemos o seu interesse em contribuir com este projeto! O LGR - 12 Passos é uma ferramenta educacional para estudantes e profissionais de Engenharia de Controle, e toda ajuda para torná-lo melhor é bem-vinda.

Abaixo estão as diretrizes de como você pode contribuir com o repositório.

## Como rodar o projeto localmente

Para rodar e testar o app na sua máquina, você vai precisar do Python 3.9+ instalado.

1. Faça um Fork do repositório no GitHub.
2. Clone o seu fork para a sua máquina local:
   ```bash
   git clone https://github.com/SEU_USUARIO/lgr-12-passos.git
   cd lgr-12-passos
   ```
3. Crie e ative um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```
4. Instale as dependências listadas no `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
5. Rode a aplicação com o Streamlit:
   ```bash
   streamlit run app.py
   ```
O aplicativo abrirá no seu navegador, normalmente em `http://localhost:8501`.

## Padrão de Clean Code e Estilo

Mantemos o projeto legível e organizado. Ao escrever código, por favor, atente-se a:

* **Responsabilidades separadas:**
  * Cálculos e lógica matemática bruta devem residir em `lgr_math.py`.
  * Criação de gráficos e objetos do Plotly devem residir em `lgr_plots.py`.
  * Renderização, layout de página e interface de usuário devem ficar restritos a `app.py`.
* **Nomes descritivos:** Use nomes em inglês ou português que deixem claro o propósito de funções e variáveis. Evite nomes como `test`, `temp`, `foo`.
* **Docstrings:** Ao adicionar ou modificar uma função pública em `lgr_math.py` ou `lgr_plots.py`, atualize ou crie a docstring explicando os parâmetros (`Args:`) e o que é retornado (`Returns:`).
* **Formatação:** Recomendamos o uso de formatadores padrão da comunidade Python, como `black` e `ruff`, antes de submeter um código.

## Suíte de Testes (pytest)

Este projeto possui uma bateria rigorosa de validação matemática que cobre desde sistemas comuns até casos degenerados (sem zero, nP=nZ, instáveis em malha aberta, etc).

**Antes de abrir um Pull Request**, você **deve** garantir que não quebrou a matemática do LGR.

1. Instale o `pytest` se ainda não estiver instalado:
   ```bash
   pip install pytest
   ```
2. Rode a suíte completa de testes:
   ```bash
   pytest -v tests/
   ```
3. Todos os testes devem passar (indicado por `PASSED`). Se você adicionou uma funcionalidade nova de cálculo, por favor, adicione também o cenário correspondente em `tests/test_invariantes.py` ou crie um novo caso de teste se justificado.

## Como propor mudanças (Pull Requests)

1. Crie uma branch para a sua funcionalidade/correção a partir da `main`:
   ```bash
   git checkout -b feature/minha-melhoria
   ```
2. Faça seus commits de forma atômica e com mensagens descritivas:
   ```bash
   git commit -m "feat: Adiciona cálculo extra para ganho de malha fechada"
   ```
3. Suba a branch para o seu fork:
   ```bash
   git push origin feature/minha-melhoria
   ```
4. Abra um **Pull Request (PR)** na página principal do repositório no GitHub, descrevendo o que foi feito, o motivo da mudança e incluindo capturas de tela se a mudança for visual.
5. Aguarde o review dos mantenedores.

Ao contribuir, você concorda que o seu código estará sob a mesma licença open-source do projeto e que você seguirá nosso [Código de Conduta](CODE_OF_CONDUCT.md).
