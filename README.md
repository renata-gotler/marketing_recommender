# 📌 iFood Case Técnico — Data Science
![Code Coverage](https://img.shields.io/badge/Code%20Coverage-100%25-success?style=flat)

📖 Visão Geral
Este projeto foi desenvolvido como parte do Case Técnico de Data Science do iFood, com o objetivo de criar uma solução baseada em dados para otimizar a distribuição de cupons e ofertas aos clientes.
A solução envolve:
- Processamento e unificação de dados de clientes, ofertas e transações.
- Construção de modelos de machine learning para prever a melhor oferta para cada cliente.
- Comunicação clara dos resultados e impacto esperado no negócio.

📊 Dados Utilizados
- offers.json: metadados das ofertas (tipo, valor mínimo, duração, canais).
- customers.json: atributos de ~17k clientes (idade, gênero, limite de crédito, data de registro).
- transactions.json: ~300k eventos (transações, ofertas recebidas, aceitação de ofertas).

🛠️ Tecnologias
- PySpark: processamento e unificação dos dados.
- Pandas/Numpy: manipulação adicional e análise exploratória.
- Scikit-learn: pipelines de pré-processamento e modelagem.
- XGBoost: modelo principal de classificação.
- MLflow: rastreamento de experimentos e logging de modelos.
- Matplotlib/Seaborn: visualização de métricas e resultados.

🔎 Abordagem
- Preparação dos dados
- Limpeza e imputação de valores faltantes.
- Criação de features derivadas (ex.: total gasto, média de gasto, quantidade de iterações).
- Unificação de clientes, ofertas e transações em um dataset único.
- Modelagem
- Pipeline com pré-processamento (imputação, encoding, scaling).
- Classificador XGBoost com validação cruzada.
- Foco em prever a melhor oferta. Sendo melhor oferta o valor esperado da compra considerando a oferta * quantidade de vezes que a oferta é aceita pelo cliente.
- Técnicas para lidar com desbalanceamento, focando na métrica F1 macro.
- Avaliação com métricas macro (F1, Precision, Recall).
- Avaliação
- Accuracy ≈ 97% (inflado pelo desbalanceamento).
- F1 Macro ≈ 0.16 (baixa performance em classes minoritárias).
- Interpretação: modelo atual evita desperdício de cupons, mas precisa evoluir para identificar melhor os clientes que aceitariam ofertas.

📈 Resultados de Negócio
- Eficiência: redução significativa no envio de cupons irrelevantes → economia de orçamento de marketing.
- Oportunidade: baixa capacidade atual de prever ofertas específicas → espaço para aumentar taxa de aceitação.
- Impacto esperado:
- +X% aumento na taxa de aceitação de cupons (após ajustes).
- Redução do custo por conversão.
- Maior fidelização e aumento do valor de vida do cliente (CLV).

🚀 Próximos Passos
- Implementar técnicas avançadas de balanceamento.
- Teste de abordagem hierárquica (“oferta vs. sem oferta” → “qual oferta”).
- Personalizar recomendações por canal (não havia essa informação na tabela de transações).
- Extrair mais variáveis conseguindo colocar no dataset de transações informações adicionais como a data da transação para entender comportamento do cliente ao longo do tempo.
- Monitorar métricas de negócio em piloto real.

▶️ Como Executar
- Clone o repositório:
git clone https://github.com/renata-gotler/marketing_recommender
cd ifood-case


- Crie e ative um ambiente virtual:
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate    # Windows


- Instale as dependências:
pip install -r requirements.txt


- Execute os notebooks:
- notebooks/1_data_processing.ipynb → prepara os dados.
- notebooks/2_modeling.ipynb → treina e avalia o modelo.

📌 Critérios de Avaliação
- Qualidade e organização do código.
- Clareza na análise exploratória.
- Justificativa das escolhas técnicas.
- Criatividade na solução.
- Comunicação clara dos resultados.
- [Apresentação](https://docs.google.com/presentation/d/1-rRTXZKefMvB9VkbZJ1pcsFOk0fYty6OLC5sDlrL2Zg/edit?usp=sharing)
