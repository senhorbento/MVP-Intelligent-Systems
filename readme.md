
# Predição do Nível do Rio Ládario

## Visão Geral
Este projeto tem como objetivo prever o nível do rio Ládario utilizando modelos de machine learning. Ele combina tecnologias Python e .NET para processamento de dados e predição, com módulos separados para aprendizado de máquina, predição e tratamento de dados.

## Notebook
- Pode ser acessado através desse [link](https://github.com/senhorbento/MVP-Intelligent-Systems/blob/main/RioL%C3%A1dario.ipynb)

## Requisitos do Sistema
- **Tratamento de Base de Dados (C# .NET 8)**: Responsável por parte do tratamento de dados.
- **Predictor (Python 3.12)**: Módulo responsável por realizar as predições do nível do rio.
- **Machine Learning (Python 3.12)**: Executa o treinamento e testes do modelo de machine learning.

## Instalação
Para instalar as dependências necessárias para o Python, execute o seguinte comando no diretório raiz do projeto:
```bash
pip install --no-cache-dir -r requirements.txt
```

## Como Executar

### Tratamento de dados
Para rodar o tratamento incial dos dados, execute:
```bash
cd databases/Treatment
dotnet restore Treatment.csproj
dotnet run Treatment.csproj
```

### Machine Learning
Para treinar ou rodar o modelo de machine learning, execute:
```bash
cd ml
python ml.py
```

### Predictor
Para rodar o preditor de nível do rio, execute:
```bash
cd predictor
python predictor.py
```

### Teste do Modelo
Para rodar os testes do modelo de predição, execute:
```bash
cd predictor
pytest test-model.py
```

## Fontes de Dados
Os dados utilizados no modelo de predição foram extraídos das seguintes fontes:
- [Dados do Nível do Rio](https://www.marinha.mil.br/chn-6/?q=alturaAnterioresRios)
- [Dados de Temperatura](https://tempo.inmet.gov.br/TabelaEstacoes/A001)

## Notas
- Certifique-se de estar utilizando Python 3.12 e .NET 8 para garantir a compatibilidade com as dependências do projeto.
- Os módulos de predição e machine learning devem ser executados separadamente conforme descrito.
