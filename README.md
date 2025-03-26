# APLICAÇÃO DE DEEP LEARNING NA AVALIAÇÃO DO CONTEÚDO NUTRICIONAL

Este repositório contém o código e os recursos desenvolvidos para a Monografia II do curso de Ciência da Computação da Universidade Federal de Ouro Preto (UFOP), sob orientação do Prof. Dr. Eduardo José da Silva Luz. O projeto aplica técnicas de aprendizado profundo para estimar o conteúdo nutricional (calorias, massa, gordura, carboidratos e proteínas) de pratos de comida a partir de imagens 2D da base Nutrition5k. Foram implementados modelos Vision Transformer (ViT) com pré-treinamentos em ImageNet e COYO-700M, comparados a um baseline pré-treinado na JFT-300M. Também foram implementados os modelos Resnet50 e InceptionV2 para fins de comparação. O pipeline completo, incluindo treinamento, geração de predições e avaliação, está automatizado em um script Bash (`run_pipeline.sh`). Este README guia a configuração e execução do projeto.

## 🚀 Começando

Estas instruções permitem configurar e executar o projeto em sua máquina local para reproduzir os experimentos da monografia.

Consulte **[Implantação](#-implantação)** para notas sobre uso em sistemas ativos.

### 📋 Pré-requisitos

Você precisará do seguinte para rodar o projeto:

- Sistema operacional baseado em Unix (Linux ou macOS; para Windows, use WSL ou adapte o script)
- Python 3.8 ou superior
- Bibliotecas Python:
  - `numpy`
  - `pandas`
  - `torch`
  - `transformers`
  - `statistics`
- Base Nutrition5k (disponível em [nutrition5k](https://github.com/google-research-datasets/Nutrition5k))
- Bash (para executar o script `run_pipeline.sh`)

### 🔧 Instalação

Siga os passos para preparar o ambiente:

1. Clone o repositório:
```
git clone https://github.com/michele-andrade/Nutricao-Inteligente.git
```

2. Crie um ambiente virtual (opcional):
```
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências:
```
pip install -r requirements.txt
```

4. Baixe a base Nutrition5k

## ⚙️ Executando o Pipeline

O script `run_pipeline.sh` automatiza o processo em cinco etapas: treinamento, geração de predições e cálculo de estatísticas. Execute-o com:

```
bash run_pipeline.sh
```

### Passos do Pipeline
1. **Treinamento**: Treina os modelos ViT (ImageNet e COYO) usando `main.py`.
2. **Predições ImageNet**: Gera predições com `generate_predictions_ViT_imagenet.py`, salvas em `output/predictions_vit_imagenet.csv`.
3. **Predições COYO**: Gera predições com `generate_predictions_ViT_COYO.py`, salvas em `output/predictions_vit_COYO.csv`.
4. **Estatísticas ImageNet**: Calcula métricas (MAE, MAE%) com `generate_eval_statistics.py`, salvando em `output/output_statistics_vit_imagenet.json`.
5. **Estatísticas COYO**: Calcula métricas para o modelo COYO, salvando em `output/output_statistics_vit_COYO.json`.

Para rodar em segundo plano:
```
nohup bash run_pipeline.sh > pipeline.log 2>&1 &
```

Os resultados estarão na pasta `output/`.

## 🔩 Análise dos Resultados

O script `generate_eval_statistics.py` calcula o Erro Médio Absoluto (MAE) e o Erro Percentual (MAE%) com base nas predições e metadados da Nutrition5k. Esses valores comparam o desempenho dos modelos ViT com o baseline da monografia.

Exemplo de saída (em `output/output_statistics_vit_imagenet.json`):
```
{
  "Total Calories": {"MAE": 95.29, "MAE%": 37.81},
  "Total Mass": {"MAE": 63.87, "MAE%": 33.77}
}
```

## 🛠️ Construído com

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Pytorch](https://pytorch.org/)
- [Python](https://www.python.org/) - Linguagem principal
- [statistics](https://docs.python.org/3/library/statistics.html)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
- [Bash](https://www.gnu.org/software/bash/) - Automação do pipeline

## ✒️ Autores

- **Michele Soares de Andrade**
- **Eduardo José da Silva Luz** - *Orientação*

## 🎁 Agradecimentos

- Ao Prof. Dr. Eduardo José da Silva Luz pela orientação;
- À UFOP pelo suporte acadêmico;
- À comunidade open-source pelas ferramentas;

---

⌨️ com ❤️ por [Michele Soares de Andrade](https://github.com/michele-andrade) 😊

---

### Notas
1. **Caminhos**: Mantive os caminhos do script original, mas você pode ajustá-los conforme sua estrutura local.
2. **Flexibilidade**: Os comandos assumem que os scripts Python estão na raiz ou em subdiretórios claros. Ajuste se necessário (ex.: `scripts/main.py`).