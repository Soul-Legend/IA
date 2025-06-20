# Classificador de Imagens: Gato vs. No-Gato
Este projeto implementa e compara diferentes arquiteturas de redes neurais para realizar
a **classificao binria de imagens** em duas categorias: **"Gato"** e **"No Gato"**. O
trabalho foi desenvolvido como parte da disciplina **INE5430 - Inteligncia Artificial**
da **UFSC**, utilizando dados disponibilizados no curso de Deep Learning de Andrew Ng.
## Estrutura do Projeto
- `classificador_gatos.py`: Script principal contendo:
- Carregamento e pr-processamento dos dados
- Implementao de quatro arquiteturas de rede
- Treinamento, avaliao e visualizao dos resultados
- `dados/`: Diretrio contendo os arquivos `.h5` necessrios:
- `train_catvnoncat.h5`
- `test_catvnoncat.h5`
- `GatoNaoGatoPedro&Matheus-2.pdf`: Relatrio detalhado do experimento.
## Modelos Implementados
1. **Regresso Logstica (Perceptron Simples)**
2. **Rede Neural Rasa** (1 camada oculta)
3. **CNN (Rede Neural Convolucional)**
4. **CNN Otimizada** com:
- Data Augmentation (`RandomFlip`, `RandomRotation`)
- Dropout (`0.5`)
- Early Stopping
## Requisitos
O projeto foi desenvolvido com **Python 3.11**. Para instalar as dependncias, use:
```
pip install numpy h5py matplotlib seaborn scikit-learn tensorflow
```
> **Nota**: O projeto utiliza `TensorFlow` com Keras integrado, portanto recomendvel o
uso de uma GPU (opcional, mas acelera significativamente o treinamento da CNN).
## Como Executar
1. **Certifique-se de que os dados `.h5` esto no diretrio `dados/`**:
```
classificador_gatos.py
dados/
train_catvnoncat.h5
test_catvnoncat.h5
```
2. **Execute o script principal**:
```
python classificador_gatos.py
```
O script ir:
- Treinar todos os quatro modelos
- Mostrar grficos de acurcia/loss por poca
- Exibir as matrizes de confuso
- Imprimir relatrios de classificao detalhados
## Resultados
| Modelo | Acurcia no Teste | Observaes principais
|
|----------------------|-------------------|--------------------------------------------
------|
| Regresso Logstica | 66.00% | Forte overfitting, dificuldades com a classe
"Gato" |
| Rede Neural Rasa | 72.00% | Leve melhora, ainda com overfitting
|
| CNN Bsica | 76.00% | Aprende melhor as features, mas apresenta
overfitting |
| CNN Otimizada | 90.00% | Excelente generalizao com tcnicas de
regularizao |
Para uma anlise completa, consulte o relatrio em PDF includo no repositrio.
## Observaes Tcnicas
- **Entrada das imagens**: 64x64 pixels com 3 canais RGB
- **Sada da rede**: Classificao binria (0 = No Gato, 1 = Gato)
- **Funo de perda**: `binary_crossentropy`
- **Otimizador**: `Adam` com taxa de aprendizado padro (0.001)
- **Batch Size**: 32
- **Ativaes**:
- `ReLU` nas camadas ocultas
- `Sigmoid` na camada de sada
