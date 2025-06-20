# Classificador de Imagens: Gato vs. No-Gato
Este projeto implementa e compara diferentes arquiteturas de redes neurais para realizar
a **classificão binária de imagens** em duas categorias: **"Gato"** e **"Não Gato"**. O
trabalho foi desenvolvido como parte da disciplina **INE5430 - Inteligência Artificial**
da **UFSC**, utilizando dados disponibilizados no curso de Deep Learning de Andrew Ng.
## Estrutura do Projeto
- `classificador_gatos.py`: Script principal contendo:
- Carregamento e pré-processamento dos dados
- Implementação de quatro arquiteturas de rede
- Treinamento, avaliação e visualização dos resultados
- `dados/`: Diretório contendo os arquivos `.h5` necessários:
- `train_catvnoncat.h5`
- `test_catvnoncat.h5`
- `GatoNaoGatoPedro&Matheus-2.pdf`: Relatório detalhado do experimento.
## Modelos Implementados
1. **Regressão Logística (Perceptron Simples)**
2. **Rede Neural Rasa** (1 camada oculta)
3. **CNN (Rede Neural Convolucional)**
4. **CNN Otimizada** com:
- Data Augmentation (`RandomFlip`, `RandomRotation`)
- Dropout (`0.5`)
- Early Stopping
## Requisitos
O projeto foi desenvolvido com **Python 3.11**. Para instalar as dependências, use:
```
pip install numpy h5py matplotlib seaborn scikit-learn tensorflow
```
> **Nota**: O projeto utiliza `TensorFlow` com Keras integrado, portanto recomendável o
uso de uma GPU (opcional, mas acelera significativamente o treinamento da CNN).
## Como Executar
1. **Certifique-se de que os dados `.h5` estão no diretório `dados/`**:
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
O script irá:
- Treinar todos os quatro modelos
- Mostrar gráficos de acurácia/loss por época
- Exibir as matrizes de confuso
- Imprimir relatórios de classificação detalhados
## Resultados
![image](https://github.com/user-attachments/assets/764989df-8024-46e4-a066-ac65752a1063)

Para uma análise completa, consulte o relatório em PDF incluído no repositório.
## Observaes Técnicas
- **Entrada das imagens**: 64x64 pixels com 3 canais RGB
- **Saída da rede**: Classificação binária (0 = No Gato, 1 = Gato)
- **Função de perda**: `binary_crossentropy`
- **Otimizador**: `Adam` com taxa de aprendizado padrão (0.001)
- **Batch Size**: 32
- **Ativaes**:
- `ReLU` nas camadas ocultas
- `Sigmoid` na camada de saída
