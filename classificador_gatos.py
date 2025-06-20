# -*- coding: utf-8 -*-
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# --- 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS ---

def load_dataset():
    """
    Carrega os datasets de treino e teste a partir dos arquivos h5.
    """
    # Carrega o conjunto de treino
    train_dataset = h5py.File('dados/train_catvnoncat.h5', "r")
    X_train_orig = np.array(train_dataset["train_set_x"][:]) # features de treino
    Y_train_orig = np.array(train_dataset["train_set_y"][:]) # labels de treino

    # Carrega o conjunto de teste
    test_dataset = h5py.File('dados/test_catvnoncat.h5', "r")
    X_test_orig = np.array(test_dataset["test_set_x"][:]) # features de teste
    Y_test_orig = np.array(test_dataset["test_set_y"][:]) # labels de teste
    
    classes = np.array(test_dataset["list_classes"][:]) # lista de classes

    # Ajusta o shape dos labels para (m, 1) onde m é o número de exemplos
    Y_train = Y_train_orig.reshape((1, Y_train_orig.shape[0])).T
    Y_test = Y_test_orig.reshape((1, Y_test_orig.shape[0])).T

    return X_train_orig, Y_train, X_test_orig, Y_test, classes

def preprocess_data(X_train_orig, X_test_orig):
    """
    Normaliza os dados de imagem e os achata (flatten) para modelos densos.
    """
    # Normaliza os valores dos pixels para o intervalo [0, 1]
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Achata as imagens de (64, 64, 3) para um vetor de (12288,)
    X_train_flatten = X_train.reshape(X_train.shape[0], -1)
    X_test_flatten = X_test.reshape(X_test.shape[0], -1)
    
    return X_train_flatten, X_test_flatten, X_train, X_test

# Carregando os dados
X_train_orig, Y_train, X_test_orig, Y_test, classes = load_dataset()

# Pré-processando os dados
# Temos duas versões dos dados: achatada (para redes densas) e original (para CNN)
X_train_flatten, X_test_flatten, X_train_cnn, X_test_cnn = preprocess_data(X_train_orig, X_test_orig)

print("--- Informações dos Dados ---")
print(f"Número de imagens de treino: {X_train_orig.shape[0]}")
print(f"Número de imagens de teste: {X_test_orig.shape[0]}")
print(f"Dimensão de cada imagem: {X_train_orig.shape[1:]}")
print(f"Shape X_train achatado: {X_train_flatten.shape}")
print(f"Shape Y_train: {Y_train.shape}")
print(f"Shape X_test achatado: {X_test_flatten.shape}")
print(f"Shape Y_test: {Y_test.shape}")
print("-" * 30 + "\n")


# --- 2. FUNÇÃO AUXILIAR PARA AVALIAÇÃO E PLOT ---

def evaluate_and_plot(model, X_test, Y_test, history, model_name):
    """
    Avalia o modelo e plota a matriz de confusão e o histórico de treinamento.
    """
    print(f"\n--- Avaliação do Modelo: {model_name} ---")

    # Avaliação final no conjunto de teste
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Acurácia no Teste: {accuracy * 100:.2f}%")
    print(f"Loss no Teste: {loss:.4f}")

    # Previsões para a matriz de confusão
    Y_pred_probs = model.predict(X_test)
    Y_pred_classes = (Y_pred_probs > 0.5).astype("int32")

    # Matriz de Confusão
    cm = confusion_matrix(Y_test, Y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Não Gato', 'Gato'], 
                yticklabels=['Não Gato', 'Gato'])
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.show()

    # Relatório de Classificação
    print("\nRelatório de Classificação:")
    print(classification_report(Y_test, Y_pred_classes, target_names=['Não Gato', 'Gato']))
    
    # Gráfico de Acurácia e Loss do Treinamento
    plt.figure(figsize=(12, 5))
    
    # Plot de Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia de Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title(f'Acurácia - {model_name}')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    # Plot de Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss de Treino')
    plt.plot(history.history['val_loss'], label='Loss de Validação')
    plt.title(f'Loss (Custo) - {model_name}')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# --- 3. MODELO 1: REGRESSÃO LOGÍSTICA (PERCEPTRON SIMPLES) ---

print("\n--- INICIANDO MODELO 1: REGRESSÃO LOGÍSTICA ---")
model_logistic = Sequential([
    # A camada de entrada é definida implicitamente pela `input_shape`
    # Uma única camada Dense com 1 neurônio e ativação sigmoide
    Dense(1, activation='sigmoid', input_shape=(X_train_flatten.shape[1],))
])

model_logistic.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

model_logistic.summary()

# Treinando o modelo
history_logistic = model_logistic.fit(X_train_flatten, Y_train,
                                    epochs=100,
                                    batch_size=32,
                                    validation_data=(X_test_flatten, Y_test),
                                    verbose=1)

# Avaliando o modelo
evaluate_and_plot(model_logistic, X_test_flatten, Y_test, history_logistic, "Regressão Logística")


# --- 4. MODELO 2: REDE NEURAL RASA (1 CAMADA OCULTA) ---

print("\n--- INICIANDO MODELO 2: REDE NEURAL RASA ---")
model_shallow = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_flatten.shape[1],)), # Camada oculta
    Dense(1, activation='sigmoid') # Camada de saída
])

model_shallow.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

model_shallow.summary()

# Treinando o modelo
history_shallow = model_shallow.fit(X_train_flatten, Y_train,
                                    epochs=100,
                                    batch_size=32,
                                    validation_data=(X_test_flatten, Y_test),
                                    verbose=1)

# Avaliando o modelo
evaluate_and_plot(model_shallow, X_test_flatten, Y_test, history_shallow, "Rede Neural Rasa")


# --- 5. MODELO 3: REDE NEURAL CONVOLUCIONAL (CNN) ---

print("\n--- INICIANDO MODELO 3: REDE NEURAL CONVOLUCIONAL (CNN) ---")
# Para a CNN, usamos os dados com a dimensão original (64, 64, 3)
input_shape_cnn = (64, 64, 3)

model_cnn = Sequential([
    # 1ª Camada Convolucional
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape_cnn),
    MaxPooling2D((2, 2)),
    
    # 2ª Camada Convolucional
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Achatar a saída para alimentar a camada densa
    Flatten(),
    
    # Camada Densa (classificador)
    Dense(64, activation='relu'),
    
    # Camada de Saída
    Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

model_cnn.summary()

# Treinando o modelo
history_cnn = model_cnn.fit(X_train_cnn, Y_train,
                            epochs=25, # CNNs convergem mais rápido neste problema
                            batch_size=32,
                            validation_data=(X_test_cnn, Y_test),
                            verbose=1)

# Avaliando o modelo
evaluate_and_plot(model_cnn, X_test_cnn, Y_test, history_cnn, "Rede Neural Convolucional (CNN)")

# --- 6. MODELO 4: CNN OTIMIZADA COM REGULARIZAÇÃO ---
from tensorflow.keras.layers import Dropout, RandomFlip, RandomRotation
from tensorflow.keras.callbacks import EarlyStopping

print("\n--- INICIANDO MODELO 4: CNN OTIMIZADA (Data Augmentation + Dropout) ---")

# Camadas de Aumento de Dados
data_augmentation = Sequential([
    RandomFlip("horizontal", input_shape=(64, 64, 3)),
    RandomRotation(0.1), # Rotação de até 10%
    # Você poderia adicionar RandomZoom, etc.
])

# Define o callback de Early Stopping
# patience=5: espera 5 épocas sem melhora na 'val_loss' antes de parar
# restore_best_weights=True: ao final, restaura os pesos da época com a melhor val_loss
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Construção do modelo otimizado
model_cnn_optimized = Sequential([
    data_augmentation, # Aplica o aumento de dados como a primeira camada
    
    # Arquitetura base da CNN
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    
    # Camada Densa com Dropout para regularização
    Dense(64, activation='relu'),
    Dropout(0.5), # "Desliga" 50% dos neurônios aleatoriamente durante o treino
    
    Dense(1, activation='sigmoid')
])

model_cnn_optimized.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

model_cnn_optimized.summary()

# Treinando o modelo otimizado com o callback
# Aumentamos o número de épocas, pois o Early Stopping cuidará de parar no momento certo.
history_cnn_optimized = model_cnn_optimized.fit(X_train_cnn, Y_train,
                                                epochs=50, # Aumente as épocas
                                                batch_size=32,
                                                validation_data=(X_test_cnn, Y_test),
                                                callbacks=[early_stopping], # Adiciona o callback
                                                verbose=1)

# Avaliando o modelo otimizado
# A avaliação usará automaticamente os melhores pesos restaurados pelo EarlyStopping
evaluate_and_plot(model_cnn_optimized, X_test_cnn, Y_test, history_cnn_optimized, "CNN Otimizada")