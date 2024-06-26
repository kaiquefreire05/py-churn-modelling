"""
Module for training and evaluating a deep learning model using Keras and Scikit-Learn.

This module loads a pre-processed dataset, defines and trains a deep learning model,
generates performance charts, and saves the trained model. It includes functions for
model creation, custom callbacks for logging and learning rate adjustments, and performance metrics.

Classes:
    LoggingCallback: Custom callback for logging metrics during model training.

Functions:
    create_model(): Creates and returns a deep learning model with the defined architecture.

Main Execution:
    - Loads training and testing data.
    - Sets up logging system.
    - Defines and compiles the deep learning model.
    - Trains the model with callbacks for early stopping, learning rate reduction, and logging.
    - Generates accuracy and loss charts during training.
    - Evaluates the model with accuracy metrics, classification report, and confusion matrix.
    - Saves the trained model and its weights.

Dependencies:
    - sklearn.metrics: accuracy_score, classification_report, confusion_matrix
    - keras.layers: Dense, Dropout, Input
    - keras.models: Sequential
    - keras.callbacks: EarlyStopping, ReduceLROnPlateau, Callback
    - logging: Logging configuration and usage
    - pickle: Loading pre-processed data
    - matplotlib.pyplot: Plotting graphs
    - seaborn: Generating confusion matrix plots
"""
# %% Importações

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from keras.layers import (
    Dense,
    Dropout,
    Input
)

from keras.models import Sequential
from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    Callback
)
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# %% Carregando as variáveis

with open('dataset-variables/churn-modelling.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# %% Confirando o logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log', mode='w')
    ]
)

log = logging.getLogger()

# %% Criando um callbacks personalizado para armazenar os dados de treino


class LoggingCallback(Callback):
    """
    LoggingCallback class for Keras models.

    Inherits from the `Callback` class provided by Keras. This callback logs
    training and validation metrics (loss and accuracy) at the end of each epoch.

    Args:
        None

    Methods:
        on_epoch_end(self, epoch, logs=None):
            Logs training and validation metrics at the end of each epoch.
    """

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        log_message = (
            f"Epoch {epoch + 1}: "
            f"loss={logs.get('loss', 0):.4f}, "
            f"accuracy={logs.get('accuracy', 0):.4f}, "
            f"val_loss={logs.get('val_loss', 0):.4f}, "
            f"val_accuracy={logs.get('val_accuracy', 0):.4f}"
        )
        log.info(log_message.strip().strip(','))

# %% Criação dos callbacks


es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
lg = LoggingCallback()

# %% Método para criação de modelo de deep learning


def create_model():
    """
    Função para retornar uma modelo de deep learning ajustado para o caso.

    returns:
        model: Modelo de deep learning pronto para uso
    """
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    return model

# %% Compilando, treinado e criando modelo


model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=1000, batch_size=8,
                    validation_data=(X_test, y_test), callbacks=[es, rlr, lg])

# %% (Plot) Gráfico dos históricos

model_accuracy = history.history['accuracy']
model_val_accuracy = history.history['val_accuracy']
model_loss = history.history['loss']
model_val_loss = history.history['val_loss']

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(model_accuracy, label='Training Accuracy')
plt.plot(model_val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.legend(fontsize='large')

plt.subplot(1, 2, 2)
plt.plot(model_loss, label='Training Loss')
plt.plot(model_val_loss, label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.legend(fontsize='large')

plt.savefig('plots/plot-training-validation.png')
plt.show()

# %% Fazendo novas previsões

predicts = model.predict(X_test)
predicts = (predicts > 0.5)

# %% Accuracy Score

acc = accuracy_score(y_true=y_test, y_pred=predicts)
print(f'A acurácia do modelo é: {acc}')

# %% Classification Report

print(
    f'Tabela de Classificação: \n\n {classification_report(y_true= y_test, y_pred= predicts)}')

# %% Matriz de confusão

confuse = confusion_matrix(y_test, predicts)
labels = ['No Exit', 'Exit']

plt.figure(figsize=(6, 5))
sns.heatmap(confuse, annot=True, cmap='coolwarm',
            xticklabels=labels, yticklabels=labels, fmt='d')
plt.title('Confusion Mattrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12, fontweight='bold')
plt.ylabel('Real', fontsize=12, fontweight='bold')
plt.savefig('plots/confusion-matrix.png')
plt.show()

# %% Salvando modelo

model.save('save_model/model.h5')
model.save_weights('save_model/weights_model.weights.h5')
