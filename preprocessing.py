"""
Module for data preprocessing for a deep learning project.

This module performs loading, cleaning, visualization, and transformation of churn dataset
data, preparing them for training a deep learning model. The module also includes logging
configuration to track the process steps.

Functions:
    - Data loading from a zip file.
    - Cleaning and transformation of data, including removal of unnecessary columns and
      OneHot encoding.
    - Data visualization with charts for better understanding of data distribution.
    - Normalization of numerical data to the same scale.
    - Splitting data into training and testing sets.
    - Saving processed data sets using pickle.

Main Execution:
    - Sets up logging system.
    - Loads data from a zip file and reads it into a DataFrame.
    - Removes unnecessary columns.
    - Generates data distribution charts, including output counts by country and gender,
      number of products, and correlation map.
    - Converts categorical columns into dummy variables.
    - Normalizes numerical data.
    - Splits data into training and testing sets.
    - Saves processed data sets into a pickle file.

Dependencies:
    - sklearn.preprocessing: StandardScaler
    - sklearn.model_selection: train_test_split
    - logging: Logging configuration and usage
    - zipfile: Extraction of zip files
    - pandas: Data manipulation
    - matplotlib.pyplot: Plotting graphs
    - seaborn: Data visualization
    - pickle: Saving processed data
"""

# %% Importações

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# %% Configurando o logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/info-preprocessing.log', mode='w')
    ]
)

logger = logging.getLogger()

# %% Carregando os dados

caminho_zip = 'dataset-variables/churn-modelling.zip'
caminho_extract = 'C:/Users/kaiqu/OneDrive/Documentos/py-deep-learning-projects/churn-modelling/dataset-variables'

with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
    zip_ref.extractall(caminho_extract)

df = pd.read_csv('dataset-variables/Churn_Modelling.csv')
logger.info(df.head())

# %% Preprocessing

# Removendo colunas que não necessárias
columns_to_remove = ['RowNumber', 'CustomerId', 'Surname']
df.drop(columns_to_remove, axis=1, inplace=True)
logger.info(df.head())

# %% Visualização de informações

# Convertendo 'Exited' para string
df['Exited'] = df['Exited'].astype(str)

# Contagem de valores nulos
print(df.isnull().sum())

# (Plot) Obtendo a contagem de pessoas que saíram e ficaram
all_noexit = df['Exited'].value_counts()['0']
all_exit = df['Exited'].value_counts()['1']
all = [all_noexit, all_exit]
colors = ['#FF6666', '#6666FF']

plt.figure(figsize=(12, 8))
plt.pie(all, labels=['No Exit', 'Exit'], colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.85, explode=(0.05, 0))
plt.legend(loc='upper right', fontsize='large')
plt.title('Porcentagem das pessoas que saíram e ficaram',
          fontsize=14, fontweight='bold')
plt.savefig('plots/pie-exited.png')
plt.show()

# (Plot) Contagem de pessoas que saíram por país
colors = ['#6A5ACD', '#483D8B']

plt.figure(figsize=(10, 8))
sns.countplot(x='Geography', hue='Exited', data=df, palette=colors)
plt.title('Contagem de pessoas que saíram e que não saíram por país',
          fontsize=14, fontweight='bold')
plt.legend(title='Exited', labels=['No Exit', 'Exit'], fontsize='large')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('Country', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.savefig('plots/count-per-country.png')
plt.show()

# (Plot) Pessoas que saíram e não saíram por gênero
colors = ['#0066CC', '#003366']
plt.figure(figsize=(10, 8))
sns.countplot(x='Gender', hue='Exited', data=df, palette=colors)
plt.title('Pessoas que saíram e não saíram por gênero',
          fontsize=14, fontweight='bold')
plt.xlabel('Gender', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.legend(title='Exited', labels=['No Exit', 'Exit'], fontsize='large')
plt.grid(True, linestyle='--', alpha=0.5, color='black')
plt.savefig('plots/count-per-gender.png')
plt.show()

# (Plot) Número de produtos para todos os clientes
colors = ['#00ac0f', '#6da873']
plt.figure(figsize=(10, 8))
sns.countplot(x='NumOfProducts', hue='Exited', data=df, palette=colors)
plt.title('Número de produtos para todos os clientes',
          fontsize=14, fontweight='bold')
plt.xlabel('Number of Products', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.legend(title='Exited', labels=['No Exit', 'Exit'], fontsize='large')
plt.grid(True, linestyle='--', alpha=0.5, color='black')
plt.savefig('plots/count-per-product.png')
plt.show()

# (Plot) Mapa de correlação dos atributos númericos

plt.figure(figsize=(10, 8))
sns.heatmap(data=df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Mapa de Correção dos numéricos', fontsize=14, fontweight='bold')
plt.savefig('plots/correlation-map-numerics.png')
plt.show()

df['Exited'] = df['Exited'].astype(int)

# %% Aplicando o padrão OneHot para as colunas Gender e Geography

columns_dummy = ['Gender', 'Geography']
df = pd.get_dummies(data=df, columns=columns_dummy, drop_first=True)
logger.info(df.head())

# %% Dividindo entre previsores e classe

X = df.drop('Exited', axis=1)
y = df['Exited']

# %% Deixando todos os valores na mesma escala

scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# %% Divindo entre bases de treino e teste

X_train, X_test, y_train, y_test = train_test_split(
    x_scaled, y, random_state=42, test_size=0.25)

logger.info(
    f'\O Shape das variáveis de treino é: {X_train.shape}, {y_train.shape}')
logger.info(
    f'\O shape das variáveis de teste é: {X_test.shape}, {y_test.shape}')

# %% Salvando as variáveis usando o pickle

with open('dataset-variables/churn-modelling.pkl', 'wb') as f:
    pickle.dump([X_train, X_test, y_train, y_test], f)
