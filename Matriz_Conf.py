import os
os.environ['OMP_NUM_THREADS'] = '2'

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

# Carrega os dados do arquivo '29013.csv' sem cabeçalho
df = pd.read_csv('29013.csv', header=None)

# Remove a primeira coluna, que parece ser um índice
df = df.drop(columns=0)

# Renomeia as colunas restantes para A, B e C, onde C é o atributo-alvo
df.columns = ['A', 'B', 'C']

# Separa as colunas de features (A e B) e o atributo-alvo (C)
X = df[['A', 'B']]
y = df['C']

# Inicializa o modelo k-means com 4 clusters e treina com os dados de features
# O parâmetro random_state é usado para garantir a reprodutibilidade dos resultados
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
kmeans.fit(X)

# Obtém os rótulos de cluster atribuídos pelo modelo a cada ponto de dados
cluster_labels = kmeans.labels_

# Gera a matriz de confusão comparando o atributo-alvo original (y)
# com os rótulos de cluster do k-means
cm = confusion_matrix(y, cluster_labels)

# Imprime a matriz de confusão
print("\nMatriz de Confusão:")
print(cm)