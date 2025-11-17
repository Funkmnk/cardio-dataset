# Bibliotecas
import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from utils import montar_divisor, plotar_cotovelo, plotar_silhouette

# Carregando dados
dados = pd.read_csv('../data/processados/dados_preprocessados.csv')

montar_divisor("DETERMINAÇÃO DO NÚMERO DE CLUSTERS", 60, "=")

# Cotovelo
montar_divisor("CALCULANDO DISTORÇÕES (Elbow Method)", 60, "-")

distorcoes = []
intervalo_k = range(2, 16)  # Testar de 2 a 15 clusters

for k in intervalo_k:
    modelo_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo_kmeans.fit(dados)
    
    # Calculando a distorção
    distorcao = sum(np.min(cdist(dados, modelo_kmeans.cluster_centers_, 
                                  'euclidean'), axis=1)) / dados.shape[0]
    distorcoes.append(distorcao)
    
    print(f"k={k}: distorção={distorcao:.4f}")

# Valor de K
montar_divisor("DETERMINANDO K", 60, "-")

x0 = intervalo_k[0]
y0 = distorcoes[0]
xn = intervalo_k[-1]
yn = distorcoes[-1]

distancias = []
for i in range(len(distorcoes)):
    x = intervalo_k[i]
    y = distorcoes[i]
    
    # Fórmula da distância (ponto-reta)
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0)**2 + (xn - x0)**2)
    
    distancias.append(numerador / denominador)

# K ótimo é onde a distância é máxima (maior "cotovelo")
k_otimo = intervalo_k[distancias.index(np.max(distancias))]

print(f"\nK (Elbow Method): {k_otimo}")
print(f"   Distância máxima: {np.max(distancias):.4f}")

# Validando com Shilhouet 
montar_divisor("CALCULANDO SILHOUETTE SCORE", 60, "-")

scores_silhouette = []

for k in intervalo_k:
    modelo_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    rotulos = modelo_kmeans.fit_predict(dados)
    
    score = silhouette_score(dados, rotulos)
    scores_silhouette.append(score)
    
    print(f"k={k}: silhouette={score:.4f}")

# K com melhor silhouette
k_melhor_silhouette = intervalo_k[scores_silhouette.index(max(scores_silhouette))]

print(f"\nK COM MELHOR SILHOUETTE: {k_melhor_silhouette}")
print(f"   Score: {max(scores_silhouette):.4f}")

montar_divisor("RECOMENDAÇÃO FINAL", 60, "-")

print(f"K ótimo (Elbow):      {k_otimo}")
print(f"K ótimo (Silhouette): {k_melhor_silhouette}")

if k_otimo == k_melhor_silhouette:
    k_recomendado = k_otimo
    print(f"\nAMBOS OS MÉTODOS CONFIRMAM!")
    print(f"   K RECOMENDADO: {k_recomendado}")
else:
    k_recomendado = k_otimo  # Priorizar Elbow
    print(f"\nPriorize o cotovelo (Elbow Method).")
    print(f"   K RECOMENDADO: {k_recomendado}")

# Plotando
montar_divisor("GERANDO GRÁFICOS", 60, "-")

# Gráficoo do Cotovelo
plotar_cotovelo(intervalo_k, distorcoes, k_otimo, '../plots/elbow_method.png')

# Gráfico do Silhouette Score
plotar_silhouette(intervalo_k, scores_silhouette, k_melhor_silhouette, 
                  '../plots/silhouette_scores.png')

print(f"\nTreinar K-Means com k={k_recomendado}")