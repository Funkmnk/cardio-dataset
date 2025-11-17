# Imports
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pickle import dump
from utils import montar_divisor, plotar_distribuicao_clusters, plotar_clusters_pca

# Carregando dados
dados = pd.read_csv('../data/processados/dados_preprocessados.csv')

montar_divisor("TREINAMENTO DO MODELO (K-MEANS)", 60, "=")

montar_divisor("TREINANDO MODELO", 60, "-")

k_otimo = 9
print(f"Número de clusters (k): {k_otimo}")

# Criando e treinando o modelo
modelo_kmeans = KMeans(n_clusters=k_otimo, random_state=42, n_init=10, max_iter=300)
modelo_kmeans.fit(dados)

print(f"Modelo treinado com sucesso!")
print(f"   Iterações até convergência: {modelo_kmeans.n_iter_}")
print(f"   Inércia final: {modelo_kmeans.inertia_:.4f}")

# Rotulando os clusters
montar_divisor("ATRIBUINDO CLUSTERS", 60, "-")

rotulos = modelo_kmeans.labels_
centroides = modelo_kmeans.cluster_centers_

# Contar pacientes por cluster
contagem_clusters = pd.Series(rotulos).value_counts().sort_index()

print("Distribuição de pacientes por cluster:")
for cluster, qtd in contagem_clusters.items():
    percentual = (qtd / len(dados)) * 100
    print(f"  Cluster {cluster}: {qtd} pacientes ({percentual:.1f}%)")

# Validação
montar_divisor("MÉTRICAS DE VALIDAÇÃO", 60, "-")

# Silhouette Score (quanto maior, melhor: -1 a 1)
score_silhouette = silhouette_score(dados, rotulos)
print(f"Silhouette Score: {score_silhouette:.4f}")
print(f"  Interpretação: ", end="")
if score_silhouette > 0.7:
    print("Excelente separação")
elif score_silhouette > 0.5:
    print("Boa separação")
elif score_silhouette > 0.25:
    print("Separação fraca")
else:
    print("Separação ruim")

# Davies-Bouldin Index (quanto menor, melhor)
indice_davies_bouldin = davies_bouldin_score(dados, rotulos)
print(f"\nDavies-Bouldin Index: {indice_davies_bouldin:.4f}")
print(f"  Interpretação: Quanto menor, melhor (ideal < 1.0)")

# Calinski-Harabasz Index (quanto maior, melhor)
indice_calinski_harabasz = calinski_harabasz_score(dados, rotulos)
print(f"\nCalinski-Harabasz Index: {indice_calinski_harabasz:.4f}")
print(f"  Interpretação: Quanto maior, melhor (clusters mais densos e separados)")

# Adicionando clusters ao dataset
montar_divisor("SALVANDO RESULTADOS", 60, "-")

# Adicionar coluna de cluster ao dataset normalizado
dados_com_clusters = dados.copy()
dados_com_clusters['cluster'] = rotulos

# Salvar dataset com clusters
dados_com_clusters.to_csv('../data/dados_clusterizados.csv', index=False)

# Adicionando clusters aos dados originais
dados_originais = pd.read_csv('../data/heart_failure_clinical_records_dataset.csv')
dados_originais.columns = dados_originais.columns.str.strip()
dados_originais_com_clusters = dados_originais.copy()
dados_originais_com_clusters['cluster'] = rotulos

dados_originais_com_clusters.to_csv('../data/clusterizados/dados_originais_com_clusters.csv', index=False)

# Salvar modelo treinado
dump(modelo_kmeans, open('../models/modelo_kmeans.model', 'wb'))

# DataFrame com centroides (dados normalizados)
df_centroides = pd.DataFrame(centroides, columns=dados.columns)
df_centroides['cluster'] = range(k_otimo)
df_centroides.to_csv('../data/clusterizados/centroides_normalizados.csv', index=False)

# Plotagem
montar_divisor("PLOTANDO A CLUSTERIZAÇÃO", 60, "-")

# Gráfico 1: Distribuição de pacientes por cluster
plotar_distribuicao_clusters(rotulos, k_otimo, '../plots/distribuicao_clusters.png')

# Gráfico 2: Clusters no espaço PCA (2D)
plotar_clusters_pca(dados.values, rotulos, centroides, '../plots/clusters_pca.png')

# Apresentação do treinamento
montar_divisor("ESPECIFICAÇÕES DO TREINAMENTO", 60, "=")

print(f"""

Configuração:
  - Algoritmo: K-Means
  - Número de clusters: {k_otimo}
  - Total de pacientes: {len(dados)}
  - Features utilizadas: {len(dados.columns)}

Métricas de Qualidade:
  - Silhouette Score: {score_silhouette:.4f}
  - Davies-Bouldin Index: {indice_davies_bouldin:.4f}
  - Calinski-Harabasz Index: {indice_calinski_harabasz:.2f}
""")