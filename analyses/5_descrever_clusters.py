# Imports
import pandas as pd
import numpy as np
from pickle import load
from scipy.stats import f_oneway
from utils import (montar_divisor, plotar_centroides_heatmap, 
                   plotar_perfil_clusters, plotar_mortalidade_clusters,
                   plotar_comparacao_features, plotar_anova_resultados)

# Carregar dados e modelos
dados_originais = pd.read_csv('../data/heart_failure_clinical_records_dataset.csv')
dados_originais.columns = dados_originais.columns.str.strip()

dados_com_clusters = pd.read_csv('../data/clusterizados/dados_originais_com_clusters.csv')
dados_com_clusters.columns = dados_com_clusters.columns.str.strip()

modelo_kmeans = load(open('../models/modelo_kmeans.model', 'rb'))
normalizador = load(open('../models/normalizador.model', 'rb'))

montar_divisor("DESCRIÇÃO DOS CLUSTERS", 60, "=")

# Revertendo a normalização
montar_divisor("REVERTENDO NORMALIZAÇÃO", 60, "-")

centroides_normalizados = modelo_kmeans.cluster_centers_
nomes_features = dados_com_clusters.drop(['DEATH_EVENT', 'cluster'], axis=1).columns.tolist()
centroides_originais = normalizador.inverse_transform(centroides_normalizados)

df_centroides = pd.DataFrame(centroides_originais, columns=nomes_features)
df_centroides['cluster'] = range(len(df_centroides))

print("Centroides revertidos!")
print(f"   Dimensões: {df_centroides.shape}")

df_centroides.to_csv('../data/clusterizados/descricao/centroides_originais.csv', index=False)

# ANNOVA
montar_divisor("ANÁLISE ANOVA - VALIDAÇÃO ESTATÍSTICA", 60, "-")

print("\nTESTANDO SIGNIFICÂNCIA ESTATÍSTICA\n")
print("H0: Médias iguais entre clusters (clusters NÃO são diferentes)")
print("H1: Pelo menos um cluster tem média diferente\n")

resultados_anova = []

for feature in nomes_features:
    # Valores por cluster
    grupos = []
    for cluster_id in range(len(df_centroides)):
        valores = dados_com_clusters[dados_com_clusters['cluster'] == cluster_id][feature]
        grupos.append(valores)
    
    # Executar ANOVA
    estatistica_f, p_valor = f_oneway(*grupos)
    
    # Interpretar resultado
    significativo = "SIM" if p_valor < 0.05 else "NÃO"
    nivel_sig = ""
    if p_valor < 0.001:
        nivel_sig = "***"
    elif p_valor < 0.01:
        nivel_sig = "**"
    elif p_valor < 0.05:
        nivel_sig = "*"
    
    resultados_anova.append({
        'feature': feature,
        'F_statistic': estatistica_f,
        'p_valor': p_valor,
        'significativo': significativo,
        'nivel': nivel_sig
    })
    
    print(f"{feature:30s} | F={estatistica_f:8.2f} | p={p_valor:.4f} {nivel_sig:3s} | {significativo}")

# DataFrame com resultados
df_anova = pd.DataFrame(resultados_anova)
df_anova = df_anova.sort_values('p_valor')

# Salvando resultados
df_anova.to_csv('../data/clusterizados/descricao/anova_resultados.csv', index=False)

montar_divisor("INTERPRETAÇÃO DO ANOVA", 60, "-")

features_significativas = df_anova[df_anova['p_valor'] < 0.05]
features_nao_significativas = df_anova[df_anova['p_valor'] >= 0.05]

print(f"\n   Features significativas (p < 0.05): {len(features_significativas)}/{len(nomes_features)}")
print(f"   Features não significativas: {len(features_nao_significativas)}/{len(nomes_features)}")

if len(features_significativas) > 0:
    print(f"\n FEATURES MAIS DISCRIMINATIVAS (melhor p-valor):\n")
    for idx, row in features_significativas.head(5).iterrows():
        print(f"   {idx+1}. {row['feature']}: p={row['p_valor']:.6f} {row['nivel']}")
    print("\n   - Estas features diferenciam MUITO BEM os clusters!")

if len(features_nao_significativas) > 0:
    print(f"\n  FEATURES NÃO DISCRIMINATIVAS (p >= 0.05):\n")
    for idx, row in features_nao_significativas.iterrows():
        print(f"   - {row['feature']}: p={row['p_valor']:.4f}")
    print("\n   - Estas features NÃO diferenciam bem os clusters")
    print("   - Clusters têm valores similares nestas variáveis")

# Estatística por cluster
montar_divisor("ESTATÍSTICAS POR CLUSTER", 60, "-")

for cluster_id in range(len(df_centroides)):
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*60}")
    
    pacientes_cluster = dados_com_clusters[dados_com_clusters['cluster'] == cluster_id]
    n_pacientes = len(pacientes_cluster)
    percentual = (n_pacientes / len(dados_com_clusters)) * 100
    
    print(f"\nTotal: {n_pacientes} pacientes ({percentual:.1f}%)")
    
    print(f"\nPERFIL MÉDIO (Centroide):")
    centroide = df_centroides[df_centroides['cluster'] == cluster_id].drop('cluster', axis=1).iloc[0]
    
    for feature in nomes_features:
        valor = centroide[feature]
        # Marcar features significativas
        sig = df_anova[df_anova['feature'] == feature]['nivel'].values[0]
        marcador = f" {sig}" if sig else ""
        print(f"  {feature}: {valor:.2f}{marcador}")
    
    mortes = pacientes_cluster['DEATH_EVENT'].sum()
    taxa_mortalidade = (mortes / n_pacientes) * 100
    print(f"\nMORTALIDADE:")
    print(f"  Óbitos: {mortes}/{n_pacientes} ({taxa_mortalidade:.1f}%)")

# Descrição dos clusters
montar_divisor("ANÁLISE COMPARATIVA ENTRE CLUSTERS", 60, "-")

resumo_clusters = []

for cluster_id in range(len(df_centroides)):
    pacientes = dados_com_clusters[dados_com_clusters['cluster'] == cluster_id]
    
    resumo = {
        'cluster': cluster_id,
        'n_pacientes': len(pacientes),
        'percentual': (len(pacientes) / len(dados_com_clusters)) * 100,
        'mortes': pacientes['DEATH_EVENT'].sum(),
        'taxa_mortalidade': (pacientes['DEATH_EVENT'].sum() / len(pacientes)) * 100,
        'idade_media': pacientes['age'].mean(),
        'fracao_ejecao_media': pacientes['ejection_fraction'].mean(),
        'creatinina_media': pacientes['serum_creatinine'].mean(),
        'tempo_medio': pacientes['time'].mean()
    }
    resumo_clusters.append(resumo)

df_resumo = pd.DataFrame(resumo_clusters)

print("\nRESUMO GERAL DOS CLUSTERS:\n")
print(df_resumo.round(2).to_string(index=False))

df_resumo.to_csv('../data/clusterizados/descricao/resumo_clusters.csv', index=False)

# Clusters de alto risco
montar_divisor("CLUSTERS DE ALTO RISCO", 60, "-")

df_resumo_ordenado = df_resumo.sort_values('taxa_mortalidade', ascending=False)

print("\nCLUSTERS ORDENADOS POR RISCO (Taxa de Mortalidade):\n")
for idx, row in df_resumo_ordenado.iterrows():
    nivel_risco = "ALTO" if row['taxa_mortalidade'] > 40 else "MÉDIO" if row['taxa_mortalidade'] > 25 else "BAIXO"
    print(f"Cluster {int(row['cluster'])}: {row['taxa_mortalidade']:.1f}% ({int(row['mortes'])}/{int(row['n_pacientes'])} óbitos) - {nivel_risco}")

# Interpretação
montar_divisor("INTERPRETAÇÃO", 60, "-")

cluster_mais_perigoso = df_resumo_ordenado.iloc[0]
print(f"1. CLUSTER MAIS PERIGOSO: Cluster {int(cluster_mais_perigoso['cluster'])}")
print(f"   - Taxa de mortalidade: {cluster_mais_perigoso['taxa_mortalidade']:.1f}%")

cluster_mais_seguro = df_resumo_ordenado.iloc[-1]
print(f"2. CLUSTER MAIS SEGURO: Cluster {int(cluster_mais_seguro['cluster'])}")
print(f"   - Taxa de mortalidade: {cluster_mais_seguro['taxa_mortalidade']:.1f}%")

cluster_maior = df_resumo.loc[df_resumo['n_pacientes'].idxmax()]
print(f"3. CLUSTER MAIS COMUM: Cluster {int(cluster_maior['cluster'])}")
print(f"   - Pacientes: {int(cluster_maior['n_pacientes'])} ({cluster_maior['percentual']:.1f}%)")
print(f"   - Taxa de mortalidade: {cluster_maior['taxa_mortalidade']:.1f}%\n")

cluster_menor = df_resumo.loc[df_resumo['n_pacientes'].idxmin()]
print(f"4. CLUSTER MAIS RARO: Cluster {int(cluster_menor['cluster'])}")
print(f"   - Pacientes: {int(cluster_menor['n_pacientes'])} ({cluster_menor['percentual']:.1f}%)")
print(f"   - Taxa de mortalidade: {cluster_menor['taxa_mortalidade']:.1f}%\n")

# Plotando
montar_divisor("GERANDO PLOTAGENS", 60, "-")

# Heatmap
plotar_centroides_heatmap(df_centroides, nomes_features, '../plots/descricao/centroides_heatmap.png')

features_principais = ['age', 'ejection_fraction', 'serum_creatinine', 'time']
plotar_perfil_clusters(df_centroides, features_principais, '../plots/descricao/perfil_clusters.png')

plotar_mortalidade_clusters(df_resumo, '../plots/descricao/mortalidade_clusters.png')

plotar_comparacao_features(dados_com_clusters, features_principais, '../plots/descricao/comparacao_features.png')

# Visualização dos resultados ANOVA
plotar_anova_resultados(df_anova, '../plots/descricao/anova_resultados.png')


montar_divisor("RESUMO DA ANÁLISE", 60, "=")

print(f"""

Total de clusters: 9
Total de pacientes: 299

Validação estatística (ANOVA):
  Features significativas: {len(features_significativas)}/{len(nomes_features)}
  Features não significativas: {len(features_nao_significativas)}/{len(nomes_features)}

Principais pontos:
  Cluster de maior risco: Cluster {int(cluster_mais_perigoso['cluster'])} ({cluster_mais_perigoso['taxa_mortalidade']:.1f}% mortalidade)
  Cluster de menor risco: Cluster {int(cluster_mais_seguro['cluster'])} ({cluster_mais_seguro['taxa_mortalidade']:.1f}% mortalidade)
  Cluster mais comum: Cluster {int(cluster_maior['cluster'])} ({cluster_maior['percentual']:.1f}% dos pacientes)
  Cluster mais raro: Cluster {int(cluster_menor['cluster'])} ({cluster_menor['percentual']:.1f}% dos pacientes)
""")

"""
Grupos de ALTO RISCO (clusters 0, 3, 8) = 55-85% mortalidade
Grupos MODERADOS (clusters 1, 2, 6) = 31-50% mortalidade
Grupos de BAIXO RISCO (clusters 4, 5, 7) = 6-14% mortalidade
"""