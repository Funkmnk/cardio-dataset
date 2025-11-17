# Biblioteca
import pandas as pd
import numpy as np
from pickle import load
from utils import montar_divisor, plotar_comparacao_paciente_cluster

# Carregar modelos e dados
modelo_kmeans = load(open('../models/modelo_kmeans.model', 'rb'))
normalizador = load(open('../models/normalizador.model', 'rb'))
df_centroides = pd.read_csv('../data/clusterizados/descricao/centroides_originais.csv')
df_resumo = pd.read_csv('../data/clusterizados/descricao/resumo_clusters.csv')

montar_divisor("PREDIÇÃO DE NOVOS PACIENTES", 60, "=")

montar_divisor("DIVIDINDO OS NÍVEIS DE RISCOS", 60, "-")

# Clusters por nível de risco
mapa_riscos = {
    'ALTO': [0, 3, 8],
    'MÉDIO': [1, 2, 6],
    'BAIXO': [4, 5, 7]
}

print("CLASSIFICAÇÃO DE RISCO:\n")
for nivel, clusters in mapa_riscos.items():
    mortalidades = df_resumo[df_resumo['cluster'].isin(clusters)]['taxa_mortalidade'].values
    print(f"{nivel:6s}: Clusters {clusters} - Mortalidade {mortalidades.min():.1f}% - {mortalidades.max():.1f}%")

# Função de predição
def prever_cluster_paciente(dados_paciente, mostrar_detalhes=True):
    """
    Classifica um novo paciente em um cluster
    
    Args:
        dados_paciente: dicionário com features do paciente
        mostrar_detalhes: se True, imprime informações detalhadas
    
    Returns:
        cluster_predito, nivel_risco, taxa_mortalidade
    """
    # Converte para DataFrame
    df_paciente = pd.DataFrame([dados_paciente])
    
    # Ordenação das colunas
    colunas_treino = normalizador.feature_names_in_
    df_paciente = df_paciente[colunas_treino]
    
    # Normalizar
    dados_normalizados = pd.DataFrame(normalizador.transform(df_paciente), columns=colunas_treino)
    
    # Prever cluster
    cluster_predito = modelo_kmeans.predict(dados_normalizados)[0]
    
    # Informações do cluster
    info_cluster = df_resumo[df_resumo['cluster'] == cluster_predito].iloc[0]
    taxa_mortalidade = info_cluster['taxa_mortalidade']
    
    # Nível de risco
    if cluster_predito in mapa_riscos['ALTO']:
        nivel_risco = 'ALTO'
    elif cluster_predito in mapa_riscos['MÉDIO']:
        nivel_risco = 'MÉDIO'
    else:
        nivel_risco = 'BAIXO'
    
    if mostrar_detalhes:
        print("\n" + "="*60)
        print("RESULTADO DA PREDIÇÃO")
        print("="*60)
        
        print(f"\nCLUSTER ATRIBUÍDO: {cluster_predito}")
        print(f"   Nível de risco: {nivel_risco}")
        print(f"   Taxa de mortalidade: {taxa_mortalidade:.1f}%")
        print(f"   Pacientes no cluster: {int(info_cluster['n_pacientes'])}")
        
        print(f"\nPERFIL DO CLUSTER {cluster_predito}:")
        centroide = df_centroides[df_centroides['cluster'] == cluster_predito].iloc[0]
        for col in colunas_treino:
            valor_centroide = centroide[col]
            valor_paciente = dados_paciente[col]
            diff = valor_paciente - valor_centroide
            simbolo = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"   {col:30s}: {valor_centroide:6.2f} | Paciente: {valor_paciente:6.2f} {simbolo}")
    
    return cluster_predito, nivel_risco, taxa_mortalidade

# Paciente de alto risco
montar_divisor("EXEMPLO 1: PACIENTE DE ALTO RISCO", 60, "-")

paciente_alto_risco = {
    'age': 75,
    'anaemia': 1,
    'creatinine_phosphokinase': 800,
    'diabetes': 1,
    'ejection_fraction': 25,
    'high_blood_pressure': 1,
    'platelets': 250000,
    'serum_creatinine': 4.5,
    'serum_sodium': 130,
    'sex': 1,
    'smoking': 1,
    'time': 50
}

print("DADOS DO PACIENTE:")
for chave, valor in paciente_alto_risco.items():
    print(f"   {chave:30s}: {valor}")

cluster1, risco1, mort1 = prever_cluster_paciente(paciente_alto_risco)

# Risco moderado
montar_divisor("EXEMPLO 3: PACIENTE DE RISCO MÉDIO", 60, "-")

paciente_medio_risco = {
    'age': 62,
    'anaemia': 0,
    'creatinine_phosphokinase': 450,
    'diabetes': 1,
    'ejection_fraction': 35,
    'high_blood_pressure': 0,
    'platelets': 260000,
    'serum_creatinine': 1.3,
    'serum_sodium': 136,
    'sex': 1,
    'smoking': 0,
    'time': 120
}

print("\nDADOS DO PACIENTE:")
for chave, valor in paciente_medio_risco.items():
    print(f"   {chave:30s}: {valor}")

cluster3, risco3, mort3 = prever_cluster_paciente(paciente_medio_risco)

# Baixo risco
montar_divisor("EXEMPLO 2: PACIENTE DE BAIXO RISCO", 60, "-")

paciente_baixo_risco = {
    'age': 55,
    'anaemia': 0,
    'creatinine_phosphokinase': 200,
    'diabetes': 0,
    'ejection_fraction': 50,
    'high_blood_pressure': 1,
    'platelets': 280000,
    'serum_creatinine': 1.0,
    'serum_sodium': 138,
    'sex': 1,
    'smoking': 0,
    'time': 180
}

print("\nDADOS DO PACIENTE:")
for chave, valor in paciente_baixo_risco.items():
    print(f"   {chave:30s}: {valor}")

cluster2, risco2, mort2 = prever_cluster_paciente(paciente_baixo_risco)

# Plotando
montar_divisor("PLOTANDO", 60, "-")

plotar_comparacao_paciente_cluster(paciente_alto_risco, cluster1, df_centroides, '../plots/predicao/predicao_paciente_alto_risco.png')

plotar_comparacao_paciente_cluster(paciente_baixo_risco, cluster2, df_centroides, '../plots/predicao/predicao_paciente_baixo_risco.png')

plotar_comparacao_paciente_cluster(paciente_medio_risco, cluster3, df_centroides, '../plots/predicao/predicao_paciente_medio_risco.png')

# Apresentação final
montar_divisor("RESUMO DAS PREDIÇÕES", 60, "=")

print(f"""

Exemplo 1 (Alto Risco):
  - Cluster: {cluster1} | Risco: {risco1} | Mortalidade: {mort1:.1f}%
  - Perfil: Idoso, creatinina alta (4.5), fumante, EF (sangue bombeado) baixa (25%)

Exemplo 2 (Baixo Risco):
  - Cluster: {cluster2} | Risco: {risco2} | Mortalidade: {mort2:.1f}%
  - Perfil: Jovem, função renal boa, EF (sangue bombeado) preservada (50%)

Exemplo 3 (Médio Risco):
  - Cluster: {cluster3} | Risco: {risco3} | Mortalidade: {mort3:.1f}%
  - Perfil: Meia-idade, diabético, EF (sangue bombeado) moderada (35%)
""")