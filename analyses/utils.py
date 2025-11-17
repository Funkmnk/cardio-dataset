# Função de formatação
def montar_divisor(texto, tamanho, simbolo):
	print("\n" + simbolo * tamanho)
	print(" " * ((tamanho - len(texto)) // 2) + texto)
	print(simbolo * tamanho + "\n")

# Funções de plotagem
import matplotlib.pyplot as plt
import numpy as np

def plotar_cotovelo(intervalo_k, distorcoes, k_otimo, caminho_saida):
    """
    Plota o gráfico do Método do Cotovelo
    
    Args:
        intervalo_k: range ou lista com valores de k testados
        distorcoes: lista com valores de distorção para cada k
        k_otimo: valor de k ótimo encontrado
        caminho_saida: caminho para salvar o gráfico
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Curva de distorções
    ax.plot(list(intervalo_k), distorcoes, 'bo-', linewidth=2, markersize=8)
    
    # Destacar k ótimo
    indice_otimo = list(intervalo_k).index(k_otimo)
    ax.plot(k_otimo, distorcoes[indice_otimo], 'ro', markersize=15, 
            label=f'K Ótimo = {k_otimo}')
    
    # Linha vertical no k ótimo
    ax.axvline(x=k_otimo, color='r', linestyle='--', alpha=0.5)
    
    # Configs
    ax.set_xlabel('Número de clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distorção', fontsize=12, fontweight='bold')
    ax.set_title('Método do cotovelo (Elbow Method)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()


def plotar_silhouette(intervalo_k, scores, k_melhor, caminho_saida):
    """
    Plota o gráfico do Silhouette Score
    
    Args:
        intervalo_k: range ou lista com valores de k testados
        scores: lista com silhouette scores para cada k
        k_melhor: valor de k com melhor score
        caminho_saida: caminho para salvar o gráfico
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotar scores
    ax.plot(list(intervalo_k), scores, 'go-', linewidth=2, markersize=8)
    
    # Destacar melhor k
    indice_melhor = list(intervalo_k).index(k_melhor)
    ax.plot(k_melhor, scores[indice_melhor], 'ro', markersize=15, 
            label=f'Melhor k = {k_melhor} (score={scores[indice_melhor]:.3f})')
    
    # Linha vertical no melhor k
    ax.axvline(x=k_melhor, color='r', linestyle='--', alpha=0.5)
    
    # Linha horizontal em 0.5
    ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5, 
               label='Limiar 0.5 (qualidade aceitável)')
    
    # Configs
    ax.set_xlabel('Número de clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title('Silhouette Score por número de clusters', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    
# Funções de platagem da clusterização
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def plotar_distribuicao_clusters(rotulos, k, caminho_saida):
    """
    Plota a distribuição de pacientes por cluster
    
    Args:
        rotulos: array com rótulos dos clusters
        k: número total de clusters
        caminho_saida: caminho para salvar o gráfico
    """
    import pandas as pd
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Distribuição de pacientes por cluster', 
                 fontsize=14, fontweight='bold')
    
    # Contar pacientes por cluster
    contagem = pd.Series(rotulos).value_counts().sort_index()
    cores = plt.cm.tab10(np.linspace(0, 1, k))
    
    # Gráfico de barras
    axes[0].bar(range(k), contagem.values, color=cores, edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Cluster', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Número de pacientes', fontsize=11, fontweight='bold')
    axes[0].set_title('Contagem por cluster')
    axes[0].set_xticks(range(k))
    axes[0].grid(axis='y', alpha=0.3)
    
    # Adiciona os valores nas barras
    for i, v in enumerate(contagem.values):
        percentual = (v / len(rotulos)) * 100
        axes[0].text(i, v + 1, f'{v}\n({percentual:.1f}%)', 
                    ha='center', fontweight='bold', fontsize=9)
    
    # Gráfico de pizza
    axes[1].pie(contagem.values, 
                labels=[f'Cluster {i}' for i in range(k)],
                autopct='%1.1f%%',
                colors=cores,
                startangle=90,
                shadow=True)
    axes[1].set_title('Proporção de pacientes')
    
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()


def plotar_clusters_pca(dados, rotulos, centroides, caminho_saida):
    """
    Reduz dimensionalidade com PCA e plota clusters em 2D
    
    Args:
        dados: array com dados normalizados
        rotulos: array com rótulos dos clusters
        centroides: array com centroides dos clusters
        caminho_saida: caminho para salvar o gráfico
    """
    # Aplicar PCA para reduzir para 2 dimensões
    pca = PCA(n_components=2, random_state=42)
    dados_pca = pca.fit_transform(dados)
    centroides_pca = pca.transform(centroides)
    
    # Criar gráfico
    fig, ax = plt.subplots(figsize=(12, 8))
    
    k = len(np.unique(rotulos))
    cores = plt.cm.tab10(np.linspace(0, 1, k))
    
    # Plotar pontos por cluster
    for cluster in range(k):
        mascara = rotulos == cluster
        ax.scatter(dados_pca[mascara, 0], 
                  dados_pca[mascara, 1],
                  c=[cores[cluster]],
                  label=f'Cluster {cluster}',
                  alpha=0.6,
                  s=50,
                  edgecolors='black',
                  linewidth=0.5)
    
    # Plotar centroides
    ax.scatter(centroides_pca[:, 0],
              centroides_pca[:, 1],
              c='red',
              marker='X',
              s=300,
              edgecolors='black',
              linewidth=2,
              label='Centroides',
              zorder=10)
    
    # Adicionar números dos centroides
    for i, (x, y) in enumerate(centroides_pca):
        ax.text(x, y, str(i), 
               fontsize=12, fontweight='bold', 
               ha='center', va='center',
               color='white')
    
    # Configurações
    variancia_explicada = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({variancia_explicada[0]*100:.1f}% variância)', 
                 fontsize=11, fontweight='bold')
    ax.set_ylabel(f'PC2 ({variancia_explicada[1]*100:.1f}% variância)', 
                 fontsize=11, fontweight='bold')
    ax.set_title('Visualização dos clusters em 2D (PCA)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    
# Funçõse de descrição de cluster
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plotar_centroides_heatmap(df_centroides, nomes_features, caminho_saida):
    """
    Plota heatmap dos centroides (valores médios de cada cluster)
    
    Args:
        df_centroides: DataFrame com centroides
        nomes_features: lista com nomes das features
        caminho_saida: caminho para salvar o gráfico
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Preparar dados (remover coluna cluster)
    dados_heatmap = df_centroides[nomes_features].T
    dados_heatmap.columns = [f'Cluster {i}' for i in range(len(df_centroides))]
    
    # Normalizar para visualização (0-1 por feature)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    dados_normalizados = pd.DataFrame(
        scaler.fit_transform(dados_heatmap),
        index=dados_heatmap.index,
        columns=dados_heatmap.columns
    )
    
    # Criar heatmap
    sns.heatmap(dados_normalizados,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0.5,
                linewidths=0.5,
                cbar_kws={'label': 'Valor normalizado (0-1)'},
                ax=ax)
    
    ax.set_title('Heatmap dos centroides (perfil médio por cluster)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Clusters', fontsize=11, fontweight='bold')
    ax.set_ylabel('Features', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()


def plotar_perfil_clusters(df_centroides, features_principais, caminho_saida):
    """
    Plota gráfico de barras comparando features principais entre clusters
    
    Args:
        df_centroides: DataFrame com centroides
        features_principais: lista com features mais importantes
        caminho_saida: caminho para salvar o gráfico
    """
    n_features = len(features_principais)
    n_clusters = len(df_centroides)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Perfil dos clusters - features principais', 
                 fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    cores = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for idx, feature in enumerate(features_principais):
        ax = axes[idx]
        
        valores = df_centroides[feature].values
        clusters = [f'C{i}' for i in range(n_clusters)]
        
        barras = ax.bar(clusters, valores, color=cores, edgecolor='black', alpha=0.8)
        
        # Adicionar valores nas barras
        for barra in barras:
            altura = barra.get_height()
            ax.text(barra.get_x() + barra.get_width()/2., altura,
                   f'{altura:.1f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_title(feature, fontweight='bold', fontsize=11)
        ax.set_xlabel('Cluster', fontsize=10)
        ax.set_ylabel('Valor médio', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()


def plotar_mortalidade_clusters(df_resumo, caminho_saida):
    """
    Plota taxa de mortalidade por cluster
    
    Args:
        df_resumo: DataFrame com resumo dos clusters
        caminho_saida: caminho para salvar o gráfico
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Taxa de mortalidade por cluster', fontsize=14, fontweight='bold')
    
    # Ordenar por mortalidade
    df_ordenado = df_resumo.sort_values('taxa_mortalidade', ascending=False)
    
    # Cores baseadas em risco
    cores = []
    for taxa in df_ordenado['taxa_mortalidade']:
        if taxa > 40:
            cores.append('#FF4444')  # Vermelho (alto risco)
        elif taxa > 25:
            cores.append('#FFA500')  # Laranja (médio risco)
        else:
            cores.append('#90EE90')  # Verde (baixo risco)
    
    # Gráfico 1: Barras
    clusters = [f"Cluster {int(c)}" for c in df_ordenado['cluster']]
    axes[0].barh(clusters, df_ordenado['taxa_mortalidade'], color=cores, 
                 edgecolor='black', alpha=0.8)
    
    # Adicionar valores
    for i, (cluster, taxa) in enumerate(zip(clusters, df_ordenado['taxa_mortalidade'])):
        axes[0].text(taxa + 1, i, f'{taxa:.1f}%', 
                    va='center', fontweight='bold', fontsize=9)
    
    axes[0].set_xlabel('Taxa de mortalidade (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Clusters ordenados por risco', fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].axvline(x=32.1, color='red', linestyle='--', alpha=0.5, 
                    label='Média Geral (32.1%)')
    axes[0].legend()
    
    # Gráfico 2: Mortes absolutas vs total
    df_plot = df_resumo.sort_values('cluster')
    x = np.arange(len(df_plot))
    width = 0.35
    
    axes[1].bar(x - width/2, df_plot['mortes'], width, 
                label='Óbitos', color='#FF6B6B', edgecolor='black', alpha=0.8)
    axes[1].bar(x + width/2, df_plot['n_pacientes'] - df_plot['mortes'], width,
                label='Sobreviventes', color='#90EE90', edgecolor='black', alpha=0.8)
    
    axes[1].set_xlabel('Cluster', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Número de pacientes', fontsize=11, fontweight='bold')
    axes[1].set_title('Óbitos vs sobreviventes', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'C{int(c)}' for c in df_plot['cluster']])
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()


def plotar_comparacao_features(dados_com_clusters, features, caminho_saida):
    """
    Plota boxplots comparando distribuições de features entre clusters
    
    Args:
        dados_com_clusters: DataFrame com dados originais e coluna cluster
        features: lista de features para comparar
        caminho_saida: caminho para salvar o gráfico
    """
    n_features = len(features)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribuição de features por cluster', fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        # Preparar dados para boxplot
        dados_plot = []
        labels = []
        for cluster in sorted(dados_com_clusters['cluster'].unique()):
            valores = dados_com_clusters[dados_com_clusters['cluster'] == cluster][feature]
            dados_plot.append(valores)
            labels.append(f'C{cluster}')
        
        # Criar boxplot
        bp = ax.boxplot(dados_plot, labels=labels, patch_artist=True)
        
        # Colorir boxes
        cores = plt.cm.tab10(np.linspace(0, 1, len(dados_plot)))
        for patch, cor in zip(bp['boxes'], cores):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)
        
        ax.set_title(feature, fontweight='bold', fontsize=11)
        ax.set_xlabel('Cluster', fontsize=10)
        ax.set_ylabel('Valor', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    
def plotar_anova_resultados(df_anova, caminho_saida):
    """
    Plota resultados do teste ANOVA (p-valores e significância)
    
    Args:
        df_anova: DataFrame com resultados do ANOVA
        caminho_saida: caminho para salvar o gráfico
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Resultados do teste ANOVA - significância estatística', 
                 fontsize=14, fontweight='bold')
    
    # Ordenar por p-valor
    df_plot = df_anova.sort_values('p_valor')
    
    # P-valores (escala log)
    cores = ['green' if p < 0.05 else 'red' for p in df_plot['p_valor']]
    
    axes[0].barh(range(len(df_plot)), df_plot['p_valor'], color=cores, 
                 edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0.05, color='orange', linestyle='--', linewidth=2, 
                    label='Limiar p=0.05')
    axes[0].set_yticks(range(len(df_plot)))
    axes[0].set_yticklabels(df_plot['feature'], fontsize=9)
    axes[0].set_xlabel('P-valor', fontsize=11, fontweight='bold')
    axes[0].set_title('P-valores por feature (menor = mais discriminativa)', fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Adicionar anotações
    for i, (p, nivel) in enumerate(zip(df_plot['p_valor'], df_plot['nivel'])):
        if nivel:
            axes[0].text(p, i, f' {nivel}', va='center', fontsize=10, fontweight='bold')
    
    # Estatística F
    cores_f = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_plot)))
    
    axes[1].barh(range(len(df_plot)), df_plot['F_statistic'], color=cores_f,
                 edgecolor='black', alpha=0.7)
    axes[1].set_yticks(range(len(df_plot)))
    axes[1].set_yticklabels(df_plot['feature'], fontsize=9)
    axes[1].set_xlabel('Estatística F', fontsize=11, fontweight='bold')
    axes[1].set_title('Estatística F (maior = maior diferença entre clusters)', fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Adicionar valores
    for i, f_val in enumerate(df_plot['F_statistic']):
        axes[1].text(f_val + 1, i, f'{f_val:.1f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()
    
# FUNÇÕES PARA ENTRADA DE PACIENTES
def plotar_comparacao_paciente_cluster(dados_paciente, cluster_predito, df_centroides, caminho_saida):
    """
    Compara valores do paciente com centroide do cluster
    
    Args:
        dados_paciente: dicionário com dados do paciente
        cluster_predito: cluster atribuído ao paciente
        df_centroides: DataFrame com centroides
        caminho_saida: caminho para salvar o gráfico
    """
    # Features principais para visualização
    features_principais = ['age', 'ejection_fraction', 'serum_creatinine', 
                          'serum_sodium', 'creatinine_phosphokinase', 'time']
    
    # Obter centroide
    centroide = df_centroides[df_centroides['cluster'] == cluster_predito].iloc[0]
    
    # Valores paciente vs centroide
    valores_paciente = [dados_paciente[f] for f in features_principais]
    valores_centroide = [centroide[f] for f in features_principais]
    
    # Criar gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(features_principais))
    largura = 0.35
    
    barras1 = ax.bar(x - largura/2, valores_centroide, largura, 
                     label=f'Centroide cluster {cluster_predito}',
                     color='steelblue', alpha=0.8, edgecolor='black')
    barras2 = ax.bar(x + largura/2, valores_paciente, largura,
                     label='Paciente novo',
                     color='coral', alpha=0.8, edgecolor='black')
    
    # Configurações
    ax.set_xlabel('Features', fontsize=11, fontweight='bold')
    ax.set_ylabel('Valor', fontsize=11, fontweight='bold')
    ax.set_title(f'Comparação: paciente vs Cluster {cluster_predito}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(features_principais, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Adicionar valores
    for barra in barras1:
        altura = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2., altura,
               f'{altura:.1f}', ha='center', va='bottom', fontsize=8)
    
    for barra in barras2:
        altura = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2., altura,
               f'{altura:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    plt.close()