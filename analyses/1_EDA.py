# Biliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from utils import montar_divisor

# Configs de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Carregandod dados
dados = pd.read_csv('../data/heart_failure_clinical_records_dataset.csv')
dados.columns = dados.columns.str.strip()

montar_divisor("ANÁLISE EXPLORATÓRIA", 60, "=")

# Infos básicas
montar_divisor("INFORMAÇÕES GERAIS DO DATASET", 60, "-")
print(f"Dimensões: {dados.shape[0]} linhas x {dados.shape[1]} colunas")
print(f"\nColunas disponíveis:\n{list(dados.columns)}")

# Infos detalhadas
montar_divisor("DADOS NULOS", 60, "-")
print(dados.info())

# Verificar valores ausentes
montar_divisor("VALORES AUSENTES POR COLUNA", 60, "-")
valores_nulos = dados.isnull().sum()
print(valores_nulos[valores_nulos > 0] if valores_nulos.sum() > 0 else "Não há valores nulos!")

# Descritiva
montar_divisor("ESTATÍSTICAS DESCRITIVAS", 60, "-")
print(dados.describe().round(2))

# Features binárias
variaveis_binarias = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'DEATH_EVENT']

# Features numéricas
variaveis_continuas = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                      'platelets', 'serum_creatinine', 'serum_sodium', 'time']

print(f"\nVARIÁVEIS BINÁRIAS: {len(variaveis_binarias)}")
print(variaveis_binarias)
print(f"VARIÁVEIS CONTÍNUAS: {len(variaveis_continuas)}")
print(variaveis_continuas)

# Distribuição das variáveis binárias
montar_divisor("DISTRIBUIÇÃO DAS VARIÁVEIS BINÁRIAS", 60, "-")
for col in variaveis_binarias:
    contagem = dados[col].value_counts()
    percentual = dados[col].value_counts(normalize=True) * 100
    print(f"{col}:")
    print(f"  0: {contagem.get(0, 0)} ({percentual.get(0, 0):.1f}%)")
    print(f"  1: {contagem.get(1, 0)} ({percentual.get(1, 0):.1f}%)\n")

# Identificação dos outliers
montar_divisor("IDENTIFICANDO OUTLIERS (Método IQR)", 60, "-")

resumo_outliers = {}
for col in variaveis_continuas:
    Q1 = dados[col].quantile(0.25)
    Q3 = dados[col].quantile(0.75)
    AIQ = Q3 - Q1
    limite_inferior = Q1 - 1.5 * AIQ
    limite_superior = Q3 + 1.5 * AIQ
    
    # Contar outliers
    outliers = dados[(dados[col] < limite_inferior) | (dados[col] > limite_superior)]
    num_outliers = len(outliers)
    pct_outliers = (num_outliers / len(dados)) * 100
    
    resumo_outliers[col] = {
        'num_outliers': num_outliers,
        'pct': pct_outliers,
        'limite_inferior': limite_inferior,
        'limite_superior': limite_superior
    }
    
    if num_outliers > 0:
        print(f"{col}: {num_outliers} outliers ({pct_outliers:.1f}%)")
        print(f"  Limites: [{limite_inferior:.2f}, {limite_superior:.2f}]")
    else:
        print(f"{col}: Sem outliers")

# Corr
montar_divisor("ANÁLISE DE CORRELAÇÃO", 60, "-")

# Matriz de correlação
matriz_correlacao = dados.corr()

# Correlações mais fortes com DEATH_EVENT
correlacoes_morte = matriz_correlacao['DEATH_EVENT'].sort_values(ascending=False)
print("Correlações com DEATH_EVENT (evento de morte):")
print(correlacoes_morte.round(3))

# Correlações fortes (> 0.7 ou < -0.7)
montar_divisor("CORRELAÇÕES FORTES (> 0.5)", 60, "-")
correlacoes_altas = []
for i in range(len(matriz_correlacao.columns)):
    for j in range(i+1, len(matriz_correlacao.columns)):
        col1 = matriz_correlacao.columns[i]
        col2 = matriz_correlacao.columns[j]
        valor_corr = matriz_correlacao.iloc[i, j]
        if abs(valor_corr) > 0.5 and col1 != 'DEATH_EVENT' and col2 != 'DEATH_EVENT':
            print(f"{col1} <-> {col2}: {valor_corr:.3f}")
            correlacoes_altas.append((col1, col2, valor_corr))

if not correlacoes_altas:
    print("Não há correlações fortes entre features")
    
# TODO: Transformar as plotagens em funções reaproveitáveis e plotar todos os dados