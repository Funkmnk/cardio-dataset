# Biblioteca
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pickle import dump
from utils import montar_divisor

# Carregar dados
dados = pd.read_csv('../data/heart_failure_clinical_records_dataset.csv')
dados.columns = dados.columns.str.strip()

montar_divisor("PRÉ-PROCESSAMENTO", 60, "=")

montar_divisor("REMOVENDO VARIÁVEL FOCO", 60, "-")

# Salvando DEATH_EVENT
alvo = dados['DEATH_EVENT'].copy()
print(f"Target salvo: {len(alvo)} registros")

# Remover do dataset de treino
dados_treino = dados.drop('DEATH_EVENT', axis=1)
print(f"Dataset de treino: {dados_treino.shape}")
print(f"Colunas: {list(dados_treino.columns)}")

# Normalizando
montar_divisor("NORMALIZANDO DADOS", 60, "-")

# Criar normalizador
normalizador = StandardScaler()

# Ajustar e transformar os dados
dados_normalizados = normalizador.fit_transform(dados_treino)

# Converter de volta para DataFrame (mantém nomes das colunas)
df_normalizado = pd.DataFrame(
    dados_normalizados, 
    columns=dados_treino.columns
)

print("Dados normalizados")
print(f"Dimensões: {df_normalizado.shape}")

# Vendo a normalização (média ~0, desvio ~1)
print("Médias (0?):")
print(df_normalizado.mean().round(4))
print("\nDesvios padrão (1?):")
print(df_normalizado.std().round(4))

# Salvando dados preprocessados
df_normalizado.to_csv('../data/processados/dados_preprocessados.csv', index=False)

# Salvando target separadamente
alvo.to_csv('../data/processados/target.csv', index=False)

# Salvando modelo do normalizador
dump(normalizador, open('../models/normalizador.model', 'wb'))

montar_divisor("ESTATÍSTICAS DOS DADOS (Normalizados)", 60, "-")
print(df_normalizado.describe().round(2))