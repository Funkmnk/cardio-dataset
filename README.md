# Clusterização de Risco Cardiovascular

Projeto de K-Means para segmentar pacientes com insuficiência cardíaca em 9 grupos de risco com base em perfis clínicos.

## Resultados Principais

Os 9 clusters foram agrupados em 3 níveis de risco com base na taxa de mortalidade (`DEATH_EVENT`):

* **ALTO RISCO (Clusters 0, 3, 8):**
    * Mortalidade: **55.3% - 84.6%**
    * Perfil: Idade avançada, creatinina sérica elevada (Média de 5.0 no Cluster 0), baixa fração de ejeção.

* **MÉDIO RISCO (Clusters 1, 2, 6):**
    * Mortalidade: **30.8% - 50.0%**

* **BAIXO RISCO (Clusters 4, 5, 7):**
    * Mortalidade: **5.9% - 13.5%**
    * Perfil: Maior tempo de acompanhamento (`time`), creatinina baixa, fração de ejeção preservada.



## Como Prever o Risco

O script `analyses/6_predicao.py` carrega os modelos treinados (normalizador e k-means) para classificar novos pacientes.

**Exemplo de Paciente (Alto Risco):**
* **Entrada:** `{'age': 75, 'serum_creatinine': 4.5, 'ejection_fraction': 25, ...}`
* **Predição:** Cluster 0
* **Resultado:** Risco **ALTO** (Mortalidade esperada: 84.6%)


## Pipeline do Projeto
1.  `1_EDA.py`: Análise Exploratória.
2.  `2_pre_process.py`: Normalização dos dados (StandardScaler).
3.  `3_determinar_k.py`: Método Elbow (resultou em K=9).
4.  `4_treinar_modelo.py`: Treinamento do K-Means (K=9).
5.  `5_descrever_clusters.py`: Análise estatística (ANOVA) e perfil dos grupos.
6.  `6_predicao.py`: Função para prever o risco de novos pacientes.

---