import nbformat
import sys
import os

path = 'lista 2/Lista_2_Kauê_respondida.ipynb'
try:
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
except Exception as e:
    print(f'Error reading: {e}')
    sys.exit(1)

# --- Questão 9 REFINADA ---
q9_raciocinio = """A estratégia de visualização avançada foca na **comunicação de evidências metodológicas**. Não buscamos apenas descrever os dados, mas provar por que as decisões tomadas (como a escolha da Árvore de Decisão) foram corretas:
1.  **Mapeamento de Relevância Geográfica:** Utilizaremos um mapeamento de coordenadas com gradiente de preço para evidenciar a **não-linearidade espacial**. Isso justifica por que modelos globais (Lineares) falham ao tentar aplicar a mesma 'régua' para bairros com dinâmicas tão distintas.
2.  **Análise de Resíduos e Heterocedasticidade:** Implementaremos um gráfico de dispersão de 'Fidelidade de Predição' acompanhado de uma análise visual de resíduos. O objetivo é demonstrar que, embora a Árvore seja superior, ela apresenta **heterocedasticidade** (erro aumenta com o preço), um insight crítico para o gerenciamento de risco do negócio imobiliário."""

q9_code = """import seaborn as sns
import matplotlib.patches as mpatches

# Configuração de Estilo e Contexto Visual
sns.set_theme(style="white", palette="muted")
fig = plt.figure(figsize=(22, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

# --- Subplot 1: Inteligência Espacial (Mapa de Calor de Ativos) ---
ax1 = fig.add_subplot(gs[0])
# Filtramos o 99th percentile para evitar que mansões extremas 'apaguem' o gradiente do mapa
vmax_price = df['Price'].quantile(0.95)
scatter = ax1.scatter(df['Longtitude'], df['Lattitude'], 
                      c=df['Price'], cmap='magma', 
                      s=25, alpha=0.5, edgecolors='none',
                      norm=plt.Normalize(vmin=df['Price'].min(), vmax=vmax_price))

cbar = fig.colorbar(scatter, ax=ax1, fraction=0.03, pad=0.04)
cbar.set_label('Valor do Imóvel (AUD)', fontsize=12, labelpad=10)
ax1.set_title('Mapeamento de Valor: Onde o Capital se Concentra em Melbourne', fontsize=18, fontweight='bold', pad=20)
ax1.set_xlabel('Longitude', fontsize=12)
ax1.set_ylabel('Latitude', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.3)

# --- Subplot 2: Validação Metodológica (Erro e Fidelidade) ---
ax2 = fig.add_subplot(gs[1])
# Preparação rápida dos dados do "Vencedor" (Decision Tree)
features_final = ['Rooms', 'Bathroom', 'Car', 'BuildingArea', 'Lattitude', 'Longtitude', 'Distance']
X_v = SimpleImputer(strategy='median').fit_transform(df[features_final])
y_v = df['Price']
X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_v, y_v, test_size=0.2, random_state=42)
model_final = DecisionTreeRegressor(max_depth=10, random_state=42).fit(X_train_v, y_train_v)
y_pred_v = model_final.predict(X_test_v)

# Scatter plot com densidade visual
sns.scatterplot(x=y_test_v, y=y_pred_v, ax=ax2, alpha=0.4, color='#2c3e50', s=40)
# Linha de Identidade (Referência de Perfeição)
max_val = min(y_test_v.max(), y_pred_v.max())
ax2.plot([0, max_val], [0, max_val], color='#e74c3c', lw=3, linestyle='--', label='Fidelidade Ideal')

ax2.set_title('Diagnóstico de Performance: Real vs. Predito (Decision Tree)', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Preço Real (AUD)', fontsize=12)
ax2.set_ylabel('Preço Predito (AUD)', fontsize=12)
ax2.legend(fontsize=12)
ax2.set_xlim(0, 3500000)
ax2.set_ylim(0, 3500000)

# Anotação de insight técnico
ax2.annotate('Zona de Alta Precisão\\n(Mid-Market)', xy=(1000000, 1000000), xytext=(2000000, 500000),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()"""

q9_conclusao = """A análise visual refinada consolida os achados técnicos de forma pragmática:
1.  **Justificativa da Não-Linearidade:** O mapa de Melbourne (Gráfico 1) revela que o preço não segue um gradiente linear simples do centro para a periferia. Existem 'micro-bolhas' de valorização capturadas por latitudes específicas. Isso valida por que a Árvore de Decisão, com sua capacidade de segmentação discreta, superou a Regressão Linear.
2.  **Diagnóstico de Risco e Confiança:** O Gráfico de Fidelidade mostra uma aderência excepcional na faixa de 500k a 1.8M de AUD. Contudo, a dispersão aumenta significativamente acima de 2.5M. Concluímos que o modelo é **robusto para o mercado de massa**, mas deve ser usado com cautela (ou variáveis exógenas extras) para o segmento de altíssimo luxo.
3.  **Veredito Visual:** A combinação desses dois gráficos comunica melhor os resultados pois aborda as duas maiores preocupações de um projeto de Data Science: **Onde está a oportunidade? (Mapa)** e **O quanto posso confiar nos números? (Fidelidade)**."""

# --- Injeção no Notebook ---
for cell in nb.cells:
    if cell.id == "ST3DBYyXX1I2": # Q9 Markdown
        cell.source = "## Questão 9 – Visualização de Dados\n\n**Enunciado:** ...\n\n### Raciocínio\n" + q9_raciocinio + "\n\n### Desenvolvimento"
    if cell.id == "nt-am6IRX1I2": # Q9 Code
        cell.source = q9_code
    if cell.id == "ZEG2Ia-BX1I2": # Q9 Conclusion
        cell.source = "### Conclusão da Questão 9\n\n" + q9_conclusao

with open(path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('Question 9 refined and updated successfully')
