"""
Regressão linear básica (1 variável).

Ideia: temos pares (x, y) e queremos a reta que melhor descreve a relação.

    y_previsto = a + b * x

- a (intercepto): valor de y quando x = 0
- b (inclinação): quanto y muda quando x aumenta 1

Exemplo daqui: horas de estudo (x) vs nota na prova (y).
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1) Dados de exemplo
# Cada ponto é: (horas estudadas, nota)
# ---------------------------------------------------------------------------
horas = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
notas = np.array([32.0, 41.0, 48.0, 55.0, 63.0, 70.0, 76.0, 85.0])

print("Dados")
print("  horas (x):", horas)
print("  notas (y):", notas)
print()


# ---------------------------------------------------------------------------
# 2) Ajustar a reta com mínimos quadrados (fórmula fechada)
#
# Queremos minimizar o erro quadrático:
#     erro = soma( (y_real - y_previsto) ** 2 )
#
# Para uma reta, a solução é:
#     b = soma( (x - media_x) * (y - media_y) ) / soma( (x - media_x) ** 2 )
#     a = media_y - b * media_x
#
# Leitura:
# - numerador de b: covariância (x e y "andam juntos"?)
# - denominador de b: quanto x varia
# - se b > 0, quando x cresce, y tende a crescer
# ---------------------------------------------------------------------------
media_x = np.mean(horas)
media_y = np.mean(notas)

numerador = np.sum((horas - media_x) * (notas - media_y))
denominador = np.sum((horas - media_x) ** 2)

b = numerador / denominador
a = media_y - b * media_x

print("Modelo encontrado")
print(f"  a (intercepto) = {a:.2f}")
print(f"  b (inclinação) = {b:.2f}")
print(f"  equação: nota = {a:.2f} + {b:.2f} * horas")
print()
print("Interpretação")
print(f"  Sem estudar (0h), a reta estima nota ≈ {a:.1f}.")
print(f"  Cada hora a mais de estudo aumenta a nota em ≈ {b:.1f} pontos.")
print()


# ---------------------------------------------------------------------------
# 3) Prever e medir qualidade
# ---------------------------------------------------------------------------
def prever(x, intercepto, inclinacao):
    """Aplica a reta a um valor (ou array) de x."""
    return intercepto + inclinacao * x


notas_previstas = prever(horas, a, b)
residuos = notas - notas_previstas  # erro em cada ponto

# MSE: média dos erros ao quadrado (quanto menor, melhor)
mse = np.mean(residuos ** 2)

# R²: fração da variação de y que a reta explica (1 = perfeito, 0 = inútil)
ss_res = np.sum(residuos ** 2)
ss_tot = np.sum((notas - media_y) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("Qualidade do ajuste")
print(f"  MSE (erro quadrático médio) = {mse:.2f}")
print(f"  R² = {r2:.3f}  → a reta explica {r2 * 100:.1f}% da variação das notas")
print()

print("Comparação ponto a ponto")
print("  horas | nota real | prevista | erro")
for x, y_real, y_hat, erro in zip(horas, notas, notas_previstas, residuos):
    print(f"  {x:5.1f} | {y_real:9.1f} | {y_hat:8.1f} | {erro:+5.1f}")
print()


# ---------------------------------------------------------------------------
# 4) Usar o modelo em um valor novo
# ---------------------------------------------------------------------------
horas_novas = 4.5
nota_estimada = prever(horas_novas, a, b)
print("Previsão para um caso novo")
print(f"  Estudando {horas_novas} horas, a nota estimada é {nota_estimada:.1f}")
print()


# ---------------------------------------------------------------------------
# 5) Desenhar os pontos e a reta
# ---------------------------------------------------------------------------
x_linha = np.linspace(horas.min() - 0.5, horas.max() + 0.5, 100)
y_linha = prever(x_linha, a, b)

plt.figure(figsize=(8, 5))
plt.scatter(horas, notas, color="steelblue", s=70, label="dados reais", zorder=3)
plt.plot(x_linha, y_linha, color="tomato", linewidth=2, label="reta do modelo")
plt.xlabel("Horas de estudo (x)")
plt.ylabel("Nota (y)")
plt.title("Regressão linear: horas vs nota")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("grafico_regressao.png", dpi=120)
plt.show()
