import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import sys
import os
import pandas as pd



BURNIN = 0.3   # Este parametros es para descartar muestras al principio

# =========================
# Cargar la senal
# =========================

if not os.path.exists("senal.csv"):
    print("ERROR CRÍTICO: No se encontró el archivo 'senal.csv'.")
    print("Es obligatorio tener los datos experimentales para calcular el RMS de las cadenas.")
    sys.exit(1)

data = pd.read_csv("senal.csv")

try:
    col_x = [c for c in data.columns if c.lower() in ['x', 'tiempo_o_x', 'time', 'fre']][0]
    col_y = [c for c in data.columns if c.lower() in ['y', 'senal_con_ruido', 'temperature', 'amplitude']][0]
    x_data = data[col_x].values
    y_data = data[col_y].values
except IndexError:
    print(f"ERROR: No se pudieron mapear las columnas en 'senal.csv'.")
    print(f"Columnas disponibles: {list(data.columns)}")
    sys.exit(1)

# suma de guassianas 
def gaussian_mixture(x, A, mu, sigma):
    y_out = np.zeros_like(x)
    for Ai, mi, si in zip(A, mu, sigma):
        y_out += Ai * np.exp(-(x - mi)**2 / (2 * si**2))
    return y_out

# =========================
# Detectar el numero de picos 
# =========================
archivos = sorted(glob.glob("result_chain_*.txt"))

if len(archivos) == 0:
    print("ERROR: No se encontraron archivos result_chain_*.txt")
    print("Asegúrate de correr primero bayes_parallel.py")
    sys.exit(1)

print(f"Cadenas encontradas: {archivos}")

cadenas = [np.loadtxt(f) for f in archivos]
n_chains = len(cadenas)
n_params = len(cadenas[0])
K = n_params // 3
print(f" Detección automática exitosa: La señal contiene K = {K} picos a reconstruir.\n")

param_names = (
    [f"A_{i+1}"     for i in range(K)] +
    [f"mu_{i+1}"    for i in range(K)] +
    [f"sigma_{i+1}" for i in range(K)]
)

# Analizar el ruido de cada cadena
rms_cadenas = []
for c in cadenas:
    A_c, mu_c, sigma_c = c[0:K], c[K:2*K], c[2*K:3*K]
    y_estimado_c = gaussian_mixture(x_data, A_c, mu_c, sigma_c)
    rms_c = np.sqrt(np.mean((y_data - y_estimado_c) ** 2))
    rms_cadenas.append(rms_c)

# Promediar el resultado final de todas las cadenas paralelas ejecutadas
combined = np.mean(cadenas, axis=0)
A_comb, mu_comb, sigma_comb = combined[0:K], combined[K:2*K], combined[2*K:3*K]
y_estimado_comb = gaussian_mixture(x_data, A_comb, mu_comb, sigma_comb)
rms_combinado = np.sqrt(np.mean((y_data - y_estimado_comb) ** 2))

# Imprimir el reporte de parámetros
print("=" * 45)
print("       RESULTADOS ESTIMADOS POR MCMC")
print("=" * 45)
print(f"{'Parámetro':<15} {'Valor Estimado (Media)':>20}")
print("-" * 45)
for i, name in enumerate(param_names):
    est = combined[i]
    print(f"{name:<15} {est:>20.4f}")
print("=" * 45)
print("\n" + "=" * 45)
print("          CALIDAD DEL AJUSTE (RMS)")
print("=" * 45)
for j, rms_val in enumerate(rms_cadenas):
    print(f"Cadena {j+1} ({os.path.basename(archivos[j])}): RMS = {rms_val:.5f}")
print("-" * 45)
print(f"PROMEDIO COMBINADO MCMC: RMS = {rms_combinado:.5f}")
print("=" * 45)

# =========================
# graficas
# =========================
n_tot_params = len(param_names)
n_cols_plot = 3
n_rows_plot = int(np.ceil(n_tot_params / n_cols_plot))

fig = plt.figure(figsize=(16, 3 * n_rows_plot))
fig.suptitle(f"Comparativa de Parámetros Estimados entre Cadenas MCMC (K={K})", fontsize=15, fontweight='bold', y=1.02)

gs = gridspec.GridSpec(n_rows_plot, n_cols_plot, figure=fig, hspace=0.6, wspace=0.4)
colores = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']

for i, name in enumerate(param_names):
    row = i // n_cols_plot
    col = i % n_cols_plot
    ax = fig.add_subplot(gs[row, col])

    valores_cadenas = [c[i] for c in cadenas]
    estimado = combined[i]
    x_pos = np.arange(n_chains)
    bars = ax.bar(x_pos, valores_cadenas,
                   color=[colores[j % len(colores)] for j in range(n_chains)],
                   alpha=0.8, edgecolor='white', linewidth=0.8)

    # Línea del promedio combinado (Consenso MCMC)
    ax.axhline(estimado, color='red', linestyle='-', linewidth=1.5,
               label=f'Media: {estimado:.3f}')

    ax.set_title(name, fontweight='bold', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'C{j+1}\n(RMS:{rms_cadenas[j]:.3f})' for j in range(n_chains)], fontsize=8)
    ax.legend(fontsize=8, loc='best')
    ax.grid(axis='y', alpha=0.3)

plt.savefig("analisis_cadenas.png", dpi=150, bbox_inches='tight')
print("Gráfica guardada en: analisis_cadenas.png")
#plt.close()

# =========================
# senal recostruida
# =========================
try:
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_data, y_data, color='lightgray', linewidth=0.8, label='Experimental Data (with noise)')
    ax.plot(x_data, y_estimado_comb, color='blue', linewidth=2.5, label=f'Gaussian Fit MCMC (K={K} peaks, RMS={rms_combinado:.4f})')

    ax.set_xlabel(str(col_x), fontsize=12)
    ax.set_ylabel(str(col_y), fontsize=12)
    ax.set_title('Astronomical Signal Fit by Montecarlo Chains (MCMC)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.savefig("senal_reconstruida.png", dpi=150, bbox_inches='tight')
    print("Gráfica guardada en: senal_reconstruida.png")
    #plt.show()

except Exception as e:
    print(f" No se pudo reconstruir la señal debido al siguiente error de formato: {e}")
plt.show()