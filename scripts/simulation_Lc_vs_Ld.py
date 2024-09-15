#NMRSI - Ignacio Lembo Ferrari - 09/09/2024

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from protocols import nogse

# Establecer tema de seaborn para las gráficas
sns.set_theme(context='paper')
sns.set_style("whitegrid")

# Parámetros de simulación
file_name = "simulations"
folder = "Lc_vs_Ld"

# Parámetros de simulación
n = 2
alpha = 0.0  # 1 (libre) = inf (rest) = 1/alpha
D0 = 0.7e-12  # m2/ms
gamma = 267.5221900  # 1/ms.mT

# Crear directorio si no existe
directory = f"../results_{file_name}/{folder}/N={int(n)}_D0_={D0}_alpha={alpha}"
os.makedirs(directory, exist_ok=True)

# Inicialización de los parámetros de contraste
Ld = np.linspace(0.2, 3.5, 1000)
Lc = np.linspace(0.2, 10.0, 1000)

# Cálculo de las longitudes características
#g_contrast = np.linspace(0.1, 1300, 2000)
#tnogse = 21.5
#tau_c = 20.0 #no pasarse de 20
#ld = np.sqrt(D0 * tnogse)
#lc = np.sqrt(D0 * tau_c)
#lg = (D0 / (gamma * g_contrast)) ** (1/3)  # m
#Ld = ld/lg
#Lc = lc/lg

# Crear una grilla de valores para Ld y Lc usando meshgrid
#Lc_grid, Ld_grid = np.meshgrid(lc / lg, ld / lg)
Ld_grid, Lc_grid = np.meshgrid(Ld,Lc)

contrast = nogse.delta_M_ad(Ld_grid, Lc_grid, n, alpha, D0)

# Crear la figura para el mapa de colores continuo
fig, ax = plt.subplots(figsize=(8, 6))

# Generar el mapa de colores continuo con pcolormesh
#c = ax.pcolormesh(ld / lg, lc / lg, contrast, cmap='inferno')
c = ax.pcolormesh(Ld_grid, Lc_grid, contrast, cmap='inferno')
# Añadir barra de color
cbar = fig.colorbar(c, ax=ax)
cbar.set_label('Contraste $\Delta M_\mathrm{NOGSE}$', fontsize=27)
# Añadir curvas de nivel 
#ax.contour(ld / lg, lc / lg, contrast, levels=10, colors='white', linewidths=0.5)
ax.contour(Ld_grid, Lc_grid, contrast, levels=10, colors='white', linewidths=0.5)

# Añadir rectas de pendiente tnogse/tc
#tc_recta = [3.3,3.56,5.09,6.54,5.21,6.24,6.48,7.5]
tc_recta = [2.2,2.12,1.71,1.61,2.04,1.38,1.59,2.0]
tnogse_recta = [15.0, 17.5, 21.5, 25.0, 27.5, 30.0, 35.0, 40.0]
#tnogse_recta = [1.0, 2.0, 2.5, 3.0, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0, 100.0, 200.0, 300.0, 600.0] 
pendientes = [tnogse_recta[i]/tc_recta[i] for i in range(len(tnogse_recta))]

palette = sns.color_palette("tab20", len(pendientes)) 
for pendiente, color in zip(pendientes, palette):
    Lc_recta = np.sqrt(1/pendiente)*Ld
    ax.plot(Ld, Lc_recta, color=color, linewidth=2, label= f"{round(pendiente, 2)}")
ax.legend(title='$T_\mathrm{NOGSE}/\\tau_c$', fontsize=15, title_fontsize=15, loc='upper right')

# Etiquetas y título del gráfico
ax.set_xlabel("Longitud de difusión $L_d$", fontsize=27)
ax.set_ylabel("Longitud de correlación $L_c$", fontsize=27)
#ax.set_yscale('log')
ax.tick_params(direction='out', top=False, right=False, left=True, bottom=True)
ax.tick_params(axis='x', rotation=0, labelsize=16, color='black')
ax.tick_params(axis='y', labelsize=16, color='black')
#ax.set_yticks([1, 2, 3])
#ax.set_yticklabels([1, 2, 3])
#ax.set_xticks([1,2,3])
#ax.set_xticklabels([1, 2, 3])
ax.set_xlim([min(Ld),max(Ld)])
ax.set_ylim([min(Lc),max(Lc)])
title = ax.set_title(f"$N$ = {n} | $\\alpha$ = {alpha} | $D_0$ = {D0} ", fontsize=15)

# #Dibujar un punto en una posicion especifica de tamaño 7
# tnogse = 21.5
# tau_c = 5.028
# gs = [75.0, 160.0, 300.0, 700.0] 
# for g in gs:
#     ld = np.sqrt(D0 * tnogse)
#     lc = np.sqrt(D0 * tau_c)
#     lg = (D0 / (gamma * g)) ** (1/3)  # m
#     Ld = ld/lg
#     Lc = lc/lg
#     ax.scatter(Ld, Lc, s=20, color='white')

# Mostrar la figura
plt.show()
# Guardar la figura en diferentes formatos
fig.savefig(f"{directory}/contrast_vs_Lc_vs_Ld.png", dpi=600)
# Cerrar la figura para liberar memoria
plt.close(fig)