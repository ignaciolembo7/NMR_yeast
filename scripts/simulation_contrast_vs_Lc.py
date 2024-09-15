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
folder = "contrast_vs_Lc"

# Parámetros de simulación
tnogse = 21.5
n = 2
alpha = 0.0 # 0  = rest, 1 = libre
tc = 20.0 #no pasarse de 20
D0 = 2.3e-12  # m2/ms
gamma = 267.5221900  # 1/ms.mT

# Crear directorio si no existe
directory = f"../results_{file_name}/{folder}/N={int(n)}_D0_={D0}_alpha={alpha}"
os.makedirs(directory, exist_ok=True)

# Cálculo de las longitudes características
#g_contrast = np.linspace(0.1, 1300, 2000)
#ld = np.sqrt(D0 * tnogse)
#lc = np.sqrt(D0 * tc)
#lg = (D0 / (gamma * g_contrast)) ** (1/3)  # m
#Ld = ld/lg
##Lc = lc/lg
#L_c_f = ((3/2)**(1/4))*(Ld**(-1/2))
#l_c_f = L_c_f*lg 

Lc = np.linspace(0.3, 3.5, 1000)    
Ld = [3.0,4.0,5.0,6.0,7.0]

fig, ax = plt.subplots(figsize=(8, 6))

palette = sns.color_palette("tab10", len(Ld)) 

for Ld_, color in zip(Ld, palette):
    contrast = nogse.delta_M_ad(Ld_, Lc, n, alpha, D0)
    Lcf = ((3/2)**(1/4))*(Ld_**(-1/2))
    ax.axvline(x=Lcf, color=color, linestyle='--', label=f"Lcf = {Lcf:.2f}")
    ax.fill_between(Lc, 0, contrast, color=color, alpha=0.5)
    ax.plot(Lc, contrast, color=color, linewidth=2, label= f"Ld = {Ld_}")

ax.legend(title_fontsize=15, fontsize = 15, loc='upper right')

# Etiquetas y título del gráfico
ax.set_xlabel("Longitud de difusión $L_c$", fontsize=27)
ax.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)

#ax.set_yscale('log')
ax.tick_params(direction='out', top=False, right=False, left=True, bottom=True)
ax.tick_params(axis='x', rotation=0, labelsize=16, color='black')
ax.tick_params(axis='y', labelsize=16, color='black')

#ax.set_yticks([1, 2, 3])
#ax.set_yticklabels([1, 2, 3])
#ax.set_xticks([1,2,3])
#ax.set_xticklabels([1, 2, 3])
ax.set_xlim([0,1.5])
#ax.set_ylim([min(lc/lg),max(lc/lg)])
title = ax.set_title(f"$N$ = {n} | $D_0$ = {D0} | $\\alpha$ = {alpha}", fontsize=15)

# Mostrar la figura
plt.show()
# Guardar la figura en diferentes formatos
fig.savefig(f"{directory}/contrast_vs_Lc.png", dpi=600)
fig.savefig(f"{directory}/contrast_vs_Lc.pdf")
# Cerrar la figura para liberar memoria
plt.close(fig)