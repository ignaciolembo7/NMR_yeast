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
folder = "contrast_vs_advar"

# Parámetros de simulación
tnogses = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
n = 2
alpha = 0.03  # 0  = rest, 1 = libre
tc = 10.0 #no pasarse de 20
D0 = 2.3e-12  # m2/ms
gamma = 267.5221900  # 1/ms.mT

# Crear directorio si no existe
directory = f"../results_{file_name}/{folder}/N={int(n)}_D0_={D0}_alpha={alpha}"
os.makedirs(directory, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 6))
fig1, ax1 = plt.subplots(figsize=(8, 6))

palette = sns.color_palette("tab10", len(tnogses)) 

for tnogse, color in zip(tnogses, palette):

    g_contrast = np.linspace(0.1, 1300, 1000)
    ld = np.sqrt(D0 * tnogse)
    lc = np.sqrt(D0 * tc)
    lg = (D0 / (gamma * g_contrast)) ** (1/3)  # m
    Ld = ld/lg
    Lc = lc/lg
    Lcf = ((3/2)**(1/4))*(Ld**(-1/2))
    lcf = Lcf*lg 
 
    contrast = nogse.delta_M_ad(Ld, Lc, n, alpha, D0)
    #ax.fill_between(Lc, 0, contrast, color=color, alpha=0.5)
    ax.plot(Ld, contrast, color=color, linewidth=2, label= f"{tnogse}")
    ax1.plot(Lcf, contrast, color=color, linewidth=2, label= f"{tnogse}")

ax.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=15, fontsize=15, loc='upper right')
ax.set_xlabel("Longitud de centro de filtro $L_c^f$", fontsize=27)
ax.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
#ax.set_yscale('log')
ax.tick_params(direction='out', top=False, right=False, left=True, bottom=True)
ax.tick_params(axis='x', rotation=0, labelsize=16, color='black')
ax.tick_params(axis='y', labelsize=16, color='black')
#ax.set_xlim([0,1.5])
title = ax.set_title(f"$N$ = {n} | $D_0$ = {D0} | $\\alpha$ = {alpha}", fontsize=15)
fig.savefig(f"{directory}/contrast_vs_Lcf.png", dpi=600)
fig.savefig(f"{directory}/contrast_vs_Lcf.pdf")

ax1.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=15, fontsize=15, loc='upper right')
ax1.set_xlabel("Longitud de difusión $L_d$", fontsize=27)
ax1.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
#ax.set_yscale('log')
ax1.tick_params(direction='out', top=False, right=False, left=True, bottom=True)
ax1.tick_params(axis='x', rotation=0, labelsize=16, color='black')
ax1.tick_params(axis='y', labelsize=16, color='black')
#ax1.set_xlim([0,3.5])
title = ax1.set_title(f"$N$ = {n} | $D_0$ = {D0} | $\\alpha$ = {alpha}", fontsize=15)
fig1.savefig(f"{directory}/contrast_vs_Ld.png", dpi=600)
fig1.savefig(f"{directory}/contrast_vs_Ld.pdf")

plt.show()
plt.close(fig)
plt.close(fig1)