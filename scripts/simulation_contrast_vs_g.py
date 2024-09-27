#NMRSI - Ignacio Lembo Ferrari - 23/09/2024

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from protocols import nogse
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_simulation"
folder = "contrast_vs_g"
modelo = "Mixto+Mixto"

gamma = 267.5221900  # 1/ms.mT
slic = 0

# Crear directorio si no existe
directory = f"../results_{file_name}/{folder}"
os.makedirs(directory, exist_ok=True)

#Cálculo de las longitudes características}
tnogse = 21.5
g_contrast = np.linspace(0.1, 1300, 1000)
n = 2

tc_1 = [0.7,] 
alpha_1 = 0
D0_1 = 2.3e-12  # m2/ms
M0_1 = [0.6,0.7,0.8,0.9,1.0]

tc_2 = 2.1
alpha_2s = 0.0
D0_2 = 0.7e-12  # m2/ms
M0_2s = 0.4

fig, ax = plt.subplots(figsize=(8, 6))

palette = sns.color_palette("tab20", len(M0_1)) 

for alpha_2, M0_2, color in zip(alpha_2s, M0_2s, palette):
    #f = nogse.fit_contrast_vs_g_mixto_mixto(tnogse, g_contrast, n, tc_1, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2)
    f = nogse.fit_contrast_vs_g_mixto(tnogse, g_contrast, n, tc_1, alpha_1, M0_1, D0_1)
    nogse.plot_contrast_vs_g_data(ax, M0_2, g_contrast, f, 0, tnogse, n, slic, color)

ax.legend(title='$M_0$', title_fontsize=15, fontsize=15, loc='upper right')
ax.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=27)
ax.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax.tick_params(axis='x', rotation=0, labelsize=16, color='black')
ax.tick_params(axis='y', labelsize=16, color='black')
title = ax.set_title(f" Extracelular | $D0_1$ = {D0_1} | $\\tau_c1$ = {tc_1}", fontsize=15)

fig.savefig(f"{directory}/contrast_vs_g.png", dpi=600)
fig.savefig(f"{directory}/contrast_vs_g.pdf")
plt.close(fig)









    # ld = np.sqrt(D0_1 * tnogse)
    # lc = np.sqrt(D0_1 * tc_1)
    # lg = (D0_1 / (gamma * g_contrast)) ** (1/3)  # m
    # L_d = ld/lg
    # L_c = lc/lg
    # L_c_f = ((3/2)**(1/4))*(L_d**(-1/2))
    # l_c_f = L_c_f*lg 
    #nogse.plot_contrast_vs_g_ptTNOGSE(ax, "ROI1", modelo, g_contrast, f, tnogse, n, slic, color)
    #nogse.plot_contrast_vs_g_ptTNOGSE(ax1, "ROI1", modelo, L_d, f, tnogse, n, slic, color)
    #nogse.plot_contrast_vs_g_ptTNOGSE(ax2, "ROI1", modelo, L_c, f, tnogse, n, slic, color)

#l_d = np.sqrt(2*D0_1*40.0)
#l_G = ((2**(3/2))*D0_1/(gamma*300.0))**(1/3)
#L_d = l_d/l_G
# print(nogse.M_nogse_free(tnogse, 10, n, tnogse/n, M0_1, D0_1))
# print(nogse.M_nogse_rest(tnogse, 10, n, tnogse/n, tc_1, M0_1, D0_1))
# print(nogse.M_nogse_mixto(tnogse, 10, n, tnogse/n, tc_1, alpha_1, M0_1, D0_1))


# ax.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=15, fontsize=15, loc='upper right')
# ax.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=27)
# ax.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
# ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
# ax.tick_params(axis='x', rotation=0, labelsize=16, color='black')
# ax.tick_params(axis='y', labelsize=16, color='black')
# title = ax.set_title(f"$N$ = {n} | $D0_1$ = {D0_1} | $\\alpha_1$ = {alpha_1} | $\\tau_c1$ = {tc_1} | $D0_2$ = {D0_2} | $\\alpha_2$ = {alpha_2} | $\\tau_c2$ = {tc_2}", fontsize=15)
# fig.savefig(f"{directory}/contrast_vs_g.png", dpi=600)
# fig.savefig(f"{directory}/contrast_vs_g.pdf")
# #plt.show()
# plt.close(fig)

# ax1.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=15, fontsize=15, loc='upper right')
# ax1.set_xlabel("Longitud de difusión $L_d$", fontsize=27)
# ax1.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
# ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
# ax1.tick_params(axis='x', rotation=0, labelsize=16, color='black')
# ax1.tick_params(axis='y', labelsize=16, color='black')
# title = ax1.set_title(f"$N$ = {n} | $D0_1$ = {D0_1} | $\\alpha_1$ = {alpha_1} | $\\tau_c1$ = {tc_1} | $D0_2$ = {D0_2} | $\\alpha_2$ = {alpha_2} | $\\tau_c2$ = {tc_2}", fontsize=15)
# ax1.set_xlim(0, 3.0)
# fig1.savefig(f"{directory}/contrast_vs_Ld.png", dpi=600)
# fig1.savefig(f"{directory}/contrast_vs_Ld.pdf")
# #plt.show()
# plt.close(fig1)

# ax2.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=15, fontsize=15, loc='upper right')
# ax2.set_xlabel("Longitud de correlación $L_c$", fontsize=27)
# ax2.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
# ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
# ax2.tick_params(axis='x', rotation=0, labelsize=16, color='black')
# ax2.tick_params(axis='y', labelsize=16, color='black')
# title = ax2.set_title(f"$N$ = {n} | $D0_1$ = {D0_1} | $\\alpha_1$ = {alpha_1} | $\\tau_c1$ = {tc_1} | $D0_2$ = {D0_2} | $\\alpha_2$ = {alpha_2} | $\\tau_c2$ = {tc_2}", fontsize=15)
# #ax2.set_xlim(0)
# fig2.savefig(f"{directory}/contrast_vs_Lc.png", dpi=600)
# fig2.savefig(f"{directory}/contrast_vs_Lc.pdf")
# #plt.show()
# plt.close(fig2)