#NMRSI - Ignacio Lembo Ferrari - 28/05/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_20240613"
folder = "plot_contrast_vs_g_fittings"
A0 = "con_A0"
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # 0.7e-12 # intra
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "Mixto+Mixto"

tnogse = float(input('Tnogse [ms]: ')) #ms
n = 2
roi = "ROI1"

#palette = sns.color_palette("tab10", len(tcs)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/tnogse={tnogse}_N={int(n)}_exp={exp}"
os.makedirs(directory, exist_ok=True)

fig, ax = plt.subplots(figsize=(8,6)) 

data = np.loadtxt(f"../results_{file_name}/contrast_vs_g_data/{A0}/tnogse={tnogse}_N={int(n)}_exp={exp}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={int(n)}.txt")
g = data[:, 0]
f = data[:, 1]
error = data[:,2]
nogse.plot_contrast_vs_g_data(ax, roi, g, f, error, tnogse, g, n, slic)

#for tc, alpha, M0, color in zip(tcs, alphas, M0s, palette):

tc_1 = 5.1
alpha_1 = 0.3
D0_1 = D0_ext
M0_1 = 0.8
tc_2 = 2.12
alpha_2 = 0.03
D0_2 = D0_int
M0_2 = 0.45

g_fit = np.linspace(np.min(g), np.max(g), num=1000)

fit = nogse.fit_contrast_vs_g_mixto_mixto(tnogse, g_fit, n, tc_1, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2)
nogse.plot_contrast_vs_g_fit(ax, roi, modelo, g, g_fit, f, fit, tnogse, n, slic, color="blue")
fit_ext = nogse.fit_contrast_vs_g_mixto(tnogse, g, n, tc_1, alpha_1, M0_1, D0_1)
nogse.plot_contrast_vs_g_fit(ax, "Extracelular", modelo, g, g, f, fit_ext, tnogse, n, slic, color="orange")
fit_int = nogse.fit_contrast_vs_g_mixto(tnogse, g, n, tc_2, alpha_2, M0_2, D0_2)
nogse.plot_contrast_vs_g_fit(ax, "Intracelular", modelo, g, g, f, fit_int, tnogse, n, slic, color="green")

title = ax.set_title(f"{modelo} | $\\tau_e$ = {tc_1} ms | $\\tau_i$ = {tc_2} ms | $\\alpha_e$ = {alpha_1} ms | $\\alpha_i$ = {alpha_2} ms | $D0_e$ = {D0_1} | $D0_i$ = {D0_2}", fontsize=12)

fig.tight_layout()
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}.pdf")
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}.png", dpi=600)
plt.close(fig)