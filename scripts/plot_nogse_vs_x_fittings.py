#NMRSI - Ignacio Lembo Ferrari - 28/05/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_20240622"
folder = "plot_nogse_vs_x_fittings"
A0 = "sin_A0"
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # 0.7e-12 # intra
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "Mixto"  # nombre carpeta modelo libre/rest/tort

num_grad = input('Gradiente: ')
tnogse = float(input('Tnogse [ms]: ')) #ms
g = float(input('g [mT/m]: ')) #mT/m
n = 2
roi = "ROI1"

#palette = sns.color_palette("tab10", len(tcs)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}"
os.makedirs(directory, exist_ok=True)

fig, ax = plt.subplots(figsize=(8,6)) 

data = np.loadtxt(f"../results_{file_name}/nogse_vs_x_data/slice={slic}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}/{roi}_data_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt")
x = data[:, 0]
f = data[:, 1]
error = data[:,2]
#nogse.plot_nogse_vs_x_data(ax, roi, x, f, error, tnogse, g, n, slic)

#for tc, alpha, M0, color in zip(tcs, alphas, M0s, palette):

M0 = 2680
tc = 7.0
alpha = 0.0
x_fit = np.linspace(np.min(x), np.max(x), num=1000)
f1 = nogse.M_nogse_mixto(tnogse, g, n, x_fit, tc, alpha, M0, D0_ext)
nogse.plot_nogse_vs_x_fit(ax, "restringido", modelo, x, x_fit, f, f1, tnogse, n, g, slic, color = 'green')  

M0 = 2680
alpha = 0.5
x_fit = np.linspace(np.min(x), np.max(x), num=1000)
f2 = nogse.M_nogse_free(tnogse,g,n,x_fit,M0,D0_ext*alpha)
nogse.plot_nogse_vs_x_fit(ax, "tortuoso", modelo, x, x_fit, f, f2, tnogse, n, g, slic, color = 'orange')  

M0 = 2680
tc = 4.2
alpha = 0.232
x_fit = np.linspace(np.min(x), np.max(x), num=1000)
f3 = nogse.M_nogse_mixto(tnogse, g, n, x_fit, tc, alpha, M0, D0_ext)
nogse.plot_nogse_vs_x_fit(ax, "fit", modelo, x, x_fit, f, f3, tnogse, n, g, slic, color = 'blue')  


fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
plt.close(fig)