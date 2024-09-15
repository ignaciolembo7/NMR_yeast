#NMRSI - Ignacio Lembo Ferrari - 28/08/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

tnogse = float(input('T_NOGSE [ms]: ')) #ms
n = float(input('N: '))
folder = "contrast_vs_g_data"
A0 = "con_A0"
slic = 0 # slice que quiero ver

file_name = "levaduras_20240613"
rois = ["ROI1"]
exps = [1]
ids = ["1 - 20 horas y 16 minutos." ]#, "1 - 22 días, 20 horas y 59 minutos"] #["1 - 15 días, 6 horas y 59 minutos", "2 - 18 días, 20 horas y 26 minutos"]#["1 - 8 días*, 5 horas y 29 minutos - 21/05", "4 - 10 días, 2 horas", "5 - 13 días, 6 horas y 51 minutos", "6 - 18 días, 22 horas y 45 minutos", "7 - 19 días, 3 horas y 29 minutos"]
palette = sns.color_palette("tab10", len(exps)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

fig, ax = plt.subplots(figsize=(8,6)) 
# Create directory if it doesn't exist
directory = f"../{file_name}/{folder}/{A0}/tnogse={tnogse}_n={int(n)}"
os.makedirs(directory, exist_ok=True)

for exp, roi, id, color in zip(exps, rois, ids, palette):

    data = np.loadtxt(f"../results_{file_name}/contrast_vs_g_data/tnogse={tnogse}_n={int(n)}_exp={exp}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={int(n)}.txt")
    g = data[:, 0]
    f = data[:, 1]
    error = data[:, 2]

    table = np.vstack((g, f, error))
    np.savetxt(f"{directory}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={n}.txt", table.T, delimiter=' ', newline='\n')

    nogse.plot_contrast_vs_g_data(ax, id, g, f, error, tnogse, n, slic, color)

fig.tight_layout()
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_n={int(n)}.pdf")
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_n={int(n)}.png", dpi=600)
plt.close(fig)