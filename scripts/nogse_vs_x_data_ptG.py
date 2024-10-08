#NMRSI - Ignacio Lembo Ferrari - 29/07/2024

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from protocols import nogse
import os
sns.set_theme(context='paper')
sns.set_style("whitegrid")

T_nogse = 37.5
n = 2
roi = 'ROI1'
slic = 0
exp = 1
file_name = "levaduras_generated"
folder = "nogse_vs_x_data_ptG"
# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/"
os.makedirs(directory, exist_ok=True)

palette = sns.color_palette("tab20", 4) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

fig, ax = plt.subplots(figsize=(8,6))

#print("T nogse = ", T_nogse, "ms")

for i, color in zip([[T_nogse, 35.0, n],[T_nogse, 75.0, n],[T_nogse, 120.0, n],[T_nogse, 375.0, n]], palette):
    data = np.loadtxt(f"../results_{file_name}/{folder}/slice={slic}/tnogse={T_nogse}_g={i[1]}_N={n}_exp={exp}/{roi}_data_nogse_vs_x_tnogse={T_nogse}_g={i[1]}_N={n}.txt")
    x = data[:, 0]
    f = data[:, 1]/data[0,1]

    #print("G = ", i[1], " - Contraste = ", f[-1] - f[0])

    nogse.plot_nogse_vs_x_data_ptG(ax, roi, x, f, i[0], i[1], i[2], slic, color)

fig.tight_layout()
fig.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={T_nogse}_N={n}_ptG.png", dpi = 600)
fig.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={T_nogse}_N={n}_ptG.pdf")
plt.close(fig)