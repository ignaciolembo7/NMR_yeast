#NMRSI - Ignacio Lembo Ferrari - 21/08/2024

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from protocols import nogse
import os
sns.set_theme(context='paper')
sns.set_style("whitegrid")

T_nogse = 21.5
n = 2
roi = 'ROI1'
slic = 0
exp = 1
file_name = "levaduras_20240622"
folder = "rest_vs_x_restdist_mode_ptG"
# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/"
os.makedirs(directory, exist_ok=True)

#palette = sns.color_palette("tab20", 4) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

palette = [
    "#1f77b4",  # Azul
    "#ff7f0e",  # Naranja
    "#f1c40f",  # Amarillo
    "#2ca02c",  # Verde
]

# Asignar la paleta personalizada
sns.set_palette(palette)

fig, ax = plt.subplots(figsize=(8,6))
fig1, ax1 = plt.subplots(figsize=(8,6))


#print("T nogse = ", T_nogse, "ms")

for i, color in zip([[T_nogse, 75.0, n, 1],[T_nogse, 160.0, n, 2],[T_nogse, 300.0, n, 3],[T_nogse, 700.0, n, 4]], palette):

    data1 = np.loadtxt(f"../results_{file_name}/nogse_vs_x_data/slice={slic}/tnogse={T_nogse}_g={i[1]}_N={n}_exp={exp}/{roi}_data_nogse_vs_x_tnogse={T_nogse}_g={i[1]}_N={n}.txt")
    x = data1[:, 0]
    f = data1[:, 1] #/data1[0,1]

    data2 = np.loadtxt(f"../results_{file_name}/nogse_vs_x_restdist_mode/tnogse={T_nogse}_g={i[1]}_N={n}_exp={exp}/{roi}_ajuste_nogse_vs_x_tnogse={T_nogse}_g={i[1]}_N={n}_exp={exp}.txt")
    x_fit = data2[:, 0]
    f_fit = data2[:, 1] #/data1[0,1]

    data3 = np.loadtxt(f"../results_{file_name}/nogse_vs_x_restdist_mode/tnogse={T_nogse}_g={i[1]}_N={n}_exp={exp}/{roi}_dist_tnogse={T_nogse}_g={i[1]}_N={n}_exp={exp}.txt")
    l_c = data3[:, 0]
    dist = data3[:, 1]

    nogse.plot_nogse_vs_x_restdist_ptTNOGSE(ax, roi, x, x_fit, f, f_fit, i[0], i[1], i[2], slic, color, label = f" G{i[3]} = {i[1]}") 
    dist = dist/np.max(dist)
    nogse.plot_lognorm_dist_ptG(ax1, roi, i[0], i[1], i[2], l_c, dist, slic, color, label = f" G{i[3]} = {i[1]}")

fig.tight_layout()
fig.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={T_nogse}_N={n}_ptG.png", dpi = 600)
fig.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={T_nogse}_N={n}_ptG.pdf")
plt.close(fig)

fig1.tight_layout()
fig1.savefig(f"{directory}/{roi}_dist_vs_lc_tnogse={T_nogse}_N={n}_ptG.png", dpi = 600)
fig1.savefig(f"{directory}/{roi}_dist_vs_lc_tnogse={T_nogse}_N={n}_ptG.pdf")
plt.close(fig1)