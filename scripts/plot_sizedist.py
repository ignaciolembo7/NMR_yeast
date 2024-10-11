#NMRSI - Ignacio Lembo Ferrari - 05/10/2024

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from protocols import nogse
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_20240622"
folder = "plot_sizedist"
slic = 0 # slice que quiero ver
D0_ext = 2.3e-12 # m2/ms extra
D0_int = 0.7e-12 # intra
zone = "int"
n = 2
roi = "ROI1"

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}"
os.makedirs(directory, exist_ok=True)

palette = sns.color_palette("tab10") # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)
# palette = [
#     "#1f77b4",  # Azul
#     #"#00008B",  # Azul oscuro
#     #"#17becf",  # Cyan
#     "#9467bd",  # Púrpura
#     "#e377c2",  # Rosa
#     "#7f7f7f",  # Gris
#     #"#8c564b",  # Marrón
#     #"#f1c40f",  # Amarillo
#     "#f1c40f",  # Amarillo oscuro
#     "#d62728",  # Rojo
# ]
# sns.set_palette(palette)

fig, ax = plt.subplots(figsize=(8,6)) 
fig1, ax1 = plt.subplots(figsize=(8,6))

D0s = [0.65,0.7,0.8,0.9,1.0]

for D0, color in zip(D0s, palette):
    print(D0)

    data = np.loadtxt(f"{directory}/globalfit_lc_sigma_D0={D0}_nogse_vs_x_restdist_mode/{roi}_dist_N={n}_exp=1.txt")
    parameters = np.loadtxt(f"{directory}/globalfit_lc_sigma_D0={D0}_nogse_vs_x_restdist_mode/{roi}_parameters_vs_tnogse.txt")
    lc_mode = parameters[1]
    lc_mode_error = parameters[2]
    sigma = parameters[7]
    sigma_error = parameters[8]

    lc = data[:, 0]
    dist = data[:, 1] 

    ax.plot(lc, dist, '-', linewidth=1, color = color, label=f"{D0}")
    ax.fill_between(lc, dist, color=color, alpha=0.3)
    ax.set_xlabel("Longitud de correlación $l_c$ [$\mu$m]", fontsize=27)
    ax.set_ylabel("P($l_c$)", fontsize=27)
    ax.legend(title='$D_0$ [$m^2$/ms]', title_fontsize=15, fontsize=15, loc='upper right')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title(f" $N$ = {n} | slice = {slic} ", fontsize=15)
    ax.set_xlim(0.5, 6)

    diameter = np.linspace(0.1, 14, 1000) 
    diameter_dist = nogse.lognormal(diameter, sigma, lc_mode/0.3)
    ax1.plot(diameter, diameter_dist, '-', linewidth=1, color = color, label=f"{D0}")
    ax1.fill_between(diameter, diameter_dist, color=color, alpha=0.3)
    ax1.set_xlabel("Diámetro de célula $d$ [$\mu$m]", fontsize=27)
    ax1.set_ylabel("P($d$)", fontsize=27)
    ax1.legend(title='$D_0$ [$m^2$/ms]', title_fontsize=15, fontsize=15, loc='upper right')
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax1.tick_params(axis='y', labelsize=16, color='black')
    title = ax1.set_title(f" $N$ = {n} | slice = {slic} ", fontsize=15)
    ax1.set_xlim(0.5, 14)


fig.tight_layout()
fig.savefig(f"{directory}/{roi}_lcdist_{zone}_ptD0.png", dpi=600)
fig.savefig(f"{directory}/{roi}_lcdist_{zone}_ptD0.pdf")

fig1.tight_layout()
fig1.savefig(f"{directory}/{roi}_diameterdist_{zone}_ptD0.png", dpi=600)
fig1.savefig(f"{directory}/{roi}_diameterdist_{zone}_ptD0.pdf")
