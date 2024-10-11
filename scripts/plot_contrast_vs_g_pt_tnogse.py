#NMRSI - Ignacio Lembo Ferrari - 27/09/2024

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from protocols import nogse
import os
sns.set_theme(context='paper')
sns.set_style("whitegrid")

roi = 'ROI1'
slic = 0
exp = 1
file_name = "levaduras_20240613"
folder = "fit_contrast_vs_g_free_restdist"
A0 = "con_A0"
modelo = "Free+RestDist"
# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}"
os.makedirs(directory, exist_ok=True)

#palette = sns.color_palette("tab10", 10) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)
# Definir manualmente una paleta de colores personalizada (sin verde ni naranja ya que los reservo para indicar las zonas intra y extracelular respectivamente).
palette = [
    #"#aec7e8"   # Azul claro
    "#1f77b4",  # Azul
    "#9467bd",  # Púrpura
    "#e377c2",  # Rosa
    "#7f7f7f",  # Gris
    "#8c564b",  # Marrón
    "#f1c40f",  # Amarillo oscuro
    "#d62728",  # Rojo
]
sns.set_palette(palette)

fig, ax = plt.subplots(figsize=(8,6))

tnogses = [17.5, 21.5, 25.0, 27.5, 30.0, 35.0, 40.0]
n=2

for tnogse, color in zip(tnogses, palette):    

    data = np.loadtxt(f"../results_{file_name}/{folder}/tnogse={tnogse}_N={n}_exp={exp}/{roi}_fit_contrast_vs_g_tnogse={tnogse}_N={n}_exp={exp}.txt")
    
    g = data[:, 0]
    f = data[:, 1] #/ data[:, 0]

    nogse.plot_contrast_vs_g_ptTNOGSE(ax, roi, modelo, g, f, tnogse, n, slic, color)

fig.tight_layout()
fig.savefig(f"{directory}/{roi}_contrast_vs_g_N={n}_ptTNOGSE.png", dpi = 600)
fig.savefig(f"{directory}/{roi}_contrast_vs_g_N={n}_ptTNOGSE.pdf")
plt.close(fig)