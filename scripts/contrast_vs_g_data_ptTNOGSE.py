#NMRSI - Ignacio Lembo Ferrari - 08/08/2024

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from protocols import nogse
import os
sns.set_theme(context='paper')
sns.set_style("whitegrid")

n = 2
roi = 'ROI1'
slic = 0
exp = 1
file_name = "levaduras_20240613"
folder = "contrast_vs_g_ptTNOGSE"
A0 = "sin_A0"

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/{A0}"
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

#print("T nogse = ", T_nogse, "ms")
for i, color in zip([[17.5, n],[21.5, n],[25.0, n],[27.5, n],[30.0, n],[35.0, n],[40.0, n]], palette):    

    data = np.loadtxt(f"../results_{file_name}/contrast_vs_g_data/{A0}/tnogse={i[0]}_N={n}_exp={exp}/{roi}_data_contrast_vs_g_tnogse={i[0]}_N={n}.txt")
    
    g = data[:, 0]
    f = data[:, 1] #/ data[:, 0]

    nogse.plot_contrast_vs_g_ptTNOGSE(ax, roi, g, f, i[0], i[1], slic, color)

fig.tight_layout()
fig.savefig(f"{directory}/{roi}_contrast_vs_g_N={n}_ptTNOGSE.png", dpi = 600)
fig.savefig(f"{directory}/{roi}_contrast_vs_g_N={n}_ptTNOGSE.pdf")
plt.close(fig)