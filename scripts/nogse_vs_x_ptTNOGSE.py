#NMRSI - Ignacio Lembo Ferrari - 29/07/2024

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
num_grad = 'G2'
file_name = "levaduras_20240622"
folder = "nogse_vs_x_data"
# Create directory if it doesn't exist
directory = f"../results_{file_name}/nogse_vs_x_ptTNOGSE/"
os.makedirs(directory, exist_ok=True)

palette = sns.color_palette("tab10", 10) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

fig, ax = plt.subplots(figsize=(8,6))

#print("T nogse = ", T_nogse, "ms")
for i, color in zip([[17.5, 210.0, n], [21.5, 160.0, n],[25.0, 120.0, n],[30.0, 100.0, n],[35.0, 80.0, n],[40.0, 70.0, n]], palette):    
    data = np.loadtxt(f"../results_{file_name}/{folder}/slice={slic}/tnogse={i[0]}_g={i[1]}_N={n}_exp={exp}/{roi}_data_nogse_vs_x_tnogse={i[0]}_g={i[1]}_N={n}.txt")
    x = data[:, 0]
    f = data[:, 1]/data[0,1]

    #print("G = ", i[1], " - Contraste = ", f[-1] - f[0])

    nogse.plot_nogse_vs_x_data_ptTNOGSE(ax, roi, x, f, i[0], i[2], slic, color)

fig.tight_layout()
fig.savefig(f"{directory}/{roi}_nogse_vs_x_{num_grad}_N={n}_ptTNOGSE.png", dpi = 600)
fig.savefig(f"{directory}/{roi}_nogse_vs_x_{num_grad}_N={n}_ptTNOGSE.pdf")
plt.close(fig)

# G=1 for i, color in zip([[15.0, 100.0, n],[17.5, 105.0, n],[21.5, 75.0, n],[25.0, 60.0, n],[27.5, 55.0, n],[30.0, 50.0, n],[32.5, 45.0, n],[35.0, 40.0, n],[37.5, 35.0, n],[40.0, 30.0, n]], palette):    
# extracelular for i, color in zip([[15.0, 275.0, n],[17.5, 210.0, n],[21.5, 160.0, n],[25.0, 120.0, n],[27.5, 110.0, n],[30.0, 100.0, n],[32.5, 90.0, n],[35.0, 80.0, n],[37.5, 75.0, n],[40.0, 70.0, n]], palette):
# G=3 for i, color in zip([[15.0, 600.0, n],[17.5, 405.0, n],[21.5, 300.0, n],[25.0, 210.0, n],[27.5, 190.0, n],[30.0, 170.0, n],[32.5, 150.0, n],[35.0, 130.0, n],[37.5, 120.0, n],[40.0, 110.0, n]], palette):    
# intracelular for i, color in zip([[15.0, 1000.0, n],[17.5, 800.0, n],[21.5, 700.0, n],[25.0, 600.0, n],[27.5, 550.0, n],[30.0, 500.0, n],[32.5, 450.0, n],[35.0, 400.0, n],[37.5, 375.0, n],[40.0, 350.0, n]], palette):
