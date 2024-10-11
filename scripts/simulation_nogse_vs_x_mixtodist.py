#NMRSI - Ignacio Lembo Ferrari - 07/10/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_simulation"
folder = "simulation_nogse_vs_x_mixtodist"
slic = 0 # slice que quiero ver
exp = 1 
roi = "ROI1"

D0_ext = 2.3e-12
D0_int = 0.7e-12 

M0 = 1
lc_mode = 1.75
sigma = 0.2
alpha = 1.0

tnogse = 21.5
gs = [75.0,160.0,300.0,700.0]
n = 2
x = np.linspace(0, tnogse/n, 21)

palette = [
    "#1f77b4",  # Azul
    "#9467bd",  # Púrpura
    #"#e377c2",  # Rosa
    #"#7f7f7f",  # Gris
    "#8c564b",  # Marrón
    #"#f1c40f",  # Amarillo
    "#d62728",  # Rojo
]

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/slice={slic}/tnogse={tnogse}_N={int(n)}_lcmode={lc_mode}_sigma={sigma}_D0={D0_int}_alpha={alpha}"
os.makedirs(directory, exist_ok=True)

fig, ax = plt.subplots(figsize=(8,6)) 

for g,color in zip(gs,palette):

    f = nogse.fit_nogse_vs_x_mixtodistmode(tnogse, g, n, x, lc_mode, sigma, alpha, M0, D0_int)
    error = 0

    nogse.plot_nogse_vs_x_data_ptG(ax, roi, x, f, tnogse, g, n, slic, color)

    table = np.vstack((x, f))
    np.savetxt(f"{directory}/{roi}_data_nogse_vs_x_tnogse={tnogse}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

    with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
        print(roi,  " - lc_mode = ", lc_mode, file=a)
        print("    ",  " - sigma = ", sigma, file=a)
        print("    ",  " - M0 = ", M0, file=a)
        print("    ",  " - D0 = ", D0_int, file=a)

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x.pdf")
fig.savefig(f"{directory}/nogse_vs_x.png", dpi=600)
plt.close(fig)