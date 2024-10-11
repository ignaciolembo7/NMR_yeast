#NMRSI - Ignacio Lembo Ferrari - 30/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_simulation"
folder = "simulation_nogse_vs_x_free"
slic = 0 # slice que quiero ver
exp = 1 
roi = "ROI1"

D0_ext = 2.3e-12
D0_int = 0.7e-12 

D0 = D0_ext
M0 = 1
alpha = 0.57

# tnogse = 15.0
# gs = [100.0,275.0,600.0,1000.0]
# tnogse = 17.5
# gs = [105.0, 210.0, 405.0, 800.0]
# tnogse = 21.5 
# gs = [75.0,160.0,300.0,700.0]
# tnogse = 25.0
# gs = [60.0,120.0,210.0,600.0]
# tnogse = 27.5
# gs = [55.0, 110.0, 190.0, 550.0]
tnogse = 30.0
gs = [50.0,100.0,170.0,500.0]
# tnogse = 32.5
# gs = [45.0,90.0,150.0,450.0]
# tnogse = 35.0
# gs = [40.0,80.0,130.0,400.0]
# tnogse = 37.5
# gs = [35.0,75.0,120.0,375.0]
# tnogse = 40.0
# gs = [30.0,70.0,110.0,350.0]

n = 2
x = np.linspace(0.5, tnogse/n, 21)

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
directory = f"../results_{file_name}/{folder}/slice={slic}/tnogse={tnogse}_N={int(n)}_alpha={alpha}_D0={D0}"
os.makedirs(directory, exist_ok=True)

fig, ax = plt.subplots(figsize=(8,6)) 

for g,color in zip(gs,palette):

    f = nogse.M_nogse_free(tnogse, g, n, x, M0, alpha*D0)
    error = 0

    nogse.plot_nogse_vs_x_data_ptG(ax, roi, x, f, tnogse, g, n, slic, color)

    table = np.vstack((x, f))
    np.savetxt(f"{directory}/{roi}_data_nogse_vs_x_tnogse={tnogse}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

    with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
        print("    ",  " - alpha = ", alpha, file=a)
        print("    ",  " - M0 = ", M0, file=a)
        print("    ",  " - D0 = ", D0, file=a)

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x.pdf")
fig.savefig(f"{directory}/nogse_vs_x.png", dpi=600)
plt.close(fig)