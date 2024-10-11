#NMRSI - Ignacio Lembo Ferrari - 30/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_simulation"
folder = "simulation_nogse_vs_x_mixto_mixto"
slic = 0 # slice que quiero ver
exp = 1 
roi = "ROI1"

D0_ext = 2.3e-12
D0_int = 0.7e-12 

D0_1 = D0_ext
M0_1 = 0.55
tc_1 = 5
alpha_1 = 0.15

D0_2 = D0_int
M0_2 = 1 - M0_1
tc_2 = 1.8
alpha_2 = 0.0

# tnogse = 21.5
# gs = [75.0,160.0,300.0,700.0]
tnogse=30.0
gs = [100.0] #[50.0,100.0,170.0,500.0]


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
directory = f"../results_{file_name}/{folder}/slice={slic}/tnogse={tnogse}_N={int(n)}"
os.makedirs(directory, exist_ok=True)

fig, ax = plt.subplots(figsize=(8,6)) 

for g,color in zip(gs,palette):

    f = nogse.fit_nogse_vs_x_mixto_mixto(tnogse, g, n, x, tc_1, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2)
    error = 0

    nogse.plot_nogse_vs_x_data_ptG(ax, roi, x, f, tnogse, g, n, slic, color)

    table = np.vstack((x, f))
    np.savetxt(f"{directory}/{roi}_data_nogse_vs_x_tnogse={tnogse}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

    with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
        print(roi,  " - tc = ", tc_1, file=a)
        print("    ",  " - alpha = ", alpha_1, file=a)
        print("    ",  " - M0 = ", M0_1, file=a)
        print("    ",  " - D0 = ", D0_ext, file=a)
        print("    ",  " - tc = ", tc_2, file=a)
        print("    ",  " - alpha = ", alpha_2, file=a)
        print("    ",  " - M0 = ", M0_2, file=a)


fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_N={int(n)}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_N={int(n)}.png", dpi=600)
plt.close(fig)