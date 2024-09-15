#NMRSI - Ignacio Lembo Ferrari - 05/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_generated"
folder = "nogse_vs_x_data_restdistmode_restdistmode"
slic = 0 # slice que quiero ver
exp = 1 
roi = "ROI1"

D0_ext = 2.3e-12
D0_int = 0.7e-12 
lc_int = 2.0
sigma_int = 0.34
M0_int = 862
lc_ext = 5.48
sigma_ext = 0.96
M0_ext = 3691

tnogse = 21.5
g = 75.0
n = 2
x = np.linspace(0, 10.75, 21)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/slice={slic}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}"
os.makedirs(directory, exist_ok=True)

f = nogse.M_nogse_restdistmode_restdistmode(tnogse, g, n, x, lc_int, sigma_int, lc_ext, sigma_ext, M0_int, M0_ext, D0_int, D0_ext)
error = 0

fig, ax = plt.subplots(figsize=(8,6)) 
nogse.plot_nogse_vs_x_data(ax, roi, x, f, error, tnogse, g, n, slic)

table = np.vstack((x, f))
np.savetxt(f"{directory}/{roi}_data_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
    print(roi,  " - lc_int = ", lc_int, file=a)
    print("    ",  " - sigma_int = ", sigma_int, file=a)
    print("    ",  " - M0_int = ", M0_int, file=a)
    print("    ",  " - D0_int = ", D0_int, file=a)
    print("    ",  " - lc_ext = ", lc_ext, file=a)
    print("    ",  " - sigma_ext = ", sigma_ext, file=a)
    print("    ",  " - M0_ext = ", M0_ext, file=a)
    print("    ",  " - D0_ext = ", D0_ext, file=a)
    
fig.tight_layout()
fig.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
fig.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
plt.close(fig)