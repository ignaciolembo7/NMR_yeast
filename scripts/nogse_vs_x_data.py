#NMRSI - Ignacio Lembo Ferrari - 28/07/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

rois = ["ROI1"]

file_name = "levaduras_20230427"
folder = "nogse_vs_x_data"
A0_folder = "sin_A0"
data_directory = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}"
slic = 0 # slice que quiero ver
exp = 1 

image_paths, method_paths = nogse.upload_NOGSE_vs_x_data(data_directory, slic)

#D0_ext = 2.3e-12
#D0_int = 0.7e-12 # intra

fig, ax = plt.subplots(figsize=(8,6)) 

idx = 0

for roi in rois: 

    mask = np.loadtxt(f"rois/mask_{idx+1}.txt")

    fig1, ax1 = plt.subplots(figsize=(8,6))
    
    T_nogse, g, x, n, f, error =  nogse.generate_NOGSE_vs_x_roi(image_paths, method_paths, mask, slic)

    # Create directory if it doesn't exist
    directory = f"../results_{file_name}/{folder}/slice={slic}/{A0_folder}/tnogse={T_nogse}_g={g}_N={int(n)}_exp={exp}"
    os.makedirs(directory, exist_ok=True)

    nogse.plot_nogse_vs_x_data(ax, roi, x, f, error, T_nogse, g, n, slic) 
    nogse.plot_nogse_vs_x_data(ax1, roi, x, f, error, T_nogse, n, n, slic)

    table = np.vstack((x, f, error))
    np.savetxt(f"{directory}/{roi}_data_nogse_vs_x_tnogse={T_nogse}_g={g}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={T_nogse}_g={g}_N={int(n)}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={T_nogse}_g={g}_N={int(n)}.png", dpi=600)
    plt.close(fig1)
    
    idx+=1

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={T_nogse}_g={g}_N={int(n)}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={T_nogse}_g={g}_N={int(n)}.png", dpi=600)
plt.close(fig)
