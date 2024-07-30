#NMRSI - Ignacio Lembo Ferrari - 02/05/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
from brukerapi.dataset import Dataset as ds
import cv2
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_20240613"
data_directory = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}"
folder = "contrast_vs_g_data"
exp = input("Experimento:")  # Nombre del experimento   
slic = 0

image_paths, method_paths = nogse.upload_contrast_data_v2(data_directory, slic)

#D0_ext = 2.3e-12
#D0_int = 0.7e-12 # intra

idx = 0
fig, ax = plt.subplots(figsize=(8,6)) #Imagen de todas las ROIS juntas
rois = ["ROI1"]

for roi in rois: 

    mask = np.loadtxt(f"rois/mask_"+ str(idx+1) +".txt")

    fig1, ax1 = plt.subplots(figsize=(8,6)) #Imagen de cada ROI

    T_nogse, g_contrast, n, f, error =  nogse.generate_contrast_roi_v2(image_paths, method_paths, mask, slic)

    directory = f"../results_{file_name}/{folder}/tnogse={T_nogse}_n={int(n)}_exp={exp}"
    os.makedirs(directory, exist_ok=True)

    nogse.plot_contrast_vs_g_data(ax, roi, g_contrast, f, error, T_nogse, n, slic)
    nogse.plot_contrast_vs_g_data(ax1, roi, g_contrast, f, error, T_nogse, n, slic)

    table = np.vstack((g_contrast, f, error))
    np.savetxt(f"{directory}/{roi}_data_contrast_vs_g_tnogse={T_nogse}_n={n}.txt", table.T, delimiter=' ', newline='\n')
    
    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={T_nogse}_n={n}.pdf")
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={T_nogse}_n={n}.png", dpi=600)
    plt.close(fig1)
    idx += 1

fig.tight_layout()
fig.savefig(f"{directory}/contrast_vs_g_tnogse={T_nogse}_n={n}.pdf")
fig.savefig(f"{directory}/contrast_vs_g_tnogse={T_nogse}_n={n}.png", dpi=600)
plt.close(fig)