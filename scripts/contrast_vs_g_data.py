#NMRSI - Ignacio Lembo Ferrari - 02/05/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

#D0_ext = 2.3e-12
#D0_int = 0.7e-12 # intra

file_name = "levaduras_20240613"
data_directory = f"C:/Users/Ignacio Lembo/Documents/data/data_{file_name}"
folder = "contrast_vs_g_data"
exp = input("Experimento:")
A0 = "con_A0"
slic = 0
rois = ["ROI1"]
masks = [1]
palette = sns.color_palette("tab10", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)
num_tnogse = 7

image_paths, method_paths = nogse.upload_contrast_data_v2(data_directory, slic)

fig, ax = plt.subplots(figsize=(8,6)) #Imagen de todas las ROIS juntas

for roi, mask, color in zip(rois,masks,palette): 

    mask = np.loadtxt(f"rois/mask_{mask}.txt")

    fig1, ax1 = plt.subplots(figsize=(8,6)) #Imagen de cada ROI

    tnogse, g_contrast, n, f, error, f_hahn, error_hahn, f_cpmg, error_cpmg =  nogse.generate_contrast_roi_v2(image_paths, method_paths, mask, slic)

    directory = f"../results_{file_name}/{folder}/{A0}/tnogse={tnogse}_N={int(n)}_exp={exp}"
    os.makedirs(directory, exist_ok=True)

    data_A0 = np.loadtxt(f"../results_{file_name}/contrast_vs_g_data/{A0}/tabla_A0_hahn_cpmg.txt")

    signal_A0_hahn = data_A0[num_tnogse, 1]
    signal_A0_error_hahn = data_A0[num_tnogse,2]
    signal_A0_cpmg = data_A0[num_tnogse, 3]
    signal_A0_error_cpmg = data_A0[num_tnogse,4]

    print(signal_A0_hahn, signal_A0_error_hahn, signal_A0_cpmg, signal_A0_error_cpmg)

    f_hahn_A0 = f_hahn/signal_A0_hahn
    f_cpmg_A0 = f_cpmg/signal_A0_cpmg

    #propagación del error en el cociente
    error_hahn_A0 = f_hahn_A0*np.sqrt( (np.array(error_hahn)/np.array(f_hahn))**2 + (signal_A0_error_hahn/signal_A0_hahn)**2)
    error_cpmg_A0 = f_cpmg_A0*np.sqrt( (np.array(error_cpmg)/np.array(f_cpmg))**2 + (signal_A0_error_cpmg/signal_A0_cpmg)**2)

    f_A0 = f_cpmg_A0 - f_hahn_A0
    error_A0 = np.sqrt(error_hahn_A0**2 + error_cpmg_A0**2)

    table = np.vstack((g_contrast, f_A0, error_A0, f_hahn_A0, error_hahn_A0, f_cpmg_A0, error_cpmg_A0))
    np.savetxt(f"{directory}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={n}.txt", table.T, delimiter=' ', newline='\n')

    nogse.plot_contrast_vs_g_data(ax, roi, g_contrast, f_A0, error_A0, tnogse, n, slic, color)
    nogse.plot_contrast_vs_g_data(ax1, roi, g_contrast, f_A0, error_A0, tnogse, n, slic, color)

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={n}.pdf")
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={n}.png", dpi=600)
    plt.close(fig1)

fig.tight_layout()
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={n}.pdf")
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={n}.png", dpi=600)
plt.close(fig)