#NMRSI - Ignacio Lembo Ferrari - 29/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

tnogse = float(input('Tnogse [ms]: ')) #ms
n = 2 #float(input('N: '))
file_name = "levaduras_20240613"
folder = "plot_contrast_vs_g_data"
A0 = "con_A0"
slic = 0 # slice que quiero ver
directory = f"../results_{file_name}/{folder}/{A0}/tnogse={tnogse}_n={int(n)}_exp=1"
os.makedirs(directory, exist_ok=True)

rois = ["ROI1"]
exps = [1]
#ids = ["1 - 20 horas y 16 minutos." ]#, "1 - 22 días, 20 horas y 59 minutos"] #["1 - 15 días, 6 horas y 59 minutos", "2 - 18 días, 20 horas y 26 minutos"]#["1 - 8 días*, 5 horas y 29 minutos - 21/05", "4 - 10 días, 2 horas", "5 - 13 días, 6 horas y 51 minutos", "6 - 18 días, 22 horas y 45 minutos", "7 - 19 días, 3 horas y 29 minutos"]
num_tnogse = 4

fig, ax = plt.subplots(figsize=(8,6)) 
palette = sns.color_palette("tab10", len(exps)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)
for exp, roi, color in zip(exps, rois, palette):

    data = np.loadtxt(f"../results_{file_name}/plot_contrast_vs_g_data/sin_A0/tnogse={tnogse}_N={int(n)}_exp={exp}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={int(n)}.txt")
    g_contrast = data[:, 0]
    f_hahn = data[:, 3]
    error_hahn = data[:, 4]
    f_cpmg = data[:, 5]
    error_cpmg = data[:, 6]
    f = np.array(f_cpmg) - np.array(f_hahn)
    error = np.sqrt(np.array(error_cpmg)**2 + np.array(error_hahn)**2)

    data_A0 = np.loadtxt(f"../results_{file_name}/plot_contrast_vs_g_data/tabla_A0_hahn_cpmg.txt")
    signal_A0_hahn = data_A0[num_tnogse, 1]
    signal_A0_error_hahn = data_A0[num_tnogse,2]
    signal_A0_cpmg = data_A0[num_tnogse, 3]
    signal_A0_error_cpmg = data_A0[num_tnogse,4]
    print(signal_A0_hahn, signal_A0_error_hahn, signal_A0_cpmg, signal_A0_error_cpmg)
    f_hahn_A0 = f_hahn/signal_A0_hahn
    f_cpmg_A0 = f_cpmg/signal_A0_cpmg
    error_hahn_A0 = f_hahn_A0*np.sqrt( (np.array(error_hahn)/np.array(f_hahn))**2 + (signal_A0_error_hahn/signal_A0_hahn)**2)
    error_cpmg_A0 = f_cpmg_A0*np.sqrt( (np.array(error_cpmg)/np.array(f_cpmg))**2 + (signal_A0_error_cpmg/signal_A0_cpmg)**2)
    f_A0 = f_cpmg_A0 - f_hahn_A0
    error_A0 = np.sqrt(error_hahn_A0**2 + error_cpmg_A0**2)
    table = np.vstack((g_contrast, f_A0, error_A0, f_hahn_A0, error_hahn_A0, f_cpmg_A0, error_cpmg_A0))
    np.savetxt(f"{directory}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={n}.txt", table.T, delimiter=' ', newline='\n')
    nogse.plot_contrast_vs_g_data(ax, roi, g_contrast, f_A0, error_A0, tnogse, n, slic, color)

    # table = np.vstack((g_contrast, f, error, f_hahn, error_hahn, f_cpmg, error_cpmg))
    # np.savetxt(f"{directory}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={n}.txt", table.T, delimiter=' ', newline='\n')
    # nogse.plot_contrast_vs_g_data(ax, roi, g_contrast, f, error, tnogse, n, slic, color)

fig.tight_layout()
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}.pdf")
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}.png", dpi=600)
plt.close(fig)