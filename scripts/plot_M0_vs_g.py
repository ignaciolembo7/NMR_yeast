#NMRSI - Ignacio Lembo Ferrari - 26/08/2024

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

D0_ext = 2.3e-12 # m2/ms extra
D0_int = 0.7e-12 # intra
D0 = D0_ext

n = 2

file_name = "levaduras_20240622"
folder = "nogse_vs_x_restdist_mode"
A0 = "sin_A0"
slic = 0 # slice que quiero ver

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}"
os.makedirs(directory, exist_ok=True)

fig2, ax2 = plt.subplots(figsize=(8,6)) 

#gs = ["G1","G4","G3","G4"]
rois =  ["ROI1","ROI1", "ROI1","ROI1","ROI1","ROI1","ROI1","ROI1","ROI1","ROI1"]
tnogses = ["15.0","17.5","21.5","25.0","27.5","30.0","32.5","35.0","37.5","40.0"] 

for roi, tnogse in zip(rois, tnogses): 

    fig1, ax1 = plt.subplots(figsize=(8,6)) 

    data = np.loadtxt(f"{directory}/{roi}_parameters_vs_g_tnogse={tnogse}.txt")

    grad = data[:, 0]
    M0 = data[:, 6]

    # Obtener los índices que ordenarían grad
    sorted_indices = np.argsort(grad)
    # Ordenar grad y M0 usando esos índices
    grad = grad[sorted_indices]
    M0 = M0[sorted_indices]

    #error_t_c = data[:, 2]

    #remover los elementos en la posicion 4, 6, 8 de tnogse y t_c 
    #tnogse = np.delete(tnogse, [4, 6, 8])
    #t_c = np.delete(t_c, [4, 6, 8])

    #ax1.errorbar(grad, M0, fmt='o-', markersize=3, linewidth=2, capsize=5, label=f"{tnogse}") #  yerr=error_t_c,
    ax1.plot(grad, M0, 'o-', markersize=7, linewidth=2, label=f"{tnogse}")
    ax1.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=27)
    ax1.set_ylabel("$M_0$", fontsize=27)
    #ax1.set_xscale('log')  # Cambiar el eje x a escala logarítmica
    #ax1.set_yscale('log')  # Cambiar el eje y a escala logarítmica
    ax1.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=15, fontsize=15, loc='best')
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax1.tick_params(axis='y', labelsize=16, color='black')
    title = ax1.set_title(f"$N$ = {n} | slice = {slic} ", fontsize=15)

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_M0_vs_g_tnogse={tnogse}.png", dpi=600)
    fig1.savefig(f"{directory}/{roi}_M0_vs_g_tnogse={tnogse}.pdf")

    #ax2.errorbar(grad, M0, fmt='o-', markersize=3, linewidth=2, capsize=5, label=f"{tnogse}")
    ax2.plot(grad, M0, 'o-', markersize=7, linewidth=2, label=f"{tnogse}")
    ax2.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=27)
    ax2.set_ylabel("$M_0$", fontsize=27)
    #ax2.set_xscale('log')  # Cambiar el eje x a escala logarítmica
    #ax2.set_yscale('log')  # Cambiar el eje y a escala logarítmica
    ax2.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=15, fontsize=15, loc='best')
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax2.tick_params(axis='y', labelsize=16, color='black')
    title = ax2.set_title(f"$N$ = {n} | slice = {slic} ", fontsize=15)

fig2.tight_layout()
fig2.savefig(f"{directory}/M0_vs_g_alltnogse.png", dpi=600)
fig2.savefig(f"{directory}/M0_vs_g_alltnogse.pdf")

#print("Valor medio M0 = ", np.mean(M0), "+-", np.std(M0))