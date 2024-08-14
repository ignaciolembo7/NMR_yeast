#NMRSI - Ignacio Lembo Ferrari - 09/08/2024

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

D0_ext = 2.3e-12 # m2/ms extra
D0_int = 0.7e-12 # intra
#D0 = D0_ext

n = 2 
slic = 0

file_name = "levaduras_20240622"
folder = "rest_vs_x_restdist_mode"

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}"
os.makedirs(directory, exist_ok=True)


fig2, ax2 = plt.subplots(figsize=(8,6)) 

for i in ["G1","G2","G3","G4"]: 

    fig1, ax1 = plt.subplots(figsize=(8,6)) 

    data = np.loadtxt(f"{directory}/parameters_vs_tnogse_" + i + ".txt")

    tnogse = data[:, 0]
    sigma = data[:, 3]
    #tau_c = (l_c**2)/(2*D0*1e12)
    error_sigma = data[:, 4]

    #remover los elementos en la posicion 4, 6, 8 de tnogse y t_c 
    #tnogse = np.delete(tnogse, [4, 6, 8])
    #t_c = np.delete(t_c, [4, 6, 8])

    #ax1.errorbar(tnogse, sigma, yerr=error_sigma, fmt='o-', markersize=3, linewidth=2, capsize=5, label=f"{i}") #  
    ax1.plot(tnogse, sigma, 'o-', markersize=9, linewidth=2, label=f"{i}")
    ax1.set_xlabel("Tiempo de difusión $\mathrm{NOGSE}$ [ms]", fontsize=18)
    ax1.set_ylabel("Desviación estándar $\\sigma$", fontsize=18)
    ax1.legend(title='Gradiente', title_fontsize=18, fontsize=18, loc='best')
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax1.tick_params(axis='y', labelsize=16, color='black')
    title = ax1.set_title(f"$N$ = {n} | slice = {slic} ", fontsize=18)

    fig1.tight_layout()
    fig1.savefig(f"{directory}/sigma_vs_tnogse_" + i + ".png", dpi=600)
    fig1.savefig(f"{directory}/sigma_vs_tnogse_" + i + ".pdf")

    #ax2.errorbar(tnogse, sigma,  fmt='o-', markersize=3, linewidth=2, capsize=5, label=f"{i}") #yerr=error_t_c,
    ax2.plot(tnogse, sigma, 'o-', markersize=9, linewidth=2, label=f"{i}")
    ax2.set_xlabel("Tiempo de difusión $\mathrm{NOGSE}$ [ms]", fontsize=18)
    ax2.set_ylabel("Desviación estándar $\\sigma$", fontsize=18)
    ax2.legend(title='Gradiente', title_fontsize=18, fontsize=18, loc='best')
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax2.tick_params(axis='y', labelsize=16, color='black')
    title = ax2.set_title(f"$N$ = {n} | slice = {slic} ", fontsize=18)

fig2.tight_layout()
fig2.savefig(f"{directory}/sigma_vs_tnogse_allG.png", dpi=600)
fig2.savefig(f"{directory}/sigma_vs_tnogse_allG.pdf")