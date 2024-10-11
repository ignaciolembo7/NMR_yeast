#NMRSI - Ignacio Lembo Ferrari - 27/09/2024

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from protocols import nogse
import os
sns.set_theme(context='paper')
sns.set_style("whitegrid")

roi = 'ROI1'
slic = 0
exp = 1
file_name = "levaduras_20240613"
folder = "fit_contrast_vs_g_free_restdist"
A0_folder = "con_A0"
modelo = "Free+RestDist"
# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}"
os.makedirs(directory, exist_ok=True)

#palette = sns.color_palette("tab10", 10) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)
# Definir manualmente una paleta de colores personalizada (sin verde ni naranja ya que los reservo para indicar las zonas intra y extracelular respectivamente).
palette = [
    "#1f77b4",  # Azul
    "#00008B",  # Azul oscuro
    "#9467bd",  # Púrpura
    "#e377c2",  # Rosa
    "#7f7f7f",  # Gris
    "#8c564b",  # Marrón
    "#f1c40f",  # Amarillo oscuro
    "#d62728",  # Rojo

]
sns.set_palette(palette)

fig1, ax1 = plt.subplots(figsize=(8,6)) 
fig2, ax2 = plt.subplots(figsize=(8,6)) 
fig3, ax3 = plt.subplots(figsize=(8,6)) 
fig4, ax4 = plt.subplots(figsize=(8,6)) 
fig5, ax5 = plt.subplots(figsize=(8,6)) 

gamma = 267.5221900 #1/ms.mT
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # intra
tnogses = [15.0, 17.5, 21.5, 25.0, 27.5, 30.0, 35.0, 40.0]
ies = [1, 2, 3, 4, 5, 6, 7, 8]
n = 2

for i, tnogse, color in zip(ies, tnogses, palette):    

    data_fit = np.loadtxt(f"../results_{file_name}/{folder}/tnogse={tnogse}_N={n}_exp={exp}/{roi}_fit_contrast_vs_g_tnogse={tnogse}_N={n}_exp={exp}.txt")
    data = np.loadtxt(f"../results_{file_name}/plot_contrast_vs_g_data/{A0_folder}/tnogse={tnogse}_N={n}_exp={exp}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={n}.txt")

    g_data = data[:, 0]
    f_data = data[:, 1] #/ data[:, 0]

    g_fit = data_fit[:, 0]
    f_fit = data_fit[:, 1] #/ data[:, 0]

    #Ordeno en el caso de que el vector g no venga ordenado de menor a mayor
    data = list(zip(g_data, f_data))
    sorted_data = sorted(data, key=lambda x: x[0])
    g_data, f_data = zip(*sorted_data)
    g_data = np.array(g_data, dtype=float)
    f_data = np.array(f_data, dtype=float)

    ld_data = np.sqrt(2*D0_ext*tnogse)
    #l_c = np.sqrt(D0_ext*tau_c[idx])
    lG_data = ((2**(3/2))*D0_ext/(gamma*g_data))**(1/3)
    Ld_data = ld_data/lG_data
    #L_c = l_c/l_G
    Lcf_data = ((3/2)**(1/4))*(Ld_data**(-1/2))
    lcf_data = Lcf_data*lG_data 
    
    #Ordeno en el caso de que el vector g no venga ordenado de menor a mayor
    # data_fit = list(zip(g_fit, f_fit))
    # sorted_data_fit = sorted(data_fit, key=lambda x: x[0])
    # g_fit, f_fit = zip(*sorted_data_fit)
    # g_fit = np.array(g_fit, dtype=float)
    # f_fit = np.array(f_fit, dtype=float)

    # data_A0 = np.loadtxt(f"../results_{file_name}/{folder}/tabla_A0_hahn_cpmg.txt")
    # signal_A0_hahn = data_A0[i, 1]
    # signal_A0_error_hahn = data_A0[i,2]
    # signal_A0_cpmg = data_A0[i, 3]
    # signal_A0_error_cpmg = data_A0[i,4]
    # A0 = (signal_A0_cpmg + signal_A0_hahn)/2
    # f_fit = f_fit/A0

    ld_fit = np.sqrt(2*D0_ext*tnogse)
    #l_c = np.sqrt(D0_ext*tau_c[idx])
    lG_fit = ((2**(3/2))*D0_ext/(gamma*g_fit))**(1/3)
    Ld_fit = ld_fit/lG_fit
    #L_c = l_c/l_G
    Lcf_fit = ((3/2)**(1/4))*(Ld_fit**(-1/2))
    lcf_fit = Lcf_fit*lG_fit 
    
    ax1.plot(g_fit, f_fit, "-", markersize=7, linewidth = 2, label = tnogse, color=color)
    ax1.plot(g_data, f_data, "o-", markersize=7, linewidth = 2, color=color)
    ax1.set_xlabel("Intensidad de gradiente g [mT/m]", fontsize=27)
    ax1.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    #title = ax1.set_title(f"$N$ = {n} || D_0 = {D0_ext} m$^2$/ms", fontsize=18)
    ax1.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=18, fontsize=18, loc='upper right')
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax1.tick_params(axis='y', labelsize=16, color='black')
    #ax1.set_xlim(0, 1400)

    ax2.plot(Lcf_fit, f_fit, "-", markersize=7, linewidth = 2, label = tnogse, color=color)
    ax2.plot(Lcf_data, f_data, "o-", markersize=7, linewidth = 2, color=color)
    ax2.set_xlabel("Longitud de centro del filtro $L_c^f$", fontsize=27)
    ax2.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    #title = ax2.set_title(f"$N$ = {n} || D_0 = {D0_ext} m$^2$/ms", fontsize=18)
    ax2.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=18, fontsize=18, loc='upper right')
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax2.tick_params(axis='y', labelsize=16, color='black')
    #ax2.set_xlim(0.4, 1.2)

    ax3.plot(Ld_fit, f_fit, "-", markersize=7, linewidth = 2, label= tnogse, color=color)
    ax3.plot(Ld_data, f_data, "o-", markersize=7, linewidth = 2, color=color)
    ax3.set_xlabel("Longitud de difusión $L_d$", fontsize=27)
    ax3.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    #title = ax3.set_title(f"$N$ = {n} || D_0 = {D0_ext} m$^2$/ms", fontsize=18)
    ax3.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=18, fontsize=18, loc='upper right')
    ax3.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax3.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax3.tick_params(axis='y', labelsize=16, color='black')
    #ax3.set_xlim(0.6, 4.6)

    ax4.plot(lcf_fit*1e6, f_fit, "-", markersize=7, linewidth = 2, label = tnogse, color=color)
    ax4.plot(lcf_data*1e6, f_data, "o-", markersize=7, linewidth = 2, color=color)
    ax4.set_xlabel("Longitud de centro del filtro $lc^f ~[\mu m]$", fontsize=27)
    ax4.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    #title = ax4.set_title(f"$N$ = {n} || D_0 = {D0_ext} m$^2$/ms", fontsize=18)
    ax4.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=18, fontsize=18, loc='upper right')
    ax4.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax4.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax4.tick_params(axis='y', labelsize=16, color='black')
    #ax4.set_xlim(0.5,11)

    #ax5.plot(L_c, f, "-o", markersize=7, linewidth = 2, label= T_NOGSE[i] )
    #ax5.set_xlabel(r"Longitud de restricción $L_C$", fontsize=18)
    #ax5.set_ylabel(r"Contraste $\mathrm{NOGSE}$ $\Delta M$ [u.a.]", fontsize=18)
    #title = ax5.set_title(("Mouse Brain || $T_\mathrm{{NOGSE}}$ = {} ms || $N$ = {} || D_0 = {} $m^2$/ms").format(T_NOGSE[0],N, D0_int), fontsize=18)
    #ax5.legend(title='ROI', title_fontsize=18, fontsize=18, loc = 'best')
    #ax5.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    #ax5.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    #ax5.tick_params(axis='y', labelsize=16, color='black')
    #ax5.set_xlim(0,1.5)

fig1.tight_layout()
fig1.savefig(f"{directory}/{roi}_contrast_vs_g_N={n}_ptTNOGSE.pdf")
fig1.savefig(f"{directory}/{roi}_contrast_vs_g_N={n}_ptTNOGSE.png", dpi= 600)
fig2.tight_layout()
fig2.savefig(f"{directory}/{roi}_contrast_vs_L_cf_N={n}_ptTNOGSE.pdf")
fig2.savefig(f"{directory}/{roi}_contrast_vs_L_cf_N={n}_ptTNOGSE.png", dpi= 600)
fig3.tight_layout()
fig3.savefig(f"{directory}/{roi}_contrast_vs_Ld_N={n}_ptTNOGSE.pdf")
fig3.savefig(f"{directory}/{roi}_contrast_vs_Ld_N={n}_ptTNOGSE.png", dpi= 600)
fig4.tight_layout()
fig4.savefig(f"{directory}/{roi}_contrast_vs_lcf_N={n}_ptTNOGSE.pdf")
fig4.savefig(f"{directory}/{roi}_contrast_vs_lcf_N={n}_ptTNOGSE.png", dpi= 600)