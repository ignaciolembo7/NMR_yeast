#NMRSI - Ignacio Lembo Ferrari - 08/08/2024

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
sns.set_theme(context='paper')
sns.set_style("whitegrid") 

file_name = "levaduras_20240613"
folder = "contrast_vs_g_var_adimensional"
A0 = "con_A0"
directory = f"../results_{file_name}/{folder}/{A0}"
os.makedirs(directory, exist_ok=True)

#Constantes
D0_ext = 2.289355e-12 #m2/ms
D0_int = 0.7e-12
gamma = 267.5221900 #1/ms.mT
#alpha = 0.2 # =1 (libre) = inf (rest) ## es 1/alpha 

fig1, ax1 = plt.subplots(figsize=(8,6)) 
fig2, ax2 = plt.subplots(figsize=(8,6)) 
fig3, ax3 = plt.subplots(figsize=(8,6)) 
fig4, ax4 = plt.subplots(figsize=(8,6)) 
fig5, ax5 = plt.subplots(figsize=(8,6)) 

#Unidades 
#[g] = mT/m 

n = 2
exp = 1 
roi = 'ROI1'

#palette = sns.color_palette("tab10", 10) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Definir manualmente una paleta de colores personalizada (sin verde ni naranja ya que los reservo para indicar las zonas intra y extracelular respectivamente).
palette = [
    #"#aec7e8"   # Azul claro
    "#1f77b4",  # Azul
    "#9467bd",  # Púrpura
    "#e377c2",  # Rosa
    "#7f7f7f",  # Gris
    "#8c564b",  # Marrón
    "#f1c40f",  # Amarillo oscuro
    "#d62728",  # Rojo
]

# Asignar la paleta personalizada
sns.set_palette(palette)

# Mostrar la paleta
#sns.palplot(palette)

for i, color in zip([[17.5, n],[21.5, n],[25.0, n],[27.5, n],[30.0, n],[35.0, n],[40.0, n]], palette):    

    data = np.loadtxt(f"../results_{file_name}/contrast_vs_g_data/{A0}/tnogse={i[0]}_N={n}_exp={exp}/{roi}_data_contrast_vs_g_tnogse={i[0]}_N={n}.txt")
    g = data[:, 0]
    f = data[:, 1] #/data[0,1]

    #Ordeno en el caso de que el vector g no venga ordenado de menor a mayor
    data = list(zip(g, f))
    sorted_data = sorted(data, key=lambda x: x[0])
    g, fit1 = zip(*sorted_data)
    g_contrast = np.array(g, dtype=float)
    f = np.array(fit1, dtype=float)

    l_d = np.sqrt(2*D0_ext*i[0])
    #l_c = np.sqrt(D0_ext*tau_c[idx])
    l_G = ((2**(3/2))*D0_ext/(gamma*g_contrast))**(1/3)
    L_d = l_d/l_G
    #L_c = l_c/l_G
    L_c_f = ((3/2)**(1/4))*(L_d**(-1/2))
    l_c_f = L_c_f*l_G 
    
    ax1.plot(g_contrast, f, "-o", markersize=7, linewidth = 2, label = i[0], color=color)
    ax1.set_xlabel("Intensidad de gradiente g [mT/m]", fontsize=27)
    ax1.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    #title = ax1.set_title(f"$N$ = {n} || D_0 = {D0_ext} m$^2$/ms", fontsize=18)
    ax1.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=18, fontsize=18, loc='upper right')
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax1.tick_params(axis='y', labelsize=16, color='black')
    #ax1.set_xlim(0, 1400)

    ax2.plot(L_c_f, f, "-o", markersize=7, linewidth = 2, label = i[0], color=color)
    ax2.set_xlabel("Longitud de centro del filtro $L_C^f$", fontsize=27)
    ax2.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    #title = ax2.set_title(f"$N$ = {n} || D_0 = {D0_ext} m$^2$/ms", fontsize=18)
    ax2.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=18, fontsize=18, loc='upper right')
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax2.tick_params(axis='y', labelsize=16, color='black')
    #ax2.set_xlim(0.4, 1.2)

    ax3.plot(L_d, f, "-o", markersize=7, linewidth = 2, label= i[0], color=color)
    ax3.set_xlabel("Longitud de difusión $L_d$", fontsize=27)
    ax3.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    #title = ax3.set_title(f"$N$ = {n} || D_0 = {D0_ext} m$^2$/ms", fontsize=18)
    ax3.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=18, fontsize=18, loc='upper right')
    ax3.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax3.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax3.tick_params(axis='y', labelsize=16, color='black')
    #ax3.set_xlim(0.6, 4.6)

    ax4.plot(l_c_f*1e6, f, "-o", markersize=7, linewidth = 2, label = i[0], color=color)
    ax4.set_xlabel("Longitud de centro del filtro $l_c^f ~[\mu m]$", fontsize=27)
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
fig1.savefig(f"{directory}/contrast_vs_g_tnogse={i[0]}_N={n}.pdf")
fig1.savefig(f"{directory}/contrast_vs_g_tnogse={i[0]}_N={n}.png", dpi= 600)
fig2.tight_layout()
fig2.savefig(f"{directory}/contrast_vs_L_cf_tnogse={i[0]}_N={n}.pdf")
fig2.savefig(f"{directory}/contrast_vs_L_cf_tnogse={i[0]}_N={n}.png", dpi= 600)
fig3.tight_layout()
fig3.savefig(f"{directory}/contrast_vs_Ld_tnogse={i[0]}_N={n}.pdf")
fig3.savefig(f"{directory}/contrast_vs_Ld_tnogse={i[0]}_N={n}.png", dpi= 600)
fig4.tight_layout()
fig4.savefig(f"{directory}/contrast_vs_lcf_tnogse={i[0]}_N={n}.pdf")
fig4.savefig(f"{directory}/contrast_vs_lcf_tnogse={i[0]}_N={n}.png", dpi= 600)