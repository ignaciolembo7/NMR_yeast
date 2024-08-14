#NMRSI - Ignacio Lembo Ferrari - 11/08/2024

import numpy as np
import matplotlib.pyplot as plt
from brukerapi.dataset import Dataset as ds
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")
import lmfit
import os 
from protocols import nogse

# Create directory if it doesn't exist
directory = f"../simulations/contrast_vs_g/"
os.makedirs(directory, exist_ok=True)

###CAMBIAR POR CADA EXPERIMENTO###
N = 2
D0_ext = 2.289355e-12 #m2/ms
D0_int = 0.7e-12

gamma = 267.5221900 #1/ms.mT
alpha = 0.2 # =1 (libre) = inf (rest) ## es 1/alpha 

fig1, ax1 = plt.subplots(figsize=(8,6)) 
#fig2, ax2 = plt.subplots(figsize=(8,6)) 
#fig3, ax3 = plt.subplots(figsize=(8,6)) 
#fig3.subplots_adjust(bottom = 0.15, top =0.95, left = 0.15, right=0.95)
#fig4, ax4 = plt.subplots(figsize=(8,6)) 
#fig5, ax5 = plt.subplots(figsize=(8,6)) 

#Contraste fantomas 
#T_NOGSE = [10.0,21.5,43.0,70.0] #ms
#tau_c = [4.64,7.68,16.21,25.29] #ms
#tau_c = [2,2,2,2]
#idx = [0,1,2,3]

#Contraste levaduras zona difusión restringida
#T_NOGSE = [11.0, 15.0, 21.5, 25.0, 30.0, 40.0] #ms
#tau_c = [3.28, 1.27, 1.44, 2.8, 1.30, 1.10] #ms
#tau_c = [1.9, 1.9, 1.9, 1.9, 1.9, 1.9] #ms

#Contraste levaduras zona difusión restringida
#T_NOGSE = [15.0, 21.5, 30.0, 40.0] #ms
#tau_c = [3.28, 1.27, 1.44, 2.8, 1.30, 1.10] #ms
#tau_c = [1.9, 1.9, 1.9, 1.9] #ms
#idx = [0,1,2,3]

T_NOGSE = [21.5]
tau_c = [2.0] 
idx = [0]

curves = []
M0 = 1
color = ["tab:red", "tab:purple", "tab:green", "tab:blue"]

for i in idx: 

    g_contrast = np.linspace(1, 1300, 1000) #mT/m

    contrast = nogse.contrast_vs_g_rest(40.0, g_contrast, N, 2.0, 1000, D0_ext) + nogse.contrast_vs_g_rest(15.0, g_contrast, N, 100.0, 200, D0_int)
    #DM1 = nogse.delta_M_free(T_NOGSE[i], g_contrast, N, 1, M0, D0_ext)
    #DM2 = nogse.delta_M_free(T_NOGSE[i], g_contrast, N, 0.2, M0, D0_ext)
    #DM = nogse.delta_M_tort(T_NOGSE[i], g_contrast, N, tau_c[i], alpha, M0, D0_teo)

    l_D = np.sqrt(D0_ext*T_NOGSE[i])
    l_c = np.sqrt(D0_ext*tau_c[i])
    l_G = (D0_ext/(gamma*g_contrast))**(1/3) #m
    L_D = l_D/l_G
    L_c = l_c/l_G
    L_c_f = ((3/2)**(1/4))*(L_D**(-1/2))
    l_c_f = L_c_f*l_G 
    
    ax1.plot(g_contrast, contrast, linewidth = 2)
    ax1.set_xlabel("Intensidad de gradiente g [mT/m]", fontsize=18)
    ax1.set_ylabel("Contraste $\Delta M_\mathrm{NOGSE}$ [u.a.]", fontsize=18)
    #title = ax1.set_title(("Levaduras || $N$ = {}  || $\\alpha$ = {}").format(N, alpha), fontsize=15)
    ax1.legend(title='$T_\mathrm{{NOGSE}}$ (ms)', title_fontsize=15, fontsize=15, loc = 'upper right')
    ax1.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax1.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax1.tick_params(axis='y', labelsize=16, color='black')
    ax1.set_xlim(0, 1300)
    
    """
    ax2.plot(L_c_f, curves[i], "-", linewidth = 2, label="$T_{\mathrm{NOGSE}}$ = " + str(T_NOGSE[i]) + " ms - rest")
    ax2.set_xlabel(r"$L_C^f$", fontsize=18)
    ax2.set_ylabel(r"Contraste $\mathrm{NOGSE}$ $\Delta M$ [u.a.]", fontsize=18)
    #title = ax2.set_title(("Levaduras || $N$ = {}  ||  $D_0$ = {} ms ||  $\\alpha$ = {}").format(N, D0_teo_int, alpha), fontsize=15)
    ax2.legend(title='$T_\mathrm{{NOGSE}}$ (ms)', title_fontsize=15, fontsize=15)
    ax2.tick_params(axis='both', labelsize=15)
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax2.tick_params(axis='y', labelsize=16, color='black')
    ax2.set_xlim(0.4, 1.2)

    ax3.plot(L_D, DM1, "-", linewidth = 2, color = color[i], label= T_NOGSE[i] )
    ax3.plot(L_D, DM2, "-", linewidth = 2, color = color[i])
    ax3.set_xlabel("Longitud de difusión $L_d$", fontsize=27)
    ax3.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    #title = ax3.set_title(("Levaduras || $N$ = {}  ||  $D_0$ = {} ms ||  $\\alpha$ = {}").format(N, D0_teo_int, alpha), fontsize=15)
    ax3.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=18, fontsize=18)
    ax3.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax3.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax3.tick_params(axis='y', labelsize=18, color='black')
    ax3.set_xlim(0.5, 4)
    
    ax4.plot(l_c_f*1e6, curves[i], "-", linewidth = 2, label = T_NOGSE[i] )
    ax4.legend(title='$T_\mathrm{{NOGSE}}$ (ms)', title_fontsize=15, fontsize=15)
    ax4.set_xlabel(r"Longitud centro de filtro $l_{c,f}$ [\mu m]", fontsize=18)
    ax4.set_ylabel(r"Contraste $\mathrm{NOGSE}$ $\Delta M$ [u.a.]", fontsize=18)
    #title = ax4.set_title(("Levaduras || $N$ = {}  ||  $D_0$ = {} ms ||  $\\alpha$ = {}").format(N, D0_teo_int, alpha), fontsize=15)
    ax4.tick_params(axis='both', labelsize=15)
    ax4.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax4.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax4.tick_params(axis='y', labelsize=16, color='black')
    ax4.set_xlim(0.5,5)

    ax5.plot(L_c, curves[i], "-", linewidth = 3, label= T_NOGSE[i] )
    ax5.legend(title='$T_\mathrm{{NOGSE}}$ (ms)', title_fontsize=15, fontsize=15)
    ax5.set_xlabel(r"Longitud de restricción $L_C$", fontsize=18)
    ax5.set_ylabel(r"Contraste $\mathrm{NOGSE}$ $\Delta M$ [u.a.]", fontsize=18)
    #title = ax5.set_title(("Levaduras || $N$ = {}  ||  $D_0$ = {} ms ||  $\\alpha$ = {}").format(N, D0_teo_int, alpha), fontsize=15)
    ax5.tick_params(axis='both', labelsize=15)
    ax5.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax5.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax5.tick_params(axis='y', labelsize=16, color='black')
    ax5.set_xlim(0,1.5)
    """

fig1.tight_layout()
fig1.savefig(f"{directory}/contrast_vs_g.png", dpi = 600)
fig1.savefig(f"{directory}/contrast_vs_g.pdf")
plt.close(fig1)








"""

fig1.savefig(r"../simulations/levaduras_contraste_vs_g_rest_mismo_tauc.pdf")
fig1.savefig(r"../simulations/levaduras_contraste_vs_g_rest_mismo_tauc.png", dpi= 600)
fig2.savefig(r"../simulations/levaduras_contraste_vs_Lcf_rest_mismo_tauc.pdf")
fig2.savefig(r"../simulations/levaduras_contraste_vs_Lcf_rest_mismo_tauc.png", dpi= 600)
fig3.savefig(r"../simulations/levaduras_contraste_vs_Ld_rest_mismo_tauc.pdf")
fig3.savefig(r"../simulations/levaduras_contraste_vs_Ld_rest_mismo_tauc.png", dpi= 600)
fig4.savefig(r"../simulations/levaduras_contraste_vs_lcf__rest_mismo_tauc.pdf")
fig4.savefig(r"../simulations/levaduras_contraste_vs_lcf__rest_mismo_tauc.png", dpi= 600)
fig5.savefig(r"../simulations/levaduras_contraste_vs_Lc_rest_mismo_tauc.pdf")
fig5.savefig(r"../simulations/levaduras_contraste_vs_Lc_rest_mismo_tauc.png", dpi= 600)
"""

"""
#Contraste levaduras zona difusión tortuosa
T_NOGSE = [15.0, 21.5, 30.0, 40.0] #ms
tau_c = [2.70, 3.96, 5.70, 8.00] #ms
#tau_c = [2,2,2,2]

idx = [0,1,2,3]
curves2 = []
curves = []
M02 = 1

for i in idx: 
    g_contrast = np.linspace(1, 3000, 10000) #mT/m

    #DM = nogse.delta_M_restricted(T_NOGSE[i], g_contrast, N, tau_c[i], M0, D0_teo)
    #DM = nogse.delta_M_free(T_NOGSE[i], g_contrast, N, alpha, M0, D0_teo)
    DM = nogse.delta_M_tort(T_NOGSE[i], g_contrast, N, tau_c[i], alpha, M02, D0_teo)
    curves.append(DM)

    #curves.append(curves1[i] + curves2[i])

    l_D = np.sqrt(D0_teo*T_NOGSE[i])
    l_c = np.sqrt(D0_teo*tau_c[i])
    l_G = (D0_teo/(gamma*g_contrast))**(1/3) #m
    L_D = l_D/l_G
    L_c = l_c/l_G
    L_c_f = ((3/2)**(1/4))*(L_D**(-1/2))
    l_c_f = L_c_f*l_G 

    ax1.plot(g_contrast, curves[i], linestyle='dashed', label = "$T_{\mathrm{NOGSE}}$ = " + str(T_NOGSE[i]) + "- $\\tau_c$ = " + str(tau_c[i]) + " ms  - tort" )
    ax1.set_xlabel("Intensidad de gradiente g [mT/m]", fontsize=16)
    ax1.set_ylabel("Contraste $\Delta M_\mathrm{NOGSE}$ [u.a.]", fontsize=16)
    #title = ax1.set_title(("Levaduras || $N$ = {}  ||  $D_0$ = {} ms ||  $\\alpha$ = {}").format(N, D0_teo, alpha), fontsize=15)
    title = ax1.set_title(("Levaduras || $N$ = {}  ||  $\\alpha$ = {}").format(N, alpha), fontsize=15)
    ax1.legend(fontsize=10, loc = 'upper right')
    ax1.tick_params(axis='both', labelsize=15)
    ax1.set_xlim(-5,2500)

    ax2.plot(L_c_f, curves[i], linestyle='dashed', label="$T_{\mathrm{NOGSE}}$ = " + str(T_NOGSE[i]) + " ms - tort")
    ax2.legend(fontsize=12)
    ax2.set_xlabel(r"$L_C^f$", fontsize=16)
    ax2.set_ylabel(r"Contraste $\mathrm{NOGSE}$ $\Delta M$ [u.a.]", fontsize=16)
    title = ax2.set_title(("Levaduras || $N$ = {}  ||  $\\alpha$ = {}").format(N, alpha), fontsize=15)
    ax2.tick_params(axis='both', labelsize=15)
    ax2.set_xlim(0.4,1.5)

    ax3.plot(L_D, curves[i], linestyle='dashed', label="$T_{\mathrm{NOGSE}}$ = " + str(T_NOGSE[i]) + " ms - tort")
    ax3.legend(fontsize=12)
    ax3.set_xlabel(r"$L_d$", fontsize=16)
    ax3.set_ylabel(r"Contraste $\mathrm{NOGSE}$ $\Delta M$ [u.a.]", fontsize=16)
    title = ax3.set_title(("Levaduras || $N$ = {}  ||  $\\alpha$ = {}").format(N, alpha), fontsize=15)
    ax3.tick_params(axis='both', labelsize=15)
    ax3.set_xlim(0, 5.5)

    ax4.plot(l_c_f*1e6, curves[i], linestyle='dashed', label="$T_{\mathrm{NOGSE}}$ = " + str(T_NOGSE[i]) + " ms - tort")
    ax4.legend(fontsize=12)
    ax4.set_xlabel(r"$l_c^f ~[\mu m]$", fontsize=16)
    ax4.set_ylabel(r"Contraste $\mathrm{NOGSE}$ $\Delta M$ [u.a.]", fontsize=16)
    title = ax4.set_title(("Levaduras || $N$ = {}  ||  $\\alpha$ = {}").format(N, alpha), fontsize=15)
    ax4.tick_params(axis='both', labelsize=15)
    plt.xlim(0.5,12)

"""

"""
T_NOGSE = [11.0, 15.0, 21.5, 25.0, 30.0, 40.0] #ms
#tau_c = [3.28, 1.27, 1.44, 2.8, 1.30, 1.10] #ms
tau_c = [1.9, 1.9, 1.9, 1.9, 1.9, 1.9] #ms

curves = []
idx = [0,1,2,3,4,5]

for i in idx: 
    g_contrast = np.linspace(0.0001, 2500, 10000) #mT/m

    DM = nogse.delta_M_restricted(T_NOGSE[i], g_contrast, N, tau_c[i], M0, D0_teo)
    #DM = nogse.delta_M_free(T_NOGSE[i], g_contrast, N, alpha, M0, D0_teo)
    #DM = nogse.delta_M_tort(T_NOGSE[i], g_contrast, N, tau_c[i], alpha, M0, D0_teo)
    curves.append(DM)

    l_D = np.sqrt(D0_teo*T_NOGSE[i])
    l_c = np.sqrt(D0_teo*tau_c[i])
    l_G = (D0_teo/(gamma*g_contrast))**(1/3) #m
    L_D = l_D/l_G
    L_c = l_c/l_G
    L_c_f = ((3/2)**(1/4))*(L_D**(-1/2))
    l_c_f = L_c_f*l_G 

    ax1.plot(g_contrast, curves[i], label = "T =" + str(T_NOGSE[i]) + " ms - $\\tau_c$ = " + str(tau_c[i]) + " ms + rest" )
    ax1.set_xlabel("Intensidad de gradiente g [mT/m]", fontsize=16)
    ax1.set_ylabel("Contraste $\Delta M_\mathrm{NOGSE}$ [u.a.]", fontsize=16)
    title = ax1.set_title(("Levaduras || $N$ = {}  ||  $D_0$ = {} ms ||  $\\alpha$ = {}").format(N, D0_teo, alpha), fontsize=15)
    ax1.legend(fontsize=9, loc = 'upper right')
    ax1.tick_params(axis='both', labelsize=15)

    ax2.plot(L_c_f, curves[i], "-", label="TNOGSE " + str(T_NOGSE[i]) + " ms - D0 = 2.3e-12 m$^2$/ms")
    ax2.legend(fontsize=9)
    ax2.set_xlabel(r"$L_C^f$", fontsize=16)
    ax2.set_ylabel(r"Contraste $\mathrm{NOGSE}$ $\Delta M$ [u.a.]", fontsize=16)
    title = ax2.set_title(("Levaduras || $N$ = {}  ||  $D_0$ = {} ms ||  $\\alpha$ = {}").format(N, D0_teo, alpha), fontsize=15)
    ax2.tick_params(axis='both', labelsize=15)
    ax2.set_xlim(0.4, 1.5)

    ax3.plot(L_D, curves[i], "-", label="TNOGSE " + str(T_NOGSE[i]) + " ms - D0 = 2.3e-12 m$^2$/ms")
    ax3.legend(fontsize=9, loc = 'upper right')
    ax3.set_xlabel(r"$L_d$", fontsize=16)
    ax3.set_ylabel(r"Contraste $\mathrm{NOGSE}$ $\Delta M$ [u.a.]", fontsize=16)
    title = ax3.set_title(("Levaduras || $N$ = {}  ||  $D_0$ = {} ms ||  $\\alpha$ = {}").format(N, D0_teo, alpha), fontsize=15)
    ax3.tick_params(axis='both', labelsize=15)
    ax3.set_xlim(0, 6)

    ax4.plot(l_c_f*1e6, curves[i], "-", label="TNOGSE " + str(T_NOGSE[i]) + " ms - D0 = 2.3e-12 m$^2$/ms")
    ax4.legend(fontsize=9)
    ax4.set_xlabel(r"$l_c^f ~[\mu m]$", fontsize=16)
    ax4.set_ylabel(r"Contraste $\mathrm{NOGSE}$ $\Delta M$ [u.a.]", fontsize=16)
    title = ax4.set_title(("Levaduras || $N$ = {}  ||  $D_0$ = {} ms ||  $\\alpha$ = {}").format(N, D0_teo, alpha), fontsize=15)
    ax4.tick_params(axis='both', labelsize=15)
    ax4.set_xlim(0, 7.5)


"""