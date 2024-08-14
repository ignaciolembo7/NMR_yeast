import numpy as np
import matplotlib.pyplot as plt
from brukerapi.dataset import Dataset as ds
import glob 
import seaborn as sns
#sns.set(context='paper')
#sns.set_style("whitegrid")
import lmfit
import sys

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
fig4, ax4 = plt.subplots(figsize=(8,6)) 
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
T_NOGSE = [15.0, 21.5, 25.0, 30.0, 40.0] #ms
#tau_c = [3.28, 1.27, 1.44, 2.8, 1.30, 1.10] #ms
tau_c_int = [1.82, 2.06, 2.0, 1.83, 1.69 ] #ms
M0_int = [0.65,0.40,0.55,0.35,0.35]

#Contraste levaduras zona difusión tortuosa
T_NOGSE = [15.0, 21.5, 25.0, 30.0, 40.0] #ms
tau_c_ext = [2.70, 3.96, 2.05, 5.70, 8.00] #ms
alpha = [0.25, 0.25, 0.25, 0.25, 0.25]
M0_ext = [0.95,0.88,0.84,1.2,1.0]

#Contraste levaduras zona difusión restringida
T_NOGSE = [21.5] #ms
#tau_c = [3.28, 1.27, 1.44, 2.8, 1.30, 1.10] #ms
tau_c_int = [2.06] #ms
M0_int = [0.40]

#Contraste levaduras zona difusión tortuosa
T_NOGSE = [21.5] #ms
tau_c_ext = [3.96] #ms
alpha = [0.25]
M0_ext = [0.6]

idx = [0]
curves1 = []
curves2 = []
curves = []


color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for i in idx: 
    g_contrast = np.linspace(1, 3000, 10000) #mT/m
    DM = nogse.contrast_vs_g_rest(T_NOGSE[i], g_contrast, N, tau_c_int[i], M0_int[i], D0_int)
    curves1.append(DM)

for i in idx: 
    g_contrast = np.linspace(1, 3000, 10000) #mT/m
    DM = nogse.delta_M_mixto(T_NOGSE[i], g_contrast,  N, tau_c_ext[i], alpha[i], M0_ext[i], D0_ext) 
    curves2.append(DM)
    
    curves.append(curves1[i] + curves2[i])
    
    l_D = np.sqrt(D0_ext*T_NOGSE[i])
    l_c = np.sqrt(D0_ext*tau_c_ext[i])
    l_G = (D0_ext/(gamma*g_contrast))**(1/3) #m
    L_D = l_D/l_G
    L_c = l_c/l_G
    L_c_f = ((3/2)**(1/4))*(L_D**(-1/2))
    l_c_f = L_c_f*l_G 

    ax1.plot(g_contrast, curves[i], linewidth = 3)
    #ax1.plot(g_contrast, curves2[i], "--",  linewidth = 2, label = "$T_{\mathrm{NOGSE}}$ = " + str(T_NOGSE[i]) + " ms " )
    ax1.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=27)
    ax1.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$ [u.a.]", fontsize=27)
    #title = ax1.set_title(("Levaduras || $N$ = {}  || $\\alpha$ = {}").format(N, alpha), fontsize=15)
    #ax1.legend(title='$T_\mathrm{{NOGSE}}$ (ms)', title_fontsize=15, fontsize=15, loc = 'upper right')
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

    ax3.plot(L_D, curves[i], "-", linewidth = 2, label= T_NOGSE[i]  )
    ax3.set_xlabel(r"Longitud de difusión $L_D$", fontsize=18)
    ax3.set_ylabel(r"Contraste $\mathrm{NOGSE}$ $\Delta M$ [u.a.]", fontsize=18)
    #title = ax3.set_title(("Levaduras || $N$ = {}  ||  $D_0$ = {} ms ||  $\\alpha$ = {}").format(N, D0_teo_int, alpha), fontsize=15)
    ax3.legend(title='$T_\mathrm{{NOGSE}}$ (ms)', title_fontsize=15, fontsize=15)
    ax3.tick_params(axis='both', labelsize=15)
    ax3.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax3.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax3.tick_params(axis='y', labelsize=16, color='black')
    ax3.set_xlim(0, 6)

    ax4.plot(l_c_f*1e6, curves1[i], "--", color = color[i],  linewidth = 2)
    ax4.plot(l_c_f*1e6, curves2[i], "-",color = color[i], linewidth = 2, label = str(T_NOGSE[i]))
    ax4.legend(title='$t_d$ [ms]', title_fontsize=21, fontsize=21)
    ax4.set_xlabel(r"Center filter length $l_{c,f} ~ [\mu m]$", fontsize=21)
    ax4.set_ylabel(r"$\mathrm{NOGSE}$ contrast $\Delta M$", fontsize=21)
    #title = ax4.set_title(("Levaduras || $N$ = {}  ||  $D_0$ = {} ms ||  $\\alpha$ = {}").format(N, D0_teo_int, alpha), fontsize=15)
    ax4.tick_params(axis='both', labelsize=15)
    ax4.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax4.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax4.tick_params(axis='y', labelsize=16, color='black')
    ax4.set_xlim(0.5,11)


    ax5.plot(L_c, curves[i], "-", linewidth = 2, label= T_NOGSE[i] )
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