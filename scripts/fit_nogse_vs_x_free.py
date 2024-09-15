#NMRSI - Ignacio Lembo Ferrari - 06/08/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import lmfit
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_20240622"
folder = "rest_vs_x_free"
num_grad = "G1"

exp = 1 #int(input('exp: '))
tnogse = float(input('T_NOGSE [ms]: ')) #ms
g = float(input('g [mT/m]: ')) #mT/m
n = 2

slic = 0 # slice que quiero ver
modelo = "free"  # nombre carpeta modelo libre/rest/tort

D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # intra
#D0 = D0_int

fig, ax = plt.subplots(figsize=(8,6)) 
fig3, ax3 = plt.subplots(figsize=(8,6)) 
rois = ["ROI1"]
palette = sns.color_palette("tab10", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}"
os.makedirs(directory, exist_ok=True)

for roi, color in zip(rois,palette):

    data = np.loadtxt(f"../results_{file_name}/nogse_vs_x_data/slice={slic}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}/{roi}_data_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt")

    x = data[:, 0]
    f = data[:, 1]
    vectores_combinados = zip(x, f)
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    x, f = zip(*vectores_ordenados)

    #remover los elementos en la posicion 18 y 19 de x y f
    #x = x[:18] + x[20:]
    #f = f[:18] + f[20:]

    #modelo M_nogse_rest_dist
    model = lmfit.Model(nogse.M_nogse_free, independent_vars=["TE", "G", "N", "x"], param_names=["M0", "D0"])
    model.set_param_hint("M0", value=3000.0)
    model.set_param_hint("D0", value=2e-12)
    params = model.make_params()
    params["M0"].vary = 1
    params["D0"].vary = 1

    result = model.fit(f, params, TE=float(tnogse), G=float(g), N=n, x=x) # Cambiar el D0 según haga falta puede ser D0_int o D0_ext
    M0_fit = result.params["M0"].value
    D0_fit = result.params["D0"].value
    error_M0 = result.params["M0"].stderr
    error_D0 = result.params["D0"].stderr

    x_fit = np.linspace(np.min(x), np.max(x), num=1000)
    fit = nogse.M_nogse_free(float(tnogse), float(g), n, x_fit, M0_fit, D0_fit)

    with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
        print(roi,  " - M0_int = ", M0_fit, "+-", error_M0, file=a)
        print("    ",  " - D0 = ", D0_fit, "+-", error_D0, file=a)

    fig1, ax1 = plt.subplots(figsize=(8,6)) 
    fig2, ax2 = plt.subplots(figsize=(8,6)) 

    nogse.plot_nogse_vs_x_free(ax, roi, modelo, x, x_fit, f, fit, tnogse, n, g, D0_fit, slic, color)
    nogse.plot_nogse_vs_x_free(ax1, roi, modelo, x, x_fit, f, fit, tnogse, n, g, D0_fit, slic, color)

    table = np.vstack((x_fit, fit))
    np.savetxt(f"{directory}/{roi}_adjust_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

    with open(f"../results_{file_name}/{folder}/tauc_vs_tnogse_{num_grad}.txt", "a") as a:
        print(tnogse,D0_fit,error_D0,M0_fit,error_M0, file=a)

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
    plt.close(fig1)

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
plt.close(fig)
