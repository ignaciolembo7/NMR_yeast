#NMRSI - Ignacio Lembo Ferrari - 24/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import lmfit
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_20240622"
folder = "fit_nogse_vs_x_mixto"
exp = 1 #int(input('exp: '))
num_grad = input('Gradiente: ')
tnogse = float(input('T_NOGSE [ms]: ')) #ms
g = float(input('g [mT/m]: ')) #mT/m
n = 2

slic = 0 # slice que quiero ver
modelo = "Mixto"

D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # intra
D0 = D0_int

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
    model = lmfit.Model(nogse.M_nogse_mixto, independent_vars=["TE", "G", "N", "D0", "x"], param_names=["t_c", "alpha", "M0"])
    model.set_param_hint("M0", value=902.0, min = 0.0, max = 10000.0, vary = 1)
    model.set_param_hint("t_c", value=1.77, min = 0.0, max = 10.0, vary = 1)
    model.set_param_hint("alpha", value=0.0, min = 0.0, max = 1.0, vary = 0)
    params = model.make_params()

    result = model.fit(f, params, TE=float(tnogse), G=float(g), N=n, D0=D0, x=x) 

    print(result.params.pretty_print())
    print(f"Chi cuadrado = {result.chisqr}")
    print(f"Reduced chi cuadrado = {result.redchi}")

    M0_fit = result.params["M0"].value
    tc_fit = result.params["t_c"].value
    M0_error = result.params["M0"].stderr
    tc_error = result.params["t_c"].stderr
    alpha_fit = result.params["alpha"].value
    alpha_error = result.params["alpha"].stderr

    x_fit = np.linspace(np.min(x), np.max(x), num=1000)
    fit = nogse.M_nogse_mixto(float(tnogse), float(g), n, x_fit, tc_fit, alpha_fit, M0_fit, D0)

    with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
        print(roi,  " - M0_int = ", M0_fit, "+-", M0_error, file=a)
        print("    ",  " - t_c = ", tc_fit, "+-", tc_error, file=a)
        print("    ",  " - alpha = ", alpha_fit, "+-", alpha_error, file=a)

    fig1, ax1 = plt.subplots(figsize=(8,6)) 
    fig2, ax2 = plt.subplots(figsize=(8,6)) 

    nogse.plot_nogse_vs_x_fit(ax, roi, modelo, x, x_fit, f, fit, tnogse, g, n, slic, color)
    nogse.plot_nogse_vs_x_fit(ax1, roi, modelo, x, x_fit, f, fit, tnogse, g, n, slic, color)

    table = np.vstack((x_fit, fit))
    np.savetxt(f"{directory}/{roi}_fit_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

    with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_tnogse.txt", "a") as a:
        print(tnogse, tc_fit, tc_error, alpha_fit, alpha_error, M0_fit, M0_error, file=a)

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
    plt.close(fig1)

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
plt.close(fig)
