#NMRSI - Ignacio Lembo Ferrari - 31/07/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import lmfit
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

tnogses = [15.0, 17.5, 21.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0]
# G1 = [100.0, 105.0, 75.0, 60.0, 55.0, 50.0, 45.0, 40.0, 35.0, 30.0]
# G2 = [275.0, 210.0, 160.0, 120.0, 110.0, 100.0, 90.0, 80.0, 75.0, 70.0]
# G3 = [600.0, 405.0, 300.0, 210.0, 190.0, 170.0, 150.0, 130.0, 120.0, 110.0]
G4 = [1000.0, 800.0, 700.0, 600.0, 550.0, 500.0, 450.0, 400.0, 375.0, 350.0]

# tnogses = [21.5]
# G4 = [600.0]

file_name = "levaduras_20240622"
folder = "fit_nogse_vs_x_rest_offset"
exp = 1 #int(input('exp: '))
num_grad = input('Gradiente: ')
A0_folder = "sin_A0"
#tnogse = float(input('Tnogse [ms]: ')) #ms
#g = float(input('g [mT/m]: ')) #mT/m
n = 2
slic = 0 # slice que quiero ver
modelo = "Rest-offset"  # nombre carpeta modelo libre/rest/tort
D0_ext = 2.3e-12 # extra
D0_int = 0.65e-12 # intra

for tnogse, g in zip(tnogses, G4):

    fig, ax = plt.subplots(figsize=(8,6)) 
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
        model = lmfit.Model(nogse.M_nogse_rest_offset, independent_vars=["TE", "G", "N", "x"], param_names=["t_c", "M0", "D0", "C"])
        model.set_param_hint("M0", value=1000.0, min = 0.0, max = 10000.0, vary = 1)
        model.set_param_hint("t_c", value=2.2, min = 0.1, max = 1000.0, vary = 1)
        model.set_param_hint("D0", value=0.65e-12, min = 0.0, max = D0_ext, vary = 0)
        model.set_param_hint("C", value=100, min = 0, vary=1)   
        params = model.make_params()

        result = model.fit(f, params, TE=float(tnogse), G=float(g), N=n, x=x)

        print(result.params.pretty_print())
        print(f"Chi cuadrado = {result.chisqr}")
        print(f"Reduced chi cuadrado = {result.redchi}")

        M0_fit = result.params["M0"].value
        tc_fit = result.params["t_c"].value
        M0_error = result.params["M0"].stderr
        tc_error = result.params["t_c"].stderr
        D0_fit = result.params["D0"].value
        D0_error = result.params["D0"].stderr
        C_fit = result.params["C"].value
        C_error = result.params["C"].stderr

        x_fit = np.linspace(np.min(x), np.max(x), num=1000)
        fit = nogse.M_nogse_rest_offset(float(tnogse), float(g), n, x_fit, tc_fit, M0_fit, D0_fit, C_fit)

        with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
            print(roi,  " - M0_int = ", M0_fit, "+-", M0_error, file=a)
            print("    ",  " - tc = ", tc_fit, "+-", tc_error, file=a)
            print("    ",  " - D0 = ", D0_fit, "+-", D0_error, file=a)
            print("    ",  " - C = ", C_fit, "+-", C_error, file=a)
            print("    ",  " - Chi cuadrado = ", result.chisqr, file=a)
            print("    ",  " - Reduced chi cuadrado = ", result.redchi, file=a)
        

        fig1, ax1 = plt.subplots(figsize=(8,6)) 

        nogse.plot_nogse_vs_x_fit(ax, roi, modelo, x, x_fit, f, fit, tnogse, g, n, slic, color)
        nogse.plot_nogse_vs_x_fit(ax1, roi, modelo, x, x_fit, f, fit, tnogse, g, n, slic, color)

        table = np.vstack((x_fit, fit))
        np.savetxt(f"{directory}/{roi}_fit_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

        with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_tnogse_G={num_grad}.txt", "a") as a:
            print(tnogse, g, tc_fit, tc_error, M0_fit, M0_error, D0_fit, D0_error, C_fit, C_error, file=a)
        
        with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_g_tnogse={tnogse}.txt", "a") as a:
            print(g, tnogse, tc_fit, tc_error, M0_fit, M0_error, D0_fit, D0_error, C_fit, C_error, file=a)
        
        fig1.tight_layout()
        fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
        fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
        plt.close(fig1)

    fig.tight_layout()
    fig.savefig(f"../results_{file_name}/{folder}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
    fig.savefig(f"../results_{file_name}/{folder}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
    plt.close(fig)
