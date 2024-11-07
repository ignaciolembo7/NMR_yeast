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
# gs = [100.0, 105.0, 75.0, 60.0, 55.0, 50.0, 45.0, 40.0, 35.0, 30.0]
# gs = [275.0, 210.0, 160.0, 120.0, 110.0, 100.0, 90.0, 80.0, 75.0, 70.0]
gs = [600.0, 405.0, 300.0, 210.0, 190.0, 170.0, 150.0, 130.0, 120.0, 110.0]
# gs = [1000.0, 800.0, 700.0, 600.0, 550.0, 500.0, 450.0, 400.0, 375.0, 350.0]

tc_ints = [1.995, 1.999, 2.102, 2.134, 2.030, 2.193, 2.089, 2.279, 2.167, 2.241]
M0_ints = [1573.33, 1416.96, 1109.74, 714.98, 407.07, 398.53, 198.05, 225.09, 123.76, 135.28]

file_name = "levaduras_20240622"
folder = "fit_nogse_vs_x_free_rest_G3"
A0 = "sin_A0"
D0_ext = 2.3e-12 # extra
D0_int = 0.65e-12 # 0.7e-12 # intra
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "Free+Rest"

num_grad = input('Gradiente: ')
#tnogse = float(input('Tnogse [ms]: ')) #ms
#g = float(input('g [mT/m]: ')) #mT/m
n = 2
rois = ["ROI1"]

for tnogse, g, tc_int, M0_int in zip(tnogses, gs, tc_ints, M0_ints):

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
        model = lmfit.Model(nogse.fit_nogse_vs_x_free_rest, independent_vars=["TE", "G", "N", "x", "D0_1"], param_names=["alpha_1", "M0_1", "tc_2", "M0_2", "D0_2"])
        model.set_param_hint("M0_1", value=2000, min=0, max = 5000, vary = 1)
        model.set_param_hint("M0_2", value=M0_int, min=0, max = 5000, vary = 1)
        #model.set_param_hint("M0_2", expr="0.42857*M0_1") 
        #model.set_param_hint("M0_2", expr="(9/11)*M0_1")
        model.set_param_hint("tc_2", value=tc_int, min = 0.1, max = 10.0, vary = 0)
        model.set_param_hint("alpha_1", value=0.5, min = 0, max = 1, vary = 1)
        model.set_param_hint("D0_2", value=D0_int, min=0, max = 1e-11, vary = 0)
        params = model.make_params()

        result = model.fit(f, params, TE=float(tnogse), G=float(g), N=n, x=x, D0_1 = D0_ext)

        print("Tnogse =", tnogse)
        print(result.params.pretty_print())
        print(f"Chi cuadrado = {result.chisqr}")
        print(f"Reduced chi cuadrado = {result.redchi}")

        M01_fit = result.params["M0_1"].value
        M01_error = result.params["M0_1"].stderr
        alpha1_fit = result.params["alpha_1"].value
        alpha1_error = result.params["alpha_1"].stderr
        M02_fit = result.params["M0_2"].value
        M02_error = result.params["M0_2"].stderr
        tc2_fit = result.params["tc_2"].value
        tc2_error = result.params["tc_2"].stderr
        D02_fit = result.params["D0_2"].value
        D02_error = result.params["D0_2"].stderr

        fig1, ax1 = plt.subplots(figsize=(8,6)) 
        fig2, ax2 = plt.subplots(figsize=(8,6)) 

        x_fit = np.linspace(np.min(x), np.max(x), num=1000)
        fit = nogse.fit_nogse_vs_x_free_rest(tnogse, g, n, x_fit, alpha1_fit, M01_fit, D0_ext, tc2_fit, M02_fit, D0_int)
        fit_1 = nogse.M_nogse_free(tnogse, g, n, x_fit, M01_fit, alpha1_fit*D0_ext)
        fit_2 = nogse.M_nogse_rest(tnogse, g, n, x_fit, tc2_fit, M02_fit, D0_int)

        nogse.plot_nogse_vs_x_fit(ax2, "fit ext", modelo, x, x_fit, f, fit_1, tnogse, g, n, slic, color = 'orange')
        nogse.plot_nogse_vs_x_fit(ax2, "fit int", modelo, x, x_fit, f, fit_2, tnogse, g, n, slic, color = 'green')
        nogse.plot_nogse_vs_x_fit(ax2, num_grad, modelo, x, x_fit, f, fit, tnogse, g, n, slic, color)
        nogse.plot_nogse_vs_x_fit(ax, num_grad, modelo, x, x_fit, f, fit, tnogse, g, n, slic, color) 
        nogse.plot_nogse_vs_x_fit(ax1, num_grad, modelo, x, x_fit, f, fit, tnogse, g, n, slic, color)

        table = np.vstack((x_fit, fit))
        np.savetxt(f"{directory}/{roi}_fit_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

        fig1.tight_layout()
        fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.pdf")
        fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.png", dpi=600)
        plt.close(fig1)
        
        fig2.tight_layout()
        fig2.savefig(f"{directory}/{roi}_nogse_vs_x_intext_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.pdf")
        fig2.savefig(f"{directory}/{roi}_nogse_vs_x_intext_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.png", dpi=600)
        plt.close(fig2)

    with open(f"{directory}/parameters_tnogse={tnogse}_N={int(n)}.txt", "a") as a:
        print("roi",  " - alpha_1 = ", alpha1_fit, "+-", alpha1_error, file=a)
        print("     ",  " - M01 = ", M01_fit, "+-", M01_error, file=a)
        print("     ",  " - D0_ext = ", D0_ext, file=a)
        print("     ",  " - tc_2 = ", tc2_fit, "+-", tc2_error, file=a)
        print("     ",  " - M02 = ", M02_fit, "+-", M02_error, file=a)
        print("     ",  " - D0_int = ", D0_int, file=a)
        print("     ",  " - Chi cuadrado = ", result.chisqr, file=a)
        print("     ",  " - Reduced chi cuadrado = ", result.redchi, file=a)

    with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_tnogse_G={num_grad}.txt", "a") as a:
        print(tnogse, alpha1_fit, alpha1_error, M01_fit, M01_error, tc2_fit, tc2_error, M02_fit, M02_error, file=a) 

    fig.tight_layout()
    fig.savefig(f"../results_{file_name}/{folder}/nogse_vs_x_tnogse={tnogse}_N={int(n)}_exp={exp}.pdf")
    fig.savefig(f"../results_{file_name}/{folder}/nogse_vs_x_tnogse={tnogse}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig)