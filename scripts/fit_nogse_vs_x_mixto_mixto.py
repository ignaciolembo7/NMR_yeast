#NMRSI - Ignacio Lembo Ferrari - 08/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import lmfit
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_20240622"
folder = "nogse_vs_x_mixto_mixto"
A0 = "sin_A0"
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # 0.7e-12 # intra
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "Mixto+Mixto"  # nombre carpeta modelo libre/rest/tort

num_grad = input('Gradiente: ')
tnogse = float(input('Tnogse [ms]: ')) #ms
g = float(input('g [mT/m]: ')) #mT/m
n = 2
rois = ["ROI1"]

fig, ax = plt.subplots(figsize=(8,6)) 
palette = sns.color_palette("tab10", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}"
os.makedirs(directory, exist_ok=True)

for roi, color in zip(rois, palette):

    fig1, ax1 = plt.subplots(figsize=(8,6)) 
    fig2, ax2 = plt.subplots(figsize=(8,6)) 

    data = np.loadtxt(f"../results_{file_name}/nogse_vs_x_data/slice={slic}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}/{roi}_data_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt")

    x = data[:, 0]
    f = data[:, 1]
    vectores_combinados = zip(x, f)
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    x, f = zip(*vectores_ordenados)

    #remover los elementos en la posicion _ de x y f 
    #x = np.delete(x, [8,9,10,11,12])
    #f = np.delete(f, [8,9,10,11,12])

    model = lmfit.Model(nogse.M_nogse_mixto_mixto, independent_vars=["TE", "G", "N", "x", "D0_1", "D0_2"], param_names=["tc_1", "alpha_1", "M0_1", "tc_2", "alpha_2", "M0_2"])
    model.set_param_hint("M0_1", value=2776, min=0, max = 10000, vary = 1)
    model.set_param_hint("M0_2", value=932, min=0, max = 10000, vary = 1)
    model.set_param_hint("tc_1", value=5.732, min = 0, max = 20.0, vary = 1)
    model.set_param_hint("tc_2", value=2.175, min = 0, max = 10.0, vary = 1)
    model.set_param_hint("alpha_1", value=0.3, min = 0, max = 1, vary = 0)
    model.set_param_hint("alpha_2", value=0.0, min = 0, max = 1, vary = 0)
    params = model.make_params()

    result = model.fit(f, params, TE=float(tnogse), G=float(g), N=n, x=x, D0_1=D0_ext, D0_2=D0_int)

    print(result.params.pretty_print())

    M01_fit = result.params["M0_1"].value
    error_M01 = result.params["M0_1"].stderr
    tc_1_fit = result.params["tc_1"].value
    error_tc_1 = result.params["tc_1"].stderr
    alpha_1_fit = result.params["alpha_1"].value
    error_alpha_1 = result.params["alpha_1"].stderr
    M02_fit = result.params["M0_2"].value
    error_M02 = result.params["M0_2"].stderr
    tc_2_fit = result.params["tc_2"].value
    error_tc_2 = result.params["tc_2"].stderr
    alpha_2_fit = result.params["alpha_2"].value
    error_alpha_2 = result.params["alpha_2"].stderr
    
    x_fit = np.linspace( np.min(x), np.max(x), num=1000)
    fit = nogse.M_nogse_mixto_mixto(tnogse, g, n, x_fit, tc_1_fit, alpha_1_fit, M01_fit, D0_ext, tc_2_fit, alpha_2_fit,  M02_fit, D0_int)
    fit_1 = nogse.M_nogse_mixto(tnogse, g, n, x_fit, tc_1_fit, alpha_1_fit, M01_fit, D0_ext)
    fit_2 = nogse.M_nogse_mixto(tnogse, g, n, x_fit, tc_2_fit, alpha_2_fit, M02_fit, D0_int)

    with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
        print(roi,  " - tc_1 = ", tc_1_fit, "+-", error_tc_1, file=a)
        print("     ",  " - alpha_1 = ", alpha_1_fit, "+-", error_alpha_1, file=a)
        print("     ",  " - M01 = ", M01_fit, "+-", error_M01, file=a)
        print("     ",  " - D0_ext = ", D0_ext, file=a)
        print("     ",  " - tc_2 = ", tc_2_fit, "+-", error_tc_2, file=a)
        print("     ",  " - alpha_2 = ", alpha_2_fit, "+-", error_alpha_2, file=a)
        print("     ",  " - M02 = ", M02_fit, "+-", error_M02, file=a)
        print("     ",  " - D0_int = ", D0_int, file=a)

    nogse.plot_nogse_vs_x_fit(ax, roi, modelo, x, x_fit, f, fit, tnogse, n, g, slic, color) 
    nogse.plot_nogse_vs_x_fit(ax1, roi, modelo, x, x_fit, f, fit, tnogse, n, g, slic, color)
    nogse.plot_nogse_vs_x_fit(ax1, "fit ext", modelo, x, x_fit, f, fit_1, tnogse, n, g, slic, color = 'orange')
    nogse.plot_nogse_vs_x_fit(ax1, "fit int", modelo, x, x_fit, f, fit_2, tnogse, n, g, slic, color = 'green')

    table = np.vstack((x_fit, fit))
    np.savetxt(f"{directory}/{roi}_ajuste_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig1)

    with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_tnogse_g={num_grad}.txt", "a") as a:
        print(tnogse, g, tc_1_fit, error_tc_1, alpha_1_fit, error_alpha_1, M01_fit, error_M01, tc_2_fit, error_tc_2, alpha_2_fit, error_alpha_2, M02_fit, error_M02, file=a)

    with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_g_tnogse={tnogse}.txt", "a") as a:
        print(g, tnogse, tc_1_fit, error_tc_1, alpha_1_fit, error_alpha_1, M01_fit, error_M01, tc_2_fit, error_tc_2, alpha_2_fit, error_alpha_2, M02_fit, error_M02, file=a)

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.png", dpi=600)
plt.close(fig)