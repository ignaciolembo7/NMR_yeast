#NMRSI - Ignacio Lembo Ferrari - 28/05/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import lmfit
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_20240613"
folder = "contrast_vs_g_mixto_mixto"
A0 = "con_A0"
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # 0.7e-12 # intra
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "Mixto+Mixto"  # nombre carpeta modelo libre/rest/tort

tnogse = float(input('Tnogse [ms]: ')) #ms
n = 2
rois = ["ROI1"]

fig, ax = plt.subplots(figsize=(8,6)) 
palette = sns.color_palette("tab10", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/{A0}/tnogse={tnogse}_N={int(n)}_exp={exp}"
os.makedirs(directory, exist_ok=True)

for roi, color in zip(rois,palette):
    
    fig1, ax1 = plt.subplots(figsize=(8,6)) 
    fig2, ax2 = plt.subplots(figsize=(8,6)) 

    data = np.loadtxt(f"../results_{file_name}/contrast_vs_g_data/{A0}/tnogse={tnogse}_N={int(n)}_exp={exp}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={int(n)}.txt")

    g = data[:, 0]
    f = data[:, 1]
    # Combinar los vectores usando zip()
    vectores_combinados = zip(g, f)
    # Ordenar los vectores combinados basándote en vector_g
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    # Separar los vectores nuevamente
    g, f = zip(*vectores_ordenados)

    #remover los elementos en la posicion _ de x y f 
    g = np.delete(g, [0])
    f = np.delete(f, [0])

    #modelo contrast_vs_g_intrarest_extrarestdist
    model = lmfit.Model(nogse.fit_contrast_vs_g_mixto_mixto, independent_vars=["TE", "G", "N", "D0_1", "D0_2"], param_names=["tc_1", "alpha_1", "M0_1", "tc_2", "alpha_2", "M0_2"])
    model.set_param_hint("M0_1", value=0.80, min=0, max = 1.0, vary = 1)
    #model.set_param_hint("M0_2", expr="1 - M0_1")
    model.set_param_hint("M0_2", value=0.2, min=0, max = 1.0, vary = 1)
    model.set_param_hint("tc_1", value=5.1, min = 0, max = 20.0, vary = 0)
    model.set_param_hint("tc_2", value=2.0, min = 0, max = 10.0, vary = 0)
    model.set_param_hint("alpha_1", value=0.30, min = 0, max=1, vary = 0)
    model.set_param_hint("alpha_2", value=0.03, min = 0, max=1, vary = 0)
    params = model.make_params()

    result = model.fit(f, params, TE=float(tnogse), G=g, N=n, D0_1= D0_ext, D0_2= D0_int) 

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
    
    g_fit = np.linspace(np.min(g), np.max(g), num=1000)
    fit = nogse.fit_contrast_vs_g_mixto_mixto(tnogse, g_fit, n, tc_1_fit, alpha_1_fit, M01_fit, D0_ext, tc_2_fit, alpha_2_fit, M02_fit, D0_int)
    fit_1 = nogse.fit_contrast_vs_g_mixto(tnogse, g_fit, n, tc_1_fit, alpha_1_fit, M01_fit, D0_ext)
    fit_2 = nogse.fit_contrast_vs_g_mixto(tnogse, g_fit, n, tc_2_fit, alpha_2_fit, M02_fit, D0_int)

    with open(f"{directory}/parameters_tnogse={tnogse}_N={int(n)}.txt", "a") as a:
        print(roi,  " - tc_1 = ", tc_1_fit, "+-", error_tc_1, file=a)
        print("     ",  " - alpha_1 = ", alpha_1_fit, "+-", error_alpha_1, file=a)
        print("     ",  " - M01 = ", M01_fit, "+-", error_M01, file=a)
        print("     ",  " - D0_ext = ", D0_ext, file=a)
        print("     ",  " - tc_2 = ", tc_2_fit, "+-", error_tc_2, file=a)
        print("     ",  " - alpha_2 = ", alpha_2_fit, "+-", error_alpha_2, file=a)
        print("     ",  " - M02 = ", M02_fit, "+-", error_M02, file=a)
        print("     ",  " - D0_int = ", D0_int, file=a)

    nogse.plot_contrast_vs_g_fit(ax, roi, modelo, g, g_fit, f, fit, tnogse, n, slic, color)

    nogse.plot_contrast_vs_g_fit(ax1, roi, modelo, g, g_fit, f, fit, tnogse, n, slic, color)
    nogse.plot_contrast_vs_g_fit(ax1, roi, modelo, g, g_fit, f, fit_1, tnogse, n, slic, color)
    nogse.plot_contrast_vs_g_fit(ax1, roi, modelo, g, g_fit, f, fit_2, tnogse, n, slic, color)
    ax1.fill_between(g_fit, 0, fit_1, color=color, alpha=0.2)
    ax1.fill_between(g_fit, 0, fit_2, color=color, alpha=0.2)

    table = np.vstack((g_fit, fit))
    np.savetxt(f"{directory}/{roi}_ajuste_nogse_vs_x_tnogse={tnogse}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.pdf")
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig1)

    with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_tnogse.txt", "a") as a:
        print(tnogse, tc_1_fit, error_tc_1, alpha_1_fit, error_alpha_1, M01_fit, error_M01, tc_2_fit, error_tc_2, alpha_2_fit, error_alpha_2, M02_fit, error_M02, file=a)

fig.tight_layout()
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.pdf")
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.png", dpi=600)
plt.close(fig)