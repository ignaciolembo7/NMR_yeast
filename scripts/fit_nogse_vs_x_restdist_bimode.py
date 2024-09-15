#NMRSI - Ignacio Lembo Ferrari - 05/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import lmfit
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_20240622"
folder = "nogse_vs_x_restdist_bimode"
A0 = "sin_A0"
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # 0.7e-12 # intra
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "Rest Dist Bimode"  # nombre carpeta modelo libre/rest/tort

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

    #modelo M_nogse_restdist def M_nogse_restdist_bimode(TE, G, N, x, lc_mode_1, sigma_1, lc_mode_2, sigma_2, M0_1, M0_2, D0_1, D0_2):

    model = lmfit.Model(nogse.M_nogse_restdist_bimode, independent_vars=["TE", "G", "N", "x", "D0_1", "D0_2"], param_names=["lc_mode_1", "lc_mode_2", "sigma_1", "sigma_2", "M0_1", "M0_2"])
    model.set_param_hint("M0_1", value=3691, min=0, max = 10000, vary = 1)
    model.set_param_hint("M0_2", value=892, min=0, max = 10000, vary = 1)
    model.set_param_hint("lc_mode_1", value=5.48, min = 1.0, max = 10.0, vary = 1)
    model.set_param_hint("lc_mode_2", value=2.0, min = 1.0, max = 10.0, vary = 1)
    model.set_param_hint("sigma_1", value=0.96, min = 0.1, max=2.0, vary = 0)
    model.set_param_hint("sigma_2", value=0.34, min = 0.1, max=2.0, vary = 0)
    params = model.make_params()

    result = model.fit(f, params, TE=float(tnogse), G=float(g), N=n, x=x, D0_1=D0_ext, D0_2=D0_int)

    M01_fit = result.params["M0_1"].value
    error_M01 = result.params["M0_1"].stderr
    M02_fit = result.params["M0_2"].value
    error_M02 = result.params["M0_2"].stderr
    lc_mode_1_fit = result.params["lc_mode_1"].value
    error_lc_mode_1 = result.params["lc_mode_1"].stderr
    lc_mode_2_fit = result.params["lc_mode_2"].value
    error_lc_mode_2 = result.params["lc_mode_2"].stderr
    sigma_1_fit = result.params["sigma_1"].value
    error_sigma_1 = result.params["sigma_1"].stderr
    sigma_2_fit = result.params["sigma_2"].value
    error_sigma_2 = result.params["sigma_2"].stderr
    
    x_fit = np.linspace( np.min(x), np.max(x), num=1000)
    fit = nogse.M_nogse_restdistmode_restdistmode(tnogse, g, n, x_fit, lc_mode_1_fit, sigma_1_fit, lc_mode_2_fit, sigma_2_fit, M01_fit, M02_fit, D0_ext, D0_int)

    lc_median_1 = lc_mode_1_fit*np.exp(sigma_1_fit**2)
    lc_mid_1 = lc_median_1*np.exp((sigma_1_fit**2)/2)
    lc_median_2 = lc_mode_2_fit*np.exp(sigma_2_fit**2)
    lc_mid_2 = lc_median_1*np.exp((sigma_2_fit**2)/2)
    
    with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
        print(roi,  " - lc_mode_1 = ", lc_mode_1_fit, "+-", error_lc_mode_1, file=a)
        print("    ",  " - lc_median_1 = ", lc_median_1, "+-", file=a)
        print("    ",  " - lc_mid_1 = ", lc_mid_1, "+-", file=a)
        print("    ",  " - sigma_1 = ", sigma_1_fit, "+-", error_sigma_1, file=a)
        print("    ",  " - M0_1 = ", M01_fit, "+-", error_M01, file=a)
        print("    ",  " - D0_ext = ", D0_ext, "+-", file=a)
        print("    ",  " - lc_mode_2 = ", lc_mode_2_fit, "+-", error_lc_mode_2, file=a)
        print("    ",  " - lc_median_2 = ", lc_median_2, "+-", file=a)
        print("    ",  " - lc_mid_2 = ", lc_mid_2, "+-", file=a)
        print("    ",  " - sigma_2 = ", sigma_2_fit, "+-", error_sigma_2, file=a)
        print("    ",  " - M0_2 = ", M02_fit, "+-", error_M02, file=a)
        print("    ",  " - D0_int = ", D0_int, "+-", file=a)

    nogse.plot_nogse_vs_x_fit(ax, roi, modelo, x, x_fit, f, fit, tnogse, n, g, slic, color) 
    nogse.plot_nogse_vs_x_fit(ax1, roi, modelo, x, x_fit, f, fit, tnogse, n, g, slic, color)

    lc = np.linspace(0.01, 40, 1000) #asi esta igual que en nogse.py
    dist1 = nogse.lognormal(lc, sigma_1_fit, lc_mode_1_fit)
    nogse.plot_lognorm_dist(ax2, roi, tnogse, n, lc, lc_mode_1_fit, sigma_1_fit, slic, color)
    dist2 = nogse.lognormal(lc, sigma_2_fit, lc_mode_2_fit)
    nogse.plot_lognorm_dist(ax2, roi, tnogse, n, lc, lc_mode_2_fit, sigma_2_fit, slic, color)

    table = np.vstack((x_fit, fit))
    np.savetxt(f"{directory}/{roi}_ajuste_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig1)

    table = np.vstack((lc, dist1, dist2))
    np.savetxt(f"{directory}/{roi}_dist_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig2.tight_layout()
    fig2.savefig(f"{directory}/{roi}_dist_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.pdf")
    fig2.savefig(f"{directory}/{roi}_dist_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig2)

    with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_tnogse_g={num_grad}.txt", "a") as a:
        print(tnogse, g, lc_mode_1_fit, error_lc_mode_1, sigma_1_fit, error_sigma_1, M01_fit, error_M01, lc_mode_2_fit, error_lc_mode_2, sigma_2_fit, error_sigma_2, M02_fit, error_M02, file=a)

    with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_g_tnogse={tnogse}.txt", "a") as a:
        print(g, tnogse, lc_mode_1_fit, error_lc_mode_1, sigma_1_fit, error_sigma_1, M01_fit, error_M01, lc_mode_2_fit, error_lc_mode_2, sigma_2_fit, error_sigma_2, M02_fit, error_M02, file=a)

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.png", dpi=600)
plt.close(fig)