#NMRSI - Ignacio Lembo Ferrari - 31/07/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import lmfit
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_20240622"
folder = "rest_vs_x_restdist_mode"

exp = 1 #int(input('exp: '))
num_grad = input('Gradiente: ')
tnogse = float(input('T_NOGSE [ms]: ')) #ms
g = float(input('g [mT/m]: ')) #mT/m
n = 2


slic = 0 # slice que quiero ver
modelo = "restdist_mode"  # nombre carpeta modelo libre/rest/tort

D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # intra
D0 = D0_ext

fig, ax = plt.subplots(figsize=(8,6)) 

rois = ["ROI1"]
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
    x = np.delete(x, [10,11])
    f = np.delete(f, [10,11])

    #modelo M_nogse_rest_dist
    model = lmfit.Model(nogse.M_nogse_rest_dist, independent_vars=["TE", "G", "N", "x", "D0"], param_names=["l_c_mode", "sigma", "M0"])
    model.set_param_hint("M0", value=438.1867644995955)
    model.set_param_hint("l_c_mode", value=9.96897654357775, min = 0.1, max = 100)
    model.set_param_hint("sigma", value=0.1, min = 0.01, max=10.0)
    params = model.make_params()
    #params["M0"].vary = False # fijo M0 en 1, los datos estan normalizados y no quiero que varíe
    params["l_c_mode"].vary = 1
    params["M0"].vary = 1
    params["sigma"].vary = 1

    result = model.fit(f, params, TE=float(tnogse), G=float(g), N=n, x=x, D0=D0) #Cambiar el D0 según haga falta puede ser D0_int o D0_ext
    M0_fit = result.params["M0"].value
    l_c_fit = result.params["l_c_mode"].value
    sigma_fit = result.params["sigma"].value
    error_M0 = result.params["M0"].stderr
    error_l_c = result.params["l_c_mode"].stderr
    error_sigma = result.params["sigma"].stderr

    x_fit = np.linspace( np.min(x), np.max(x), num=1000)
    fit = nogse.M_nogse_rest_dist(float(tnogse), float(g), n, x_fit, l_c_fit, sigma_fit, M0_fit, D0)

    l_c_median = l_c_fit*np.exp(sigma_fit**2)
    l_c_mid = l_c_median*np.exp((sigma_fit**2)/2)

    with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
        print(roi,  " - M0 = ", M0_fit, "+-", error_M0, file=a)
        print(roi,  " - l_c_mode = ", l_c_fit, "+-", error_l_c, file=a)
        print(roi,  " - l_c_median = ", l_c_median, "+-", file=a)
        print(roi,  " - l_c_mid = ", l_c_mid, "+-", file=a)
        print(roi,  " - sigma = ", sigma_fit, "+-", error_sigma, file=a)
        print(roi,  " - D0 = ", D0, "+-", file=a)
    
    nogse.plot_nogse_vs_x_restdist(ax, roi, modelo, x, x_fit, f, fit, tnogse, n, g, l_c_fit, slic, color) 
    nogse.plot_nogse_vs_x_restdist(ax1, roi, modelo, x, x_fit, f, fit, tnogse, n, g, l_c_fit, slic, color)

    l_c = np.linspace(0.01, 40, 1000) #asi esta igual que en nogse.py
    dist = nogse.lognormal(l_c, sigma_fit, l_c_fit)
    nogse.plot_lognorm_dist(ax2, roi, tnogse, n, l_c, l_c_fit, sigma_fit, slic, color)

    table = np.vstack((x_fit, fit))
    np.savetxt(f"{directory}/{roi}_ajuste_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig1)

    table = np.vstack((l_c, dist))
    np.savetxt(f"{directory}/{roi}_dist_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig2.tight_layout()
    fig2.savefig(f"{directory}/{roi}_dist_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.pdf")
    fig2.savefig(f"{directory}/{roi}_dist_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig2)

    with open(f"../results_{file_name}/{folder}/parameters_vs_tnogse_{num_grad}.txt", "a") as a:
        print(tnogse, l_c_fit, error_l_c, sigma_fit, error_sigma, M0_fit, error_M0, file=a)

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.png", dpi=600)
plt.close(fig)