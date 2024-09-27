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
folder = "fit_contrast_vs_g_mixtodist_mixtodist"
A0 = "con_A0"
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # 0.7e-12 # intra
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "Mixto+Mixto" 

tnogse = float(input('Tnogse [ms]: ')) #ms
n = 2
rois = ["ROI1"]

fig, ax = plt.subplots(figsize=(8,6)) 
palette = sns.color_palette("tab10", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/tnogse={tnogse}_n={int(n)}_exp={exp}"
os.makedirs(directory, exist_ok=True)

for roi, color in zip(rois,palette):

    data = np.loadtxt(f"../results_{file_name}/contrast_vs_g_data/{A0}/tnogse={tnogse}_n={int(n)}_exp={exp}/{roi}_data_contrast_vs_g_tnogse={tnogse}_n={int(n)}.txt")

    g = data[:, 0]
    f = data[:, 1]
    # Combinar los vectores usando zip()
    vectores_combinados = zip(g, f)
    # Ordenar los vectores combinados basándote en vector_g
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    # Separar los vectores nuevamente
    g, f = zip(*vectores_ordenados)

    #modelo M_nogse_rest_dist
    model = lmfit.Model(nogse.fit_contrast_vs_g_mixtodist_mixtodist, independent_vars=["TE", "G", "N", "D0_1", "D0_2"], param_names=["lc_mode_1","sigma_1","alpha_1","M0_1","lc_mode_2","alpha_2","sigma_2","M0_2"])
    model.set_param_hint("M0_1", value=0.6, min=0, max = 1.0, vary = 1)
    #model.set_param_hint("M0_2", expr="1 - M0_1")
    model.set_param_hint("M0_2", value=0.4, min=0, max = 1.0, vary = 1)
    model.set_param_hint("lc_mode_1", value=0.5, min = 0.5, max = 15.0, vary = 1)
    model.set_param_hint("lc_mode_2", value=2.0, min = 0.5, max = 10.0, vary = 1)
    model.set_param_hint("sigma_1", value=1.0, min = 0, max=2.0, vary = 1)
    model.set_param_hint("sigma_2", value=1.0, min = 0, max=2.0, vary = 1)
    model.set_param_hint("alpha_1", value=0.43, min = 0, max=1, vary = 1)
    model.set_param_hint("alpha_2", value=0.0, min = 0, max=1, vary = 0)
    params = model.make_params()

    result = model.fit(f, params, TE=float(tnogse), G=g, N=n, D0_1=D0_ext, D0_2=D0_int) #Cambiar el D0 según haga falta puede ser D0_int o D0_ext

    print(result.params.pretty_print())
    print(f"Chi cuadrado = {result.chisqr}")
    print(f"Reduced chi cuadrado = {result.redchi}")

    M01_fit = result.params['M0_1'].value
    M01_error = result.params['M0_1'].stderr
    lc1_fit = result.params['lc_mode_1'].value
    lc1_error = result.params['lc_mode_1'].stderr
    sigma1_fit = result.params['sigma_1'].value
    sigma1_error = result.params['sigma_1'].stderr
    alpha1_fit = result.params['alpha_1'].value
    alpha1_error = result.params['alpha_1'].stderr
    M02_fit = result.params['M0_2'].value
    M02_error = result.params['M0_2'].stderr
    lc2_fit = result.params['lc_mode_2'].value
    lc2_error = result.params['lc_mode_2'].stderr
    sigma2_fit = result.params['sigma_2'].value
    sigma2_error = result.params['sigma_2'].stderr
    alpha2_fit = result.params['alpha_2'].value
    alpha2_error = result.params['alpha_2'].stderr

    g_fit = np.linspace(np.min(g), np.max(g), num=1000)
    fit = nogse.fit_contrast_vs_g_mixtodist_mixtodist(tnogse, g_fit, n, lc1_fit, sigma1_fit, alpha1_fit, M01_fit, D0_ext, lc2_fit, sigma2_fit, alpha2_fit, M02_fit, D0_int)
    fit_1 = nogse.fit_contrast_vs_g_mixtodist(tnogse, g_fit, n, lc1_fit, sigma1_fit, alpha1_fit, M01_fit, D0_ext)
    fit_2 = nogse.fit_contrast_vs_g_mixtodist(tnogse, g_fit, n, lc2_fit, sigma2_fit, alpha2_fit, M02_fit, D0_int)

    fig1, ax1 = plt.subplots(figsize=(8,6)) 

    nogse.plot_contrast_vs_g_fit(ax, roi, modelo, g, g_fit, f, fit, tnogse, n, slic, color)
    nogse.plot_contrast_vs_g_fit(ax1, "Extracelular", modelo, g, g_fit, f, fit_1, tnogse, n, slic, "orange")
    nogse.plot_contrast_vs_g_fit(ax1, "Intracelular", modelo, g, g_fit, f, fit_2, tnogse, n, slic, "green")
    ax1.fill_between(g_fit, 0, fit_1, color="orange", alpha=0.2)
    ax1.fill_between(g_fit, 0, fit_2, color="green", alpha=0.2)
    nogse.plot_contrast_vs_g_fit(ax1, roi, modelo, g, g_fit, f, fit, tnogse, n, slic, color)

    table = np.vstack((g_fit, fit))
    np.savetxt(f"{directory}/{roi}_fit_contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.pdf")
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig1)

    with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_tnogse.txt", "a") as a:
        print(tnogse, lc1_fit, lc1_error, sigma1_fit, sigma1_error, alpha1_fit, alpha1_error, M01_fit, M01_error, lc2_fit, lc2_error, sigma2_fit, sigma2_error, alpha2_fit, alpha2_error, M02_fit, M02_error, file=a)

    fig2, ax2 = plt.subplots(figsize=(8,6)) 

    lc1_median = lc1_fit*np.exp(sigma1_fit**2)
    lc1_mid = lc1_median*np.exp((sigma1_fit**2)/2)
    lc2_median = lc2_fit*np.exp(sigma2_fit**2)
    lc2_mid = lc1_median*np.exp((sigma2_fit**2)/2)
    lc = np.linspace(0.01, 30, 1000) #asi esta igual que en nogse.py
    dist1 = nogse.lognormal(lc, sigma1_fit, lc1_fit)
    nogse.plot_lognorm_dist(ax2, "Extracelular", tnogse, n, lc, lc1_fit, sigma1_fit, slic, color = "orange")
    dist2 = nogse.lognormal(lc, sigma2_fit, lc2_fit)
    nogse.plot_lognorm_dist(ax2, "Intracelular", tnogse, n, lc, lc2_fit, sigma2_fit, slic, color = "green")

    with open(f"{directory}/parameters_tnogse={tnogse}_N={int(n)}.txt", "a") as a:
        print(roi,  " - lc_mode_1 = ", lc1_fit, "+-", lc1_error, file=a)
        print("    ",  " - lc_median_1 = ", lc1_median, "+-", file=a)
        print("    ",  " - lc_mid_1 = ", lc1_mid, "+-", file=a)
        print("    ",  " - sigma_1 = ", sigma1_fit, "+-", sigma1_error, file=a)
        print("    ",  " - alpha_1 = ", alpha1_fit, "+-", alpha1_error, file=a)
        print("    ",  " - M0_1 = ", M01_fit, "+-", M01_error, file=a)
        print("    ",  " - D0_1 = ", D0_ext, "+-", file=a)
        print("    ",  " - lc_mode_2 = ", lc2_fit, "+-", lc2_error, file=a)
        print("    ",  " - lc_median_2 = ", lc2_median, "+-", file=a)
        print("    ",  " - lc_mid_2 = ", lc2_mid, "+-", file=a)
        print("    ",  " - sigma_2 = ", sigma2_fit, "+-", sigma2_error, file=a)
        print("    ",  " - alpha_2 = ", alpha2_fit, "+-", alpha2_error, file=a)
        print("    ",  " - M0_2 = ", M02_fit, "+-", M02_error, file=a)
        print("    ",  " - D0_2 = ", D0_int, "+-", file=a)

    table = np.vstack((lc, dist1, dist2))
    np.savetxt(f"{directory}/{roi}_dist_tnogse={tnogse}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig2.tight_layout()
    fig2.savefig(f"{directory}/{roi}_dist_tnogse={tnogse}_N={int(n)}_exp={exp}.pdf")
    fig2.savefig(f"{directory}/{roi}_dist_tnogse={tnogse}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig2)

fig.tight_layout()
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.pdf")
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.png", dpi=600)
plt.close(fig)