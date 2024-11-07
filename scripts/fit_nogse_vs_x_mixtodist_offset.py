#NMRSI - Ignacio Lembo Ferrari - 24/09/2024

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

#parameters = {(tnogse, num_grad): {'lc_mode': [], 'lc_mode': [], 'lc_mode': [], 'lc_mode': [], 'lc_mode': [], 'C': [], 'C_error': [], 'chi': []} for tnogse in tnogses}

for tnogse, g in zip(tnogses, G4):

    file_name = "levaduras_20240622"
    folder = "fit_nogse_vs_x_mixtodist_sinoffset"
    exp = 1 #int(input('exp: '))
    num_grad = "G4" #input('Gradiente: ')
    #tnogse = float(input('Tnogse [ms]: ')) #ms
    #g = float(input('g [mT/m]: ')) #mT/m
    n = 2

    slic = 0 # slice que quiero ver
    modelo = "MixtoDist_offset"

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

    for roi, color in zip(rois, palette):

        fig1, ax1 = plt.subplots(figsize=(8,6)) 
        fig2, ax2 = plt.subplots(figsize=(8,6)) 

        data = np.loadtxt(f"../results_{file_name}/nogse_vs_x_data/slice={slic}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}/{roi}_data_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt")

        x = data[:, 0]
        f = data[:, 1]
        vectores_combinados = zip(x, f)
        vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
        x, f = zip(*vectores_ordenados)

        #modelo M_nogse_rest_dist
        model = lmfit.Model(nogse.fit_nogse_vs_x_mixtodistmode_offset, independent_vars=["TE", "G", "N", "x", "D0"], param_names=["lc_mode", "sigma", "alpha", "M0", "C"])
        model.set_param_hint("M0", value=100, max = 5000, vary=1)
        model.set_param_hint("lc_mode", value=1.76, min = 0.1, max = 4.0, vary=0)
        model.set_param_hint("sigma", value=0.1, min = 0.01, max=2.5, vary=0)
        model.set_param_hint("alpha", value=0, min = 0.0, max=1.0, vary=0)
        model.set_param_hint("C", value=0, min=0.0, max = 1000.0, vary=0)
        params = model.make_params()

        result = model.fit(f, params, TE=float(tnogse), G=float(g), N=n, x=x, D0=D0) #Cambiar el D0 según haga falta puede ser D0_int o D0_ext

        print(result.params.pretty_print())
        print(f"Chi cuadrado = {result.chisqr}")
        print(f"Reduced chi cuadrado = {result.redchi}")

        M0_fit = result.params["M0"].value
        lc_fit = result.params["lc_mode"].value
        sigma_fit = result.params["sigma"].value
        alpha_fit = result.params["alpha"].value
        C_fit = result.params["C"].value
        M0_error = result.params["M0"].stderr
        lc_error = result.params["lc_mode"].stderr
        sigma_error = result.params["sigma"].stderr
        alpha_error = result.params["alpha"].stderr
        C_error = result.params["C"].stderr

        x_fit = np.linspace(np.min(x), np.max(x), num=1000)
        fit = nogse.fit_nogse_vs_x_mixtodistmode_offset(float(tnogse), float(g), n, x_fit, lc_fit, sigma_fit, alpha_fit, M0_fit, D0, C_fit)

        l_c_median = lc_fit*np.exp(sigma_fit**2)
        l_c_mid = l_c_median*np.exp((sigma_fit**2)/2)

        with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
            print(roi,  " - l_c_mode = ", lc_fit, "+-", lc_error, file=a)
            print("    ",  " - l_c_median = ", l_c_median, "+-", file=a)
            print("    ",  " - l_c_mid = ", l_c_mid, "+-", file=a)
            print("    ",  " - sigma = ", sigma_fit, "+-", sigma_error, file=a)
            print("    ",  " - alpha = ", alpha_fit, "+-", alpha_error, file=a)
            print("    ",  " - M0 = ", M0_fit, "+-", M0_error, file=a)
            print("    ",  " - C = ", C_fit, "+-", C_error, file=a)
            print("    ",  " - D0 = ", D0, "+-", file=a)
        
        nogse.plot_nogse_vs_x_fit(ax, roi, modelo, x, x_fit, f, fit, tnogse, g, n, slic, color) 
        nogse.plot_nogse_vs_x_fit(ax1, roi, modelo, x, x_fit, f, fit, tnogse, g, n, slic, color)

        l_c = np.linspace(0.01, 40, 1000) #asi esta igual que en nogse.py
        dist = nogse.lognormal(l_c, sigma_fit, lc_fit)
        nogse.plot_lognorm_dist(ax2, roi, tnogse, n, l_c, lc_fit, sigma_fit, slic, color)

        table = np.vstack((x_fit, fit))
        np.savetxt(f"{directory}/{roi}_fit_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

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

        with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_tnogse_g={num_grad}.txt", "a") as a:
            print(tnogse, g, lc_fit, lc_error, sigma_fit, sigma_error, M0_fit, M0_error, file=a)

        with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_g_tnogse={tnogse}.txt", "a") as a:
            print(tnogse, g, lc_fit, lc_error, sigma_fit, sigma_error, M0_fit, M0_error, file=a)

    fig.tight_layout()
    fig.savefig(f"../results_{file_name}/{folder}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.pdf")
    fig.savefig(f"../results_{file_name}/{folder}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig)