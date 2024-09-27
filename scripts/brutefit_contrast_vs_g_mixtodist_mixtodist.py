#NMRSI - Ignacio Lembo Ferrari - 23/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")
from lmfit import Minimizer, create_params

file_name = "levaduras_20240613"
folder = "brutefit_contrast_vs_g_mixtodist_mixtodist"
A0 = "con_A0"
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "Mixto+Mixto"  # nombre carpeta modelo libre/rest/tort

tnogse = float(input('Tnogse [ms]: ')) #ms
n = 2
rois = ["ROI1"]

fig, ax = plt.subplots(figsize=(8,6)) 
rois = ["ROI1"]
palette = sns.color_palette("tab10", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/{A0}/tnogse={tnogse}_N={int(n)}_exp={exp}"
os.makedirs(directory, exist_ok=True)

def fcn2min(params, g, f):
    M0_1 = params['M0_1']
    lc_1 = params['lc_1']
    sigma_1 = params['sigma_1']
    alpha_1 = params['alpha_1']
    M0_2 = params['M0_2']
    lc_2 = params['lc_2']
    sigma_2 = params['sigma_2']
    alpha_2 = params['alpha_2']
    model = nogse.fit_contrast_vs_g_mixtodist_mixtodist(tnogse, g, n, lc_1, sigma_1, alpha_1, M0_1, D0_ext, lc_2, sigma_2, alpha_2, M0_2, D0_int)
    return model - f

for roi, color in zip(rois,palette):

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
    g = np.delete(g, [0]) #1,26,27,28,29,30,31,32,33,34,35,36,37
    f = np.delete(f, [0])

    lc1_min = 1
    lc1_max = 20.0
    sigma1_min = 0.1
    sigma1_max = 1.0
    alpha1_min = 0.0
    alpha1_max = 1.0
    M01_min = 0
    M01_max = 1

    lc2_min = 0
    lc2_max = 2
    sigma2_min = 0.0001
    sigma2_max = 0.1
    alpha2_min = 0.0
    alpha2_max = 0.2
    M02_min = 0.29
    M02_max = 0.31
    tc_2 = 2.1

    steps = 10

    params = create_params( lc_1 = dict(value=(lc1_min + lc1_max)/2, min=lc1_min, max=lc1_max, brute_step=(lc1_max - lc1_min)/steps, vary=True),
                            sigma_1 = dict(value=(sigma1_min + sigma1_max)/2, min=sigma1_min, max=sigma1_max, brute_step=(sigma1_max - sigma1_min)/steps, vary=True),
                            alpha_1 = dict(value=(alpha1_min + alpha1_max)/2, min=alpha1_min, max=alpha1_max, brute_step=(alpha1_max - alpha1_min)/steps, vary=True),
                            M0_1 = dict(value=(M01_min+M01_max)/2, min=M01_min, max=M01_max, brute_step=(M01_max - M01_min)/steps, vary=True),
                            lc_2 = dict(value=np.sqrt(2*(D0_int*1e12)*tc_2), brute_step=(lc2_max - lc2_min)/steps, vary=False),
                            sigma_2 = dict(value=0.001, brute_step=(sigma2_max - sigma2_min)/steps, vary=False),
                            alpha_2 = dict(value=0.0, brute_step=(alpha2_max - alpha2_min)/steps, vary=False),
                            M0_2 = dict(value=(M02_min+M02_max)/2, min=M02_min, max=M02_max, brute_step=(M02_max - M02_min)/steps, vary=False)
                            )

    fitter = Minimizer(fcn2min, params, fcn_args=(g, f))
    result_brute = fitter.minimize(method='brute', keep=1)

    nogse.plot_results_brute(result_brute, best_vals=True, varlabels=None, output=f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_chi.png")

    print(result_brute.params.pretty_print())
    print(f"Chi cuadrado = {result_brute.chisqr}")
    print(f"Reduced chi cuadrado = {result_brute.redchi}")

    M01_fit = result_brute.params['M0_1'].value
    M01_error = result_brute.params['M0_1'].stderr
    lc1_fit = result_brute.params['lc_1'].value
    lc1_error = result_brute.params['lc_1'].stderr
    sigma1_fit = result_brute.params['sigma_1'].value
    sigma1_error = result_brute.params['sigma_1'].stderr
    alpha1_fit = result_brute.params['alpha_1'].value
    alpha1_error = result_brute.params['alpha_1'].stderr
    M02_fit = result_brute.params['M0_2'].value
    M02_error = result_brute.params['M0_2'].stderr
    lc2_fit = result_brute.params['lc_2'].value
    lc2_error = result_brute.params['lc_2'].stderr
    sigma2_fit = result_brute.params['sigma_2'].value
    sigma2_error = result_brute.params['sigma_2'].stderr
    alpha2_fit = result_brute.params['alpha_2'].value
    alpha2_error = result_brute.params['alpha_2'].stderr

    g_fit = np.linspace(np.min(g), np.max(g), num=1000)
    fit = nogse.fit_contrast_vs_g_mixtodist_mixtodist(tnogse, g_fit, n, lc1_fit, sigma1_fit, alpha1_fit, M01_fit, D0_ext, lc2_fit, sigma2_fit, alpha2_fit, M02_fit, D0_int)
    fit_1 = nogse.fit_contrast_vs_g_mixtodist(tnogse, g_fit, n, lc1_fit, sigma1_fit, alpha1_fit, M01_fit, D0_ext)
    fit_2 = nogse.fit_contrast_vs_g_mixtodist(tnogse, g_fit, n, lc2_fit, sigma2_fit, alpha2_fit, M02_fit, D0_int)

    with open(f"{directory}/parameters_tnogse={tnogse}_N={int(n)}.txt", "a") as a:
        print(roi,  " - lc_1 = ", lc1_fit, "+-", lc1_error, file=a)
        print("     ",  " - sigma_1 = ", sigma1_fit, "+-", sigma1_error, file=a)
        print("     ",  " - alpha_1 = ", alpha1_fit, "+-", alpha1_error, file=a)
        print("     ",  " - M01 = ", M01_fit, "+-", M01_error, file=a)
        print("     ",  " - D0_ext = ", D0_ext, file=a)
        print("     ",  " - lc_2 = ", lc2_fit, "+-", lc2_error, file=a)
        print("     ",  " - sigma_2 = ", sigma2_fit, "+-", sigma2_error, file=a)
        print("     ",  " - alpha_2 = ", alpha2_fit, "+-", alpha2_error, file=a)
        print("     ",  " - M02 = ", M02_fit, "+-", M02_error, file=a)
        print("     ",  " - D0_int = ", D0_int, file=a)

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

    with open(f"../results_{file_name}/{folder}/{A0}/{roi}_parameters_vs_tnogse.txt", "a") as a:
        print(tnogse, lc1_fit, lc1_error, sigma1_fit, sigma1_error, alpha1_fit, alpha1_error, M01_fit, M01_error, lc2_fit, lc2_error, sigma2_fit, sigma2_error, alpha2_fit, alpha2_error, M02_fit, M02_error, file=a)

    fig2, ax2 = plt.subplots(figsize=(8,6)) 

    lc1_median = lc1_fit*np.exp(sigma1_fit**2)
    lc1_mid = lc1_median*np.exp((sigma1_fit**2)/2)
    lc2_median = lc2_fit*np.exp(sigma2_fit**2)
    lc2_mid = lc1_median*np.exp((sigma2_fit**2)/2)
    lc = np.linspace(0.01, 40, 1000) #asi esta igual que en nogse.py
    dist1 = nogse.lognormal(lc, sigma1_fit, lc1_fit)
    nogse.plot_lognorm_dist(ax2, "Extracelular", tnogse, n, lc, lc1_fit, sigma1_fit, slic, color = "orange")
    dist2 = nogse.lognormal(lc, sigma2_fit, lc2_fit)
    #nogse.plot_lognorm_dist(ax2, "Intracelular", tnogse, n, lc, lc2_fit, sigma2_fit, slic, color = "green")

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