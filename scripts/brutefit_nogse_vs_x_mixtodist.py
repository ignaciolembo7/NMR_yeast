#NMRSI - Ignacio Lembo Ferrari - 24/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")
from lmfit import Minimizer, create_params, fit_report

file_name = "levaduras_20240622"
folder = "brutefit_nogse_vs_x_mixtodist"
A0 = "sin_A0"
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # 0.7e-12 # intra
D0 = D0_int
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "Mixto"

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

def fcn2min(params, x, f):
    M0 = params['M0']
    lc_mode = params['lc_mode']
    sigma = params['sigma']
    alpha = params['alpha']
    model = nogse.fit_nogse_vs_x_mixtodistmode(tnogse, g, n, x, lc_mode, sigma, alpha, M0, D0)
    return model - f

for roi, color in zip(rois,palette):

    data = np.loadtxt(f"../results_{file_name}/nogse_vs_x_data/slice={slic}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}/{roi}_data_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt")

    x = data[:, 0]
    f = data[:, 1]
    # Combinar los vectores usando zip()
    vectores_combinados = zip(x, f)
    # Ordenar los vectores combinados basándote en vector_g
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    # Separar los vectores nuevamente
    x, f = zip(*vectores_ordenados)

    lc_min = 0.1
    lc_max = 2.5
    sigma_min = 0.1
    sigma_max = 2.5
    alpha_min = 0.0
    alpha_max = 1.0
    M0_min = 800
    M0_max = 1000

    steps = 10

    params = create_params( lc_mode = dict(value=(lc_min + lc_max)/2, min=lc_min, max=lc_max, brute_step=(lc_max - lc_min)/steps, vary=True),
                            sigma = dict(value=(sigma_min + sigma_max)/2, min=sigma_min, max=sigma_max, brute_step=(sigma_max - sigma_min)/steps, vary=True),
                            #alpha = dict(value=(alpha1_min + alpha1_max)/2, min=alpha1_min, max=alpha1_max, brute_step=(alpha1_max - alpha1_min)/steps, vary=False),
                            alpha = dict(value=0, brute_step=(alpha_max - alpha_min)/steps, vary=False),
                            M0 = dict(value=(M0_min+M0_max)/2, min=M0_min, max=M0_max, brute_step=(M0_max - M0_min)/steps, vary=True)
                            )

    fitter = Minimizer(fcn2min, params, fcn_args=(x, f))
    result_brute = fitter.minimize(method='brute', keep=1)

    print(result_brute.params.pretty_print())
    print(f"Chi cuadrado = {result_brute.chisqr}")
    print(f"Reduced chi cuadrado = {result_brute.redchi}")

    nogse.plot_results_brute(result_brute, best_vals=True, varlabels=None, output=f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_chi.png")

    M0_fit = result_brute.params['M0'].value
    lc_mode_fit = result_brute.params['lc_mode'].value
    sigma_fit = result_brute.params['sigma'].value
    alpha_fit = result_brute.params['alpha'].value
    M0_error = result_brute.params['M0'].stderr
    lc_mode_error = result_brute.params['lc_mode'].stderr
    sigma_error = result_brute.params['sigma'].stderr
    alpha_error = result_brute.params['alpha'].stderr

    x_fit = np.linspace(np.min(x), np.max(x), num=1000)
    fit = nogse.fit_nogse_vs_x_mixtodistmode(tnogse, g, n, x_fit, lc_mode_fit, sigma_fit, alpha_fit, M0_fit, D0)

    l_c_median = lc_mode_fit*np.exp(sigma_fit**2)
    l_c_mid = l_c_median*np.exp((sigma_fit**2)/2)

    with open(f"{directory}/parameters_tnogse={tnogse}_N={int(n)}.txt", "a") as a:
        print(roi,  " - l_c_mode = ", lc_mode_fit, "+-", lc_mode_error, file=a)
        print("    ",  " - l_c_median = ", l_c_median, "+-", file=a)
        print("    ",  " - l_c_mid = ", l_c_mid, "+-", file=a)
        print("    ",  " - sigma = ", sigma_fit, "+-", sigma_error, file=a)
        print("    ",  " - alpha = ", alpha_fit, "+-", alpha_error, file=a)
        print("    ",  " - M0 = ", M0_fit, "+-", M0_error, file=a)
        print("    ",  " - D0 = ", D0, "+-", file=a)

    fig1, ax1 = plt.subplots(figsize=(8,6)) 
    fig2, ax2 = plt.subplots(figsize=(8,6)) 

    nogse.plot_nogse_vs_x_fit(ax, roi, modelo, x, x_fit, f, fit, tnogse, n, g, slic, color)
    nogse.plot_nogse_vs_x_fit(ax1, roi, modelo, x, x_fit, f, fit, tnogse, n, g, slic, color)

    table = np.vstack((x_fit, fit))
    np.savetxt(f"{directory}/{roi}_fit_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
    plt.close(fig1)

    lc = np.linspace(0.001, 30, 1000) #asi esta igual que en nogse.py
    dist = nogse.lognormal(lc, sigma_fit, lc_mode_fit)
    nogse.plot_lognorm_dist(ax2, roi, tnogse, n, lc, lc_mode_fit, sigma_fit, slic, color)
    
    table = np.vstack((lc, dist))
    np.savetxt(f"{directory}/{roi}_dist_vs_lc_tnogse={tnogse}_N={int(n)}.txt", table.T, delimiter=' ', newline='\n')
    
    fig2.tight_layout()
    fig2.savefig(f"{directory}/{roi}_dist_vs_lc_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
    fig2.savefig(f"{directory}/{roi}_dist_vs_lc_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
    plt.close(fig2)

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
plt.close(fig)

"""
best_result = copy.deepcopy(result_brute)
for candidate in result_brute.candidates:
    trial = fitter.minimize(method='leastsq', params=candidate.params)
    if trial.chisqr < best_result.chisqr:
        best_result = trial
M0_int_fit = best_result.params['M0_int'].value
M0_ext_fit = best_result.params['M0_ext'].value
t_c_int_fit = best_result.params['t_c_int'].value
t_c_ext_fit = best_result.params['t_c_ext'].value
alpha_fit = best_result.params['alpha'].value
error_t_c_int = best_result.params['t_c_int'].stderr
error_t_c_ext = best_result.params['t_c_ext'].stderr
error_alpha = best_result.params['alpha'].stderr
""" 