#NMRSI - Ignacio Lembo Ferrari - 16/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")
from lmfit import Minimizer, create_params, fit_report

file_name = "levaduras_20240622"
folder = "brutefit_contrast_vs_g_mixto"
A0 = "sin_A0"
D0 = 2.3e-12 # extra
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "Mixto"

tnogse = float(input('Tnogse [ms]: ')) #ms
n = 2
rois = ["ROI1"]

fig, ax = plt.subplots(figsize=(8,6)) 
rois = ["ROI1"]
palette = sns.color_palette("tab10", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/tnogse={tnogse}_N={int(n)}_exp={exp}"
os.makedirs(directory, exist_ok=True)

def fcn2min(params, g, f):
    M0 = params['M0']
    tc = params['tc']
    alpha = params['alpha']
    model = nogse.fit_contrast_vs_g_mixto(tnogse, g, n, tc, alpha, M0, D0)
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

    tc_min = 0.0
    tc_max = 19
    alpha_min = 0.1
    alpha_max = 1.0
    M0_min = 1000
    M0_max = 5000 
    steps = 50
    params = create_params( tc = dict(value=(tc_min + tc_max)/2, min=tc_min, max=tc_max, brute_step=(tc_max - tc_min)/steps, vary=False),
                            alpha = dict(value=(alpha_min + alpha_max)/2, min=alpha_min, max=alpha_max, brute_step=(alpha_max - alpha_min)/steps, vary=True),
                            M0 = dict(value=(M0_min+M0_max)/2, min=M0_min, max=M0_max, brute_step=(M0_max - M0_min)/steps, vary=True)
                            )

    fitter = Minimizer(fcn2min, params, fcn_args=(g, f))
    result_brute = fitter.minimize(method='brute', keep=1) #Ns = 25

    nogse.plot_results_brute(result_brute, best_vals=True, varlabels=None, output=f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_chi.png")

    M0_fit = result_brute.params['M0'].value
    tc_fit = result_brute.params['tc'].value
    alpha_fit = result_brute.params['alpha'].value
    error_M0 = result_brute.params['M0'].stderr
    error_tc = result_brute.params['tc'].stderr
    error_alpha = result_brute.params['alpha'].stderr

    g_fit = np.linspace(np.min(g), np.max(g), num=1000)
    fit = nogse.fit_contrast_vs_g_mixto(tnogse, g_fit, n, tc_fit, alpha_fit, M0_fit, D0)

    with open(f"{directory}/parameters_tnogse={tnogse}_N={int(n)}.txt", "a") as a:
        print(roi,  " - tc = ", tc_fit, "+-", error_tc, file=a)
        print("    ",  " - alpha = ", alpha_fit, "+-", error_alpha, file=a)
        print("    ",  " - M0 = ", M0_fit, "+-", error_M0, file=a)
        print("    ",  " - D0 = ", D0, "+-", file=a)

    fig1, ax1 = plt.subplots(figsize=(8,6)) 

    nogse.plot_contrast_vs_g_fit(ax, roi, modelo, g, g_fit, f, fit, tnogse, n, slic, color)
    nogse.plot_contrast_vs_g_fit(ax1, roi, modelo, g, g_fit, f, fit, tnogse, n, slic, color)

    table = np.vstack((g_fit, fit))
    np.savetxt(f"{directory}/{roi}_fit_contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.pdf")
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig1)

    with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_tnogse.txt", "a") as a:
        print(tnogse, tc_fit, error_tc, alpha_fit, error_alpha, M0_fit, error_M0, file=a)

fig.tight_layout()
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.pdf")
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.png", dpi=600)
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