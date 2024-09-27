#NMRSI - Ignacio Lembo Ferrari - 17/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")
from lmfit import Minimizer, create_params

file_name = "levaduras_20240622"
folder = "brutefit_nogse_vs_x_mixto"
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
    tc = params['tc']
    alpha = params['alpha']
    model = nogse.M_nogse_mixto(tnogse, g, n, x, tc, alpha, M0, D0_int)
    return model - f

for roi, color in zip(rois,palette):

    fig1, ax1 = plt.subplots(figsize=(8,6)) 

    data = np.loadtxt(f"../results_{file_name}/nogse_vs_x_data/slice={slic}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}/{roi}_data_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt")

    x = data[:, 0]
    f = data[:, 1]
    # Combinar los vectores usando zip()
    vectores_combinados = zip(x, f)
    # Ordenar los vectores combinados basándote en vector_g
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    # Separar los vectores nuevamente
    x, f = zip(*vectores_ordenados)

    tc1_min = 0.1
    tc1_max = 2.5
    alpha1_min = 0.0
    alpha1_max = 1.0
    M01_min = 800
    M01_max = 1000

    steps = 50

    params = create_params( tc_1 = dict(value=(tc1_min + tc1_max)/2, min=tc1_min, max=tc1_max, brute_step=(tc1_max - tc1_min)/steps, vary=True),
                            #alpha_1 = dict(value=(alpha1_min + alpha1_max)/2, min=alpha1_min, max=alpha1_max, brute_step=(alpha1_max - alpha1_min)/steps, vary=False),
                            alpha_1 = dict(value=0, brute_step=(alpha1_max - alpha1_min)/steps, vary=False),
                            M0_1 = dict(value=(M01_min+M01_max)/2, min=M01_min, max=M01_max, brute_step=(M01_max - M01_min)/steps, vary=True)
                            )

    fitter = Minimizer(fcn2min, params, fcn_args=(x, f))
    result_brute = fitter.minimize(method='brute', keep=1) 

    print(result_brute.params.pretty_print())
    print(f"Chi cuadrado = {result_brute.chisqr}")
    print(f"Reduced chi cuadrado = {result_brute.redchi}")

    nogse.plot_results_brute(result_brute, best_vals=True, varlabels=None, output=f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_chi.png")

    M0_fit = result_brute.params['M0'].value
    M0_error = result_brute.params['M0'].stderr
    tc_fit = result_brute.params['tc'].value
    tc_error = result_brute.params['tc'].stderr
    alpha_fit = result_brute.params['alpha'].value
    alpha_error = result_brute.params['alpha'].stderr

    x_fit = np.linspace( np.min(x), np.max(x), num=1000)
    fit = nogse.M_nogse_mixto(tnogse, g, n, x_fit, tc_fit, alpha_fit, M0_fit, D0)

    with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
        print(roi,  " - tc = ", tc_fit, "+-", tc_error, file=a)
        print("     ",  " - alpha = ", alpha_fit, "+-", alpha_error, file=a)
        print("     ",  " - M0 = ", M0_fit, "+-", M0_error, file=a)
        print("     ",  " - D0 = ", D0, file=a)

    nogse.plot_nogse_vs_x_fit(ax, roi, modelo, x, x_fit, f, fit, tnogse, n, g, slic, color) 
    nogse.plot_nogse_vs_x_fit(ax1, roi, modelo, x, x_fit, f, fit, tnogse, n, g, slic, color)

    table = np.vstack((x_fit, fit))
    np.savetxt(f"{directory}/{roi}_fit_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig1)

    with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_tnogse_g={num_grad}.txt", "a") as a:
        print(tnogse, g, tc_fit, tc_error, alpha_fit, alpha_error, M0_fit, M0_error, file=a) 

    with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_g_tnogse={tnogse}.txt", "a") as a:
        print(g, tnogse, tc_fit, tc_error, alpha_fit, alpha_error, M0_fit, M0_error, file=a) 

fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}.png", dpi=600)
plt.close(fig)