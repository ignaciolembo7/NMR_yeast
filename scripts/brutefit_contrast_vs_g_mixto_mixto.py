#NMRSI - Ignacio Lembo Ferrari - 16/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")
from lmfit import Minimizer, create_params

file_name = "levaduras_20240613"
folder = "brutefit_contrast_vs_g_mixto_mixto"
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
    tc_1 = params['tc_1']
    alpha_1 = params['alpha_1']
    M0_2 = params['M0_2']
    tc_2 = params['tc_2']
    alpha_2 = params['alpha_2']
    model = nogse.fit_contrast_vs_g_mixto_mixto(tnogse, g, n, tc_1, alpha_1, M0_1, D0_ext, tc_2, alpha_2, M0_2, D0_int)
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
    #g = np.delete(g, [0]) #1,26,27,28,29,30,31,32,33,34,35,36,37
    #f = np.delete(f, [0])

    tc1_min = 0.1
    tc1_max = 5.0
    alpha1_min = 0.543
    alpha1_max = 0.545
    M01_min =0
    M01_max =1
    tc2_min = 1.507
    tc2_max = 1.509
    alpha2_min = 0.0167
    alpha2_max = 0.0169
    M02_min =0
    M02_max =1
    
    steps = 20

    # params = create_params( tc_1 = dict(value=(tc1_min + tc1_max)/2, min=tc1_min, max=tc1_max, brute_step=(tc1_max - tc1_min)/steps, vary=False),
    #                         alpha_1 = dict(value=(alpha1_min + alpha1_max)/2, min=alpha1_min, max=alpha1_max, brute_step=(alpha1_max - alpha1_min)/steps, vary=False),
    #                         M0_1 = dict(value=(M01_min+M01_max)/2, min=M01_min, max=M01_max, brute_step=(M01_max - M01_min)/steps, vary=False),
    #                         tc_2 = dict(value=(tc2_min + tc2_max)/2, min=tc2_min, max=tc2_max, brute_step=(tc2_max - tc2_min)/steps, vary=True),
    #                         alpha_2 = dict(value=0.0168, brute_step=(alpha2_max - alpha2_min)/steps, vary=False),
    #                         M0_2 = dict(value=(M02_min+M02_max)/2, min=M02_min, max=M02_max, brute_step=(M02_max - M02_min)/steps, vary=False)
    #                         )

    params = create_params( tc_1 = dict(value=(tc1_min + tc1_max)/2, min=tc1_min, max=tc1_max, brute_step=(tc1_max - tc1_min)/steps, vary=True),
                            alpha_1 = dict(value=0.523, brute_step=(alpha1_max - alpha1_min)/steps, vary=False),
                            M0_1 = dict(value=0.5, min=M01_min, max=M01_max, brute_step=(M01_max - M01_min)/steps, vary=True),
                            tc_2 = dict(value=2.08, brute_step=(tc2_max - tc2_min)/steps, vary=False),
                            alpha_2 = dict(value=0.0027, brute_step=(alpha2_max - alpha2_min)/steps, vary=False),
                            M0_2 = dict(value=0.4, min=M02_min, max=M02_max, brute_step=(M02_max - M02_min)/steps, vary=True)
                            )
    
    fitter = Minimizer(fcn2min, params, fcn_args=(g, f))
    result_brute = fitter.minimize(method='brute', keep=1)

    print(result_brute.params.pretty_print())
    print(f"Chi cuadrado = {result_brute.chisqr}")
    print(f"Reduced chi cuadrado = {result_brute.redchi}")

    nogse.plot_results_brute(result_brute, best_vals=True, varlabels=None, output=f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_chi.png")

    M01_fit = result_brute.params['M0_1'].value
    M01_error = result_brute.params['M0_1'].stderr
    tc1_fit = result_brute.params['tc_1'].value
    tc1_error = result_brute.params['tc_1'].stderr
    alpha1_fit = result_brute.params['alpha_1'].value
    alpha1_error = result_brute.params['alpha_1'].stderr
    M02_fit = result_brute.params['M0_2'].value
    M02_error = result_brute.params['M0_2'].stderr
    tc2_fit = result_brute.params['tc_2'].value
    tc2_error = result_brute.params['tc_2'].stderr
    alpha2_fit = result_brute.params['alpha_2'].value
    alpha2_error = result_brute.params['alpha_2'].stderr

    g_fit = np.linspace(np.min(g), np.max(g), num=1000)
    fit = nogse.fit_contrast_vs_g_mixto_mixto(tnogse, g_fit, n, tc1_fit, alpha1_fit, M01_fit, D0_ext, tc2_fit, alpha2_fit, M02_fit, D0_int)
    fit_1 = nogse.fit_contrast_vs_g_mixto(tnogse, g_fit, n, tc1_fit, alpha1_fit, M01_fit, D0_ext)
    fit_2 = nogse.fit_contrast_vs_g_mixto(tnogse, g_fit, n, tc2_fit, alpha2_fit, M02_fit, D0_int)

    with open(f"{directory}/parameters_tnogse={tnogse}_N={int(n)}.txt", "a") as a:
        print(roi,  " - tc_1 = ", tc1_fit, "+-", tc1_error, file=a)
        print("     ",  " - alpha_1 = ", alpha1_fit, "+-", alpha1_error, file=a)
        print("     ",  " - M01 = ", M01_fit, "+-", M01_error, file=a)
        print("     ",  " - D0_ext = ", D0_ext, file=a)
        print("     ",  " - tc_2 = ", tc2_fit, "+-", tc2_error, file=a)
        print("     ",  " - alpha_2 = ", alpha2_fit, "+-", alpha2_error, file=a)
        print("     ",  " - M02 = ", M02_fit, "+-", M02_error, file=a)
        print("     ",  " - D0_int = ", D0_int, file=a)
        print("     ",  " - chi_square = ", result_brute.chisqr, file=a)
        print("     ",  " - reduced_chi_square = ", result_brute.redchi, file=a)

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
        print(tnogse, tc1_fit, tc1_error, alpha1_fit, alpha1_error, M01_fit, M01_error, tc2_fit, tc2_error, alpha2_fit, alpha2_error, M02_fit, M02_error, file=a)

fig.tight_layout()
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.pdf")
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.png", dpi=600)
plt.close(fig)