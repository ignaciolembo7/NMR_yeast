#NMRSI - Ignacio Lembo Ferrari - 26/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
from lmfit import Model, Parameters, minimize
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_20240613"
folder = "globalfit_M0_tc_alpha_contrast_vs_g_free_rest"
A0 = "sin_A0"
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # intra
n = 2
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "Free+Rest"
roi = "ROI1"

tnogses = [17.5,21.5,25.0,27.5,30.0,35.0,40.0]

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/{A0}"
os.makedirs(directory, exist_ok=True)

# Cargo la data para cada curva
gs = []
fs = []

for tnogse in tnogses:
        
    data = np.loadtxt(f"../results_{file_name}/plot_contrast_vs_g_data/{A0}/tnogse={tnogse}_N={int(n)}_exp={exp}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={int(n)}.txt")
    g = data[:, 0]
    f = data[:, 1] #/data[0,1]
    vectores_combinados = zip(g, f)
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    g, f = zip(*vectores_ordenados)

    # Convertir de nuevo a arrays de NumPy para la indexación
    g = np.array(g)
    f = np.array(f)
    # Filtrar los valores donde f sea positivo o cero
    mask = f >= 0
    g = g[mask]
    f = f[mask]

    gs.append(g)
    fs.append(f)

# Parámetros genéricos
params = Parameters()
#for i in range(len(xs)):
params.add(f'alpha1', value=0.4, min=0.1, max=1.0, vary = 1)
params.add(f'tc2', value=2.2, min=1.0, max=15.0, vary = 1)
params.add(f'alpha2', value=0.0, min=0.0, max=1.0, vary = 0)
params.add('M01', value=2000, min=0, max=10000, vary=1)
params.add('M02', value=1500, min=0, max=10000, vary=1)
#params.add('M02', expr="1 - M01")
#params.add(f'D0_1', value=D0_ext, min = D0_int, max = D0_ext, vary=False)
#params.add(f'D0_2', value=D0_int, min = D0_int, max = D0_ext, vary=False)

def objective_function(params, x_list=gs, fs_list=fs):
    residuals = []
    for i, (g, fs_data) in enumerate(zip(x_list, fs_list)):
        alpha1_fit = params[f'alpha1'].value
        M01_fit = params[f'M01'].value
        tc2_fit = params[f'tc2'].value
        alpha2_fit = params[f'alpha2'].value
        M02_fit = params[f'M02'].value
        model = nogse.fit_contrast_vs_g_free_mixto(tnogses[i], gs[i], n, alpha1_fit, M01_fit, D0_ext, tc2_fit, alpha2_fit, M02_fit, D0_int)

        fs_data = np.array(fs_data)

        if np.isnan(model).any() or np.isnan(fs_data).any():
            raise ValueError("NaN detected in model or fs values")
                          
        residuals.append(fs_data - model)
    
    return np.concatenate(residuals)

result = minimize(objective_function, params)

# Display fitting results
print(result.params.pretty_print())
print(f"Chi cuadrado = {result.chisqr}")
print(f"Reduced chi cuadrado = {result.redchi}")

tnogses = [15.0,17.5,21.5,25.0,27.5,30.0,35.0,40.0]

palette = sns.color_palette("tab10")

fig, ax = plt.subplots(figsize=(8, 6))

for tnogse, color in zip(tnogses, palette):

    data = np.loadtxt(f"../results_{file_name}/plot_contrast_vs_g_data/{A0}/tnogse={tnogse}_N={int(n)}_exp={exp}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={int(n)}.txt")
    g = data[:, 0]
    f = data[:, 1] #/data[0,1]
    vectores_combinados = zip(g, f)
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    g, f = zip(*vectores_ordenados)

   # Convertir de nuevo a arrays de NumPy para la indexación
    g = np.array(g)
    f = np.array(f)

    # Filtrar los valores donde f sea positivo o cero
    mask = f >= 0
    g = g[mask]
    f = f[mask]

    fig1, ax1 = plt.subplots(figsize=(8,6)) 
    fig2, ax2 = plt.subplots(figsize=(8,6)) 

    M01_fit = result.params[f'M01'].value
    M01_error = result.params[f'M01'].stderr
    M02_fit = result.params[f'M02'].value
    M02_error = result.params[f'M02'].stderr
    alpha1_fit = result.params[f'alpha1'].value
    alpha1_error = result.params[f'alpha1'].stderr
    tc2_fit = result.params[f'tc2'].value
    tc2_error = result.params[f'tc2'].stderr
    alpha2_fit = result.params[f'alpha2'].value
    alpha2_error = result.params[f'alpha2'].stderr

    g_fit = np.linspace(0, np.max(g), num=1000)
    fit = nogse.fit_contrast_vs_g_free_mixto(tnogse, g_fit, n, alpha1_fit, M01_fit, D0_ext, tc2_fit, alpha2_fit, M02_fit, D0_int)
    fit_1 = nogse.fit_contrast_vs_g_free(tnogse, g_fit, n, alpha1_fit, M01_fit, D0_ext)
    fit_2 = nogse.fit_contrast_vs_g_mixto(tnogse, g_fit, n, tc2_fit, alpha2_fit, M02_fit, D0_int)

    with open(f"{directory}/parameters_N={int(n)}.txt", "a") as a:
        print("roi",  " - alpha_1 = ", alpha1_fit, "+-", alpha1_error, file=a)
        print("     ",  " - M01 = ", M01_fit, "+-", M01_error, file=a)
        print("     ",  " - D0_1 = ", D0_ext, file=a)
        print("     ",  " - tc_2 = ", tc2_fit, "+-", tc2_error, file=a)
        print("     ",  " - alpha_2 = ", alpha2_fit, "+-", alpha2_error, file=a)
        print("     ",  " - M02 = ", M02_fit, "+-", M02_error, file=a)
        print("     ",  " - D0_2 = ", D0_int, file=a)
        print("     ",  " - Chi cuadrado = ", result.chisqr, file=a)
        print("     ",  " - Reduced chi cuadrado = ", result.redchi, file=a)
        
    label = f"{tnogse} ms"
    nogse.plot_contrast_vs_g_fit(ax, label, modelo, g, g_fit, f, fit, tnogse, n, slic, color)

    nogse.plot_contrast_vs_g_fit(ax1, "Extracelular", modelo, g, g_fit, f, fit_1, tnogse, n, slic, color = "tab:orange")
    nogse.plot_contrast_vs_g_fit(ax1, "Intracelular", modelo, g, g_fit, f, fit_2, tnogse, n, slic, color = "tab:green")
    nogse.plot_contrast_vs_g_fit(ax1, roi, modelo, g, g_fit, f, fit, tnogse, n, slic, color = "tab:blue")

    ax1.fill_between(g_fit, 0, fit_1, color = "tab:orange", alpha=0.2)
    ax1.fill_between(g_fit, 0, fit_2, color = "tab:green", alpha=0.2)

    table = np.vstack((g_fit, fit))
    np.savetxt(f"{directory}/{roi}_fit_contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"../results_{file_name}/{folder}/{A0}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.pdf")
    fig1.savefig(f"../results_{file_name}/{folder}/{A0}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig1)

with open(f"../results_{file_name}/{folder}/{A0}/{roi}_parameters_vs_tnogse.txt", "a") as a:
    print(alpha1_fit, alpha1_error, M01_fit, M01_error, tc2_fit, tc2_error, alpha2_fit, alpha2_error, M02_fit, M02_error, file=a)

fig.tight_layout()
fig.savefig(f"../results_{file_name}/{folder}/{A0}/contrast_vs_g_N={int(n)}_exp={exp}.pdf")
fig.savefig(f"../results_{file_name}/{folder}/{A0}/contrast_vs_g_N={int(n)}_exp={exp}.png", dpi=600)
plt.close(fig)