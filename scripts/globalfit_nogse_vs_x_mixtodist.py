#NMRSI - Ignacio Lembo Ferrari - 09/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
from lmfit import Model, Parameters, minimize
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_20240622"
folder = "nogse_vs_x_mixtodist_globalfit"
A0 = "sin_A0"
D0_folder = "D0_ext"
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # intra
D0 = 1.35e-12 # fitted
n = 2
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "Mixto Dist"  # nombre carpeta modelo libre/rest/tort

num_grads = ["G2", "G3"]  # Añadir los gradientes que desees
tnogses = ["21.5", "21.5"]  # Añadir los tnogses correspondientes
gs = ["160.0", "300.0"]  # Añadir las intensidades de gradiente correspondientes
rois = ["ROI1", "ROI1"]

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/{D0_folder}/tnogse={tnogses[0]}_g={num_grads}_curves_N={int(n)}_exp={exp}"
os.makedirs(directory, exist_ok=True)

# Cargo la data para cada curva
xs = []
fs = []

for roi, tnogse, g, num_grad in zip(rois, tnogses, gs, num_grads):
    data = np.loadtxt(f"../results_{file_name}/nogse_vs_x_data/slice={slic}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}/{roi}_data_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt")

    x = data[:, 0]
    f = data[:, 1]

    vectores_combinados = zip(x, f)
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    x, f = zip(*vectores_ordenados)
    
    xs.append(x)
    fs.append(f)

# Parámetros genéricos
params = Parameters()
for i in range(len(xs)):
    #params.add(f'tc{i+1}', value=8.0, min=1.0, max=15.0, vary = True)
    #params.add(f'alpha{i+1}', value=0.5, min=0.1, max=1.0, vary = True)
    #params.add(f'D0{i+1}', value=D0_ext, min = D0_int, max = D0_ext, vary=False)
    params.add(f'sigma{i+1}', value=1.4, min=0.1, max=2.0, vary=True)
params.add('M0', value=2900, vary=True)
params.add('lc_mode', value=8.0, min=3.0, max=25.0, vary = True)
params.add(f'alpha', value=0.3, min=0.01, max=1.0, vary = False)
params.add(f'D0', value=D0_ext, min = D0_int, max = D0_ext, vary=False)

def objective_function(params, x_list=xs, fs_list=fs):
    residuals = []
    for i, (x, fs_data) in enumerate(zip(x_list, fs_list)):
        sigma = params[f'sigma{i+1}']
        lc_mode = params['lc_mode']
        alpha = params['alpha']
        M0 = params['M0']
        D0 = params['D0']
        model = nogse.fit_nogse_vs_x_mixtodistmode(float(tnogses[i]), float(gs[i]), n, x, lc_mode, alpha, sigma, M0, D0)
        fs_data = np.array(fs_data)

        if np.isnan(model).any() or np.isnan(fs_data).any():
            raise ValueError("NaN detected in model or fs values")
                          
        residuals.append(fs_data - model)
    
    return np.concatenate(residuals)

result = minimize(objective_function, params)

# Display fitting results
print(result.params.pretty_print())

#alpha_fits = []
#error_alphas = []
#tc_fits = []
#error_tcs = []
#D0_fits = []
#error_D0s = []
sigma_fits = []
error_sigmas = []

M0_fit = result.params["M0"].value
error_M0 = result.params["M0"].stderr
lc_mode_fit = result.params["lc_mode"].value
error_lc_mode = result.params["lc_mode"].stderr
alpha_fit = result.params["alpha"].value
error_alpha = result.params["alpha"].stderr
D0_fit = result.params["D0"].value
error_D0 = result.params["D0"].stderr

for i in range(len(xs)):
    sigma_fits.append(result.params[f'sigma{i+1}'].value)
    error_sigmas.append(result.params[f'sigma{i+1}'].stderr)
    #tc_fits.append(result.params[f'tc{i+1}'].value)
    #alpha_fits.append(result.params[f'alpha{i+1}'].value)
    #error_tcs.append(result.params[f'tc{i+1}'].stderr)
    #error_alphas.append(result.params[f'alpha{i+1}'].stderr)
    #D0_fits.append(result.params[f"D0{i+1}"].value)
    #error_D0s.append(result.params[f"D0{i+1}"].stderr)

x_fit = np.linspace(np.min(xs[0]), np.max(xs[0]), num=1000)
fits = [nogse.fit_nogse_vs_x_mixtodistmode(float(tnogses[i]), float(gs[i]), n, x_fit, lc_mode_fit, alpha_fit, sigma_fits[i], M0_fit, D0_fit) for i in range(len(xs))]

palette = sns.color_palette("tab10", len(rois))

fig, ax = plt.subplots(figsize=(8,6)) 
fig1, ax1 = plt.subplots(figsize=(8,6)) 

for i, (f, fit, num_grad, g, color) in enumerate(zip(fs, fits, num_grads, gs, palette)):
    nogse.plot_nogse_vs_x_fit_ptG(ax, rois[0], modelo, xs[i], x_fit, f, fit, tnogses[i], n, slic, color, label=f"{num_grad} = {g}") 

fig.tight_layout()
fig.savefig(f"{directory}/{rois[0]}_nogse_vs_x_tnogse={tnogses[0]}_N={int(n)}_exp={exp}.pdf")
fig.savefig(f"{directory}/{rois[0]}_nogse_vs_x_tnogse={tnogses[0]}_N={int(n)}_exp={exp}.png", dpi=600)
plt.close(fig)

with open(f"{directory}/parameters_tnogse={tnogses[0]}_N={int(n)}.txt", "a") as a:
    print(f"{rois[0]} - tc = {lc_mode_fit} +- {error_lc_mode}", file=a)
    print(f"    - sigma = {alpha_fit} +- {error_alpha}", file=a)
    print(f"    - D0 = {D0_fit} +- {error_D0}", file=a)
    print(f"    - M0 = {M0_fit} +- {error_M0}", file=a)
    for i in range(len(xs)):
        print(f"    - sigma_{i+1} = {sigma_fits[i]} +- {error_sigmas[i]}", file=a)

table = np.vstack([x_fit] + fits)
np.savetxt(f"{directory}/{rois[0]}_globalfit_nogse_vs_x_tnogse={tnogses[0]}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

for roi, tnogse, g, num_grad, sigma_fit, error_sigma in zip(rois, tnogses, gs, num_grads, sigma_fits, error_sigmas):
    with open(f"../results_{file_name}/{folder}/{D0_folder}/{roi}_parameters_vs_tnogse_g={num_grad}.txt", "a") as a:
        print(tnogse, g, lc_mode_fit, error_lc_mode, alpha_fit, error_alpha, sigma_fit, error_sigma, M0_fit, error_M0, D0_fit, error_D0, file=a)

for roi, tnogse, g, num_grad, sigma_fit, error_sigma in zip(rois, tnogses, gs, num_grads, sigma_fits, error_sigmas):
    with open(f"../results_{file_name}/{folder}/{D0_folder}/{roi}_parameters_vs_g_tnogse={tnogse}.txt", "a") as a:    
        print(g, tnogse, lc_mode_fit, error_lc_mode, alpha_fit, error_alpha, sigma_fit, error_sigma, M0_fit, error_M0, D0_fit, error_D0, file=a)