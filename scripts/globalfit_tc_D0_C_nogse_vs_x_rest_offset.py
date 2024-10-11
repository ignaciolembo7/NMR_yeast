#NMRSI - Ignacio Lembo Ferrari - 02/09/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
from lmfit import Model, Parameters, minimize
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

file_name = "levaduras_20240622"
folder = "globalfit_tc_nogse_vs_x_rest_offset"
A0 = "sin_A0"
D0_ext = 2.3e-12 # extra
# D0_int = 0.4e-12 # intra
n = 2
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
roi = "ROI1"
modelo = "Rest-Offset"  # nombre carpeta modelo libre/rest/tort

num_grad = "G4"
tnogses = [15.0, 17.5, 21.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0]
# G1 = [100.0, 105.0, 75.0, 60.0, 55.0, 50.0, 45.0, 40.0, 35.0, 30.0]
# G2 = [275.0, 210.0, 160.0, 120.0, 110.0, 100.0, 90.0, 80.0, 75.0, 70.0]
# G3 = [600.0, 405.0, 300.0, 210.0, 190.0, 170.0, 150.0, 130.0, 120.0, 110.0]
gs = [1000.0, 800.0, 700.0, 600.0, 550.0, 500.0, 450.0, 400.0, 375.0, 350.0]

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}"
os.makedirs(directory, exist_ok=True)

# Cargo la data para cada curva
xs = []
fs = []

for tnogse, g in zip(tnogses, gs):
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
    params.add(f'M0{i+1}', value=1500, min=0, max = 5000, vary = 1)
    # params.add(f'lc{i+1}', value=8.0, min=1.0, max=15.0, vary = True)
    # params.add(f'sigma{i+1}', value=1.5, min=0.1, max=5.0, vary = True)
    # params.add(f'D0{i+1}', value=D0_ext, min = D0_int, max = D0_ext, vary=False)
params.add(f'tc', value=1.75, min=0.1, max=5.0, vary = 1)
params.add(f'D0', value=0.65e-12, min = 0, max = D0_ext, vary = 0)
params.add(f'C', value=100, min = 0, vary=1)

def objective_function(params, x_list=xs, fs_list=fs):
    residuals = []
    for i, (x, fs_data) in enumerate(zip(x_list, fs_list)):
        tc = params[f'tc']
        M0 = params[f'M0{i+1}']
        D0 = params[f'D0']
        C = params[f'C']
        model = nogse.M_nogse_rest_offset(float(tnogses[i]), float(gs[i]), n, x, tc, M0, D0, C)
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

M0_fits = []
M0_errors = []

for i in range(len(xs)):
    M0_fits.append(result.params[f'M0{i+1}'].value)
    M0_errors.append(result.params[f'M0{i+1}'].stderr)

tc_fit = result.params[f'tc'].value
tc_error = result.params[f'tc'].stderr
D0_fit = result.params[f'D0'].value
D0_error = result.params[f'D0'].stderr
C_fit = result.params[f'C'].value
C_error = result.params[f'C'].stderr

with open(f"{directory}/parameters_N={int(n)}.txt", "a") as a:
    print("roi",  " - tc = ", tc_fit, " +/- ", tc_error, file=a)
    print("     ",  " - D0 = ", D0_fit, " +/- ", D0_error, file=a)
    print("     ",  " - C = ", C_fit, " +/- ", C_error, file=a)
    print("     ",  " - Chi cuadrado = ", result.chisqr, file=a)
    print("     ",  " - Reduced chi cuadrado = ", result.redchi, file=a)
    for i in range(len(xs)):
        print(f"    - M0_{i+1} = {M0_fits[i]} +- {M0_errors[i]}", file=a)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

for i in range(len(xs)):
    
    fig1, ax1 = plt.subplots(figsize=(8,6)) 

    x_fit = np.linspace(np.min(xs[i]), np.max(xs[i]), num=1000)
    fit = nogse.M_nogse_rest(float(tnogses[i]), float(gs[i]), n, x_fit, tc_fit, M0_fits[i], D0_fit)
    nogse.plot_nogse_vs_x_fit(ax, num_grad, modelo, xs[i], x_fit, fs[i], fit, tnogses[i], gs[i], n, slic, color = 'blue') 
    nogse.plot_nogse_vs_x_fit(ax1, num_grad, modelo, xs[i], x_fit, fs[i], fit, tnogses[i], gs[i], n, slic, color = 'blue')

    table = np.vstack((x_fit, fit))
    np.savetxt(f"{directory}/{roi}_fit_nogse_vs_x_tnogse={tnogses[i]}_g={gs[i]}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogses[i]}_g={gs[i]}_N={int(n)}_exp={exp}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogses[i]}_g={gs[i]}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig1)

with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_tnogse.txt", "a") as a:
    print(tnogse, tc_fit, tc_error, C_fit, C_error, file=a) 

with open(f"../results_{file_name}/{folder}/{roi}_M0_vs_tnogse.txt", "a") as a:
    for i in range(len(xs)):
        print(tnogse, M0_fits[i], M0_errors[i], file=a)