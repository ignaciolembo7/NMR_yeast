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
folder = "nogse_vs_x_restdist_mode_globalfit"
A0 = "sin_A0"
D0_folder = "D0_ext"
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # intra
D0 = 1.35e-12 # fitted
n = 2
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "restdist_mode"  # nombre carpeta modelo libre/rest/tort

tnogses = [15.0, 17.5, 21.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0]
# G1 = [100.0, 105.0, 75.0, 60.0, 55.0, 50.0, 45.0, 40.0, 35.0, 30.0]
# G2 = [275.0, 210.0, 160.0, 120.0, 110.0, 100.0, 90.0, 80.0, 75.0, 70.0]
# G3 = [600.0, 405.0, 300.0, 210.0, 190.0, 170.0, 150.0, 130.0, 120.0, 110.0]
G4 = [1000.0, 800.0, 700.0, 600.0, 550.0, 500.0, 450.0, 400.0, 375.0, 350.0]

num_grads = ["G2", "G3"]  # Añadir los gradientes que desees
tnogses = ["27.5", "27.5"]  # Añadir los tnogses correspondientes
gs = ["110.0", "190.0"]  # Añadir las intensidades de gradiente correspondientes
rois = ["ROI1", "ROI1"]

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/{D0_folder}/tnogse={tnogses[0]}_g=G2G3_curves_N={int(n)}_exp={exp}"
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
    params.add(f'lc{i+1}', value=8.0, min=1.0, max=15.0, vary = True)
    params.add(f'sigma{i+1}', value=1.5, min=0.1, max=5.0, vary = True)
    params.add(f'D0{i+1}', value=D0_ext, min = D0_int, max = D0_ext, vary=False)
params.add('M0', value=1500, vary=True)

def objective_function(params, x_list=xs, fs_list=fs):
    residuals = []
    for i, (x, fs_data) in enumerate(zip(x_list, fs_list)):
        lc = params[f'lc{i+1}']
        sigma = params[f'sigma{i+1}']
        M0 = params['M0']
        D0 = params[f'D0{i+1}']
        model = nogse.M_nogse_restdist(float(tnogses[i]), float(gs[i]), n, x, lc, sigma, M0, D0)
        fs_data = np.array(fs_data)

        if np.isnan(model).any() or np.isnan(fs_data).any():
            raise ValueError("NaN detected in model or fs values")
                          
        residuals.append(fs_data - model)
    
    return np.concatenate(residuals)

result = minimize(objective_function, params)

# Display fitting results
print(result.params.pretty_print())

sigma_fits = []
error_sigmas = []
lc_fits = []
error_lcs = []
D0_fits = []
error_D0s = []

M0_fit = result.params["M0"].value
error_M0 = result.params["M0"].stderr
#D0_fit = result.params["D0"].value
#error_D0 = result.params["D0"].stderr

for i in range(len(xs)):
    lc_fits.append(result.params[f'lc{i+1}'].value)
    sigma_fits.append(result.params[f'sigma{i+1}'].value)
    error_lcs.append(result.params[f'lc{i+1}'].stderr)
    error_sigmas.append(result.params[f'sigma{i+1}'].stderr)
    D0_fits.append(result.params[f"D0{i+1}"].value)
    error_D0s.append(result.params[f"D0{i+1}"].stderr)

x_fit = np.linspace(np.min(xs[0]), np.max(xs[0]), num=1000)
fits = [nogse.M_nogse_restdist(float(tnogses[i]), float(gs[i]), n, x_fit, lc_fits[i], sigma_fits[i], M0_fit, D0_fits[i]) for i in range(len(xs))]

palette = sns.color_palette("tab10", len(rois))

fig, ax = plt.subplots(figsize=(8,6)) 
fig1, ax1 = plt.subplots(figsize=(8,6)) 

for i, (f, fit, lc_fit, sigma_fit, num_grad, g, color) in enumerate(zip(fs, fits, lc_fits, sigma_fits, num_grads, gs, palette)):
    nogse.plot_nogse_vs_x_restdist_ptG(ax, rois[0], modelo, xs[i], x_fit, f, fit, tnogses[i], n, slic, color, label=f"{num_grad} = {g}") 
    l_c = np.linspace(0.01, 40, 1000)
    dist = nogse.lognormal(l_c, sigma_fit, lc_fit)
    nogse.plot_lognorm_dist_ptG(ax1, rois[0], tnogses[i], n, l_c, dist, slic, color, label=f"{num_grad} = {g}")

fig.tight_layout()
fig.savefig(f"{directory}/{rois[0]}_nogse_vs_x_tnogse={tnogses[0]}_N={int(n)}_exp={exp}.pdf")
fig.savefig(f"{directory}/{rois[0]}_nogse_vs_x_tnogse={tnogses[0]}_N={int(n)}_exp={exp}.png", dpi=600)
plt.close(fig)

fig1.tight_layout()
fig1.savefig(f"{directory}/{rois[0]}_dist_tnogse={tnogses[0]}_N={int(n)}_exp={exp}.pdf")
fig1.savefig(f"{directory}/{rois[0]}_dist_tnogse={tnogses[0]}_N={int(n)}_exp={exp}.png", dpi=600)
plt.close(fig1)

with open(f"{directory}/parameters_tnogse={tnogses[0]}_N={int(n)}.txt", "a") as a:
    for i in range(len(xs)):
        print(f"{rois[0]} - lc_mode_{i+1} = {lc_fits[i]} +- {error_lcs[i]}", file=a)
        print(f"    - sigma_{i+1} = {sigma_fits[i]} +- {error_sigmas[i]}", file=a)
        print(f"    - D0_{i+1} = {D0_fits[i]} +- {error_D0s[i]}", file=a)
    print(f"    - M0 = {M0_fit} +- {error_M0}", file=a)

table = np.vstack([x_fit] + fits)
np.savetxt(f"{directory}/{rois[0]}_globalfit_nogse_vs_x_tnogse={tnogses[0]}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

for roi, tnogse, g, num_grad, lc_fit, error_lc, sigma_fit, error_sigma, D0_fit, error_D0 in zip(rois, tnogses, gs, num_grads, lc_fits, error_lcs, sigma_fits, error_sigmas, D0_fits, error_D0s):
    with open(f"../results_{file_name}/{folder}/{D0_folder}/{roi}_parameters_vs_tnogse_g={num_grad}.txt", "a") as a:
        print(tnogse, g, lc_fit, error_lc, sigma_fit, error_sigma, M0_fit, error_M0, D0_fit, error_D0, file=a)

for roi, tnogse, g, num_grad, lc_fit, error_lc, sigma_fit, error_sigma, D0_fit, error_D0 in zip(rois, tnogses, gs, num_grads, lc_fits, error_lcs, sigma_fits, error_sigmas, D0_fits, error_D0s):
    with open(f"../results_{file_name}/{folder}/{D0_folder}/{roi}_parameters_vs_g_tnogse={tnogse}.txt", "a") as a:    
        print(g, tnogse, lc_fit, error_lc, sigma_fit, error_sigma, M0_fit, error_M0, D0_fit, error_D0, file=a)











"""
def linear_model(x, slope, intercept, shear):
    return slope * x + intercept + shear * x**2

x = np.linspace(0, 10, 100)

# Dataset 1 parameters
slope1 = 2.5
intercept1 = 1.0
shear = 0.1  # Shared parameter

# Dataset 2 parameters
slope2 = 3.0
intercept2 = -0.5

# Generating data
y1 = linear_model(x, slope1, intercept1, shear) + np.random.normal(scale=0.2, size=x.size)
y2 = linear_model(x, slope2, intercept2, shear) + np.random.normal(scale=0.2, size=x.size)

params = Parameters()
params.add('slope1', value=2.0)
params.add('intercept1', value=0.0)
params.add('slope2', value=2.0)
params.add('intercept2', value=0.0)
params.add('shear', value=0.1, vary=True)  # Shared parameter

def objective_function(params, x, y1, y2):
    slope1 = params['slope1']
    intercept1 = params['intercept1']
    slope2 = params['slope2']
    intercept2 = params['intercept2']
    shear = params['shear']

    model1 = linear_model(x, slope1, intercept1, shear)
    model2 = linear_model(x, slope2, intercept2, shear)
    a = np.concatenate([(y1 - model1), (y2 - model2)])

    return a

from lmfit import minimize

# Minimize the objective function
result = minimize(objective_function, params, args=(x, y1, y2))

# Display fitting results
print(result.params.pretty_print())

# Best fit parameters
slope1_fit = result.params['slope1'].value
intercept1_fit = result.params['intercept1'].value
slope2_fit = result.params['slope2'].value
intercept2_fit = result.params['intercept2'].value
shear_fit = result.params['shear'].value

y1_fit = linear_model(x, slope1_fit, intercept1_fit, shear_fit)
y2_fit = linear_model(x, slope2_fit, intercept2_fit, shear_fit)

plt.figure(figsize=(10, 6))
plt.scatter(x, y1, label='Data 1')
plt.plot(x, y1_fit, label='Fit 1', color='red')

plt.scatter(x, y2, label='Data 2')
plt.plot(x, y2_fit, label='Fit 2', color='blue')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Global Fit with Shared Shear Parameter')
plt.show()

"""

