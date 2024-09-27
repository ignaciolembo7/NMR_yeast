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
folder = "globalfit_nogse_vs_x_mixto_mixto"
A0 = "sin_A0"
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # intra
n = 2
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "Mixto+Mixto"

palette = [
    "#1f77b4",  # Azul
    "#9467bd",  # Púrpura
    #"#e377c2",  # Rosa
    #"#7f7f7f",  # Gris
    "#8c564b",  # Marrón
    #"#f1c40f",  # Amarillo
    "#d62728",  # Rojo
]

num_grads = ["G1","G2","G3","G4"]  # Añadir los gradientes que desees
rois = ["ROI1","ROI1","ROI1","ROI1"]
#tnogses = [17.5, 17.5, 17.5, 17.5]
#gs = [105.0, 210.0, 405.0, 800.0]
tnogses = [21.5,21.5,21.5,21.5] 
gs = [75.0,160.0,300.0,700.0]

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/tnogse={tnogses[0]}_curves_N={int(n)}_exp={exp}"
os.makedirs(directory, exist_ok=True)

# Cargo la data para cada curva
xs = []
fs = []

for roi, tnogse, g, num_grad in zip(rois, tnogses, gs, num_grads):
    data = np.loadtxt(f"../results_{file_name}/nogse_vs_x_data/slice={slic}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}/{roi}_data_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt")

    x = data[:, 0]
    f = data[:, 1] #/data[0,1]

    vectores_combinados = zip(x, f)
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    x, f = zip(*vectores_ordenados)
    
    xs.append(x)
    fs.append(f)

# Parámetros genéricos
params = Parameters()
for i in range(len(xs)):
    params.add(f'tc_1_{i+1}', value=1, min=0.1, max=10.0, vary = False)
    params.add(f'alpha_1_{i+1}', value=0.59, min=0.1, max=1.0, vary = False)
    params.add(f'tc_2_{i+1}', value=1.8, min=1.0, max=15.0, vary = False)
    params.add(f'alpha_2_{i+1}', value=0.0, min=0.0, max=1.0, vary = False)
params.add('M0_1', value=5000, min=0, max=10000, vary=1)
params.add('M0_2', value=5000, min=0, max=10000, vary=1)
#params.add(f'D0_1', value=D0_ext, min = D0_int, max = D0_ext, vary=False)
#params.add(f'D0_2', value=D0_int, min = D0_int, max = D0_ext, vary=False)

#params.add('tc', value=8.0, min=1.0, max=15.0, vary = True)
#params.add(f'alpha', value=0.8, min=0.1, max=1.0, vary = True)
#params.add(f'D0', value=D0_ext, min = D0_int, max = D0_ext, vary=True)

def objective_function(params, x_list=xs, fs_list=fs):
    residuals = []
    for i, (x, fs_data) in enumerate(zip(x_list, fs_list)):
        tc_1_fit = params[f'tc_1_{i+1}'].value
        alpha_1_fit = params[f'alpha_1_{i+1}'].value
        M01_fit = params[f'M0_1'].value
        tc_2_fit = params[f'tc_2_{i+1}'].value
        alpha_2_fit = params[f'alpha_2_{i+1}'].value
        M02_fit = params[f'M0_2'].value
        model = nogse.fit_nogse_vs_x_mixto_mixto(tnogses[i], gs[i], n, x, tc_1_fit, alpha_1_fit, M01_fit, D0_ext, tc_2_fit, alpha_2_fit, M02_fit, D0_int)

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

M01_fit = result.params[f'M0_1'].value
error_M01 = result.params[f'M0_1'].stderr
M02_fit = result.params[f'M0_2'].value
error_M02 = result.params[f'M0_2'].stderr
#D01_fit = result.params[f'D0_ext'].value
#D02_fit = result.params[f'D0_int'].value

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
palette = sns.color_palette("tab10", len(xs)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

for i in range(len(xs)):
    
    fig1, ax1 = plt.subplots(figsize=(8,6)) 

    tc_1_fit = result.params[f'tc_1_{i+1}'].value
    tc_1_error = result.params[f'tc_1_{i+1}'].stderr
    alpha_1_fit = result.params[f'alpha_1_{i+1}'].value
    alpha_1_error = result.params[f'alpha_1_{i+1}'].stderr
    tc_2_fit = result.params[f'tc_2_{i+1}'].value
    tc_2_errors = result.params[f'tc_2_{i+1}'].stderr
    alpha_2_fit = result.params[f'alpha_2_{i+1}'].value
    alpha_2_error = result.params[f'alpha_2_{i+1}'].stderr

    x_fit = np.linspace( np.min(x), np.max(x), num=1000)
    fit = nogse.fit_nogse_vs_x_mixto_mixto(tnogses[i], gs[i], n, x_fit, tc_1_fit, alpha_1_fit, M01_fit, D0_ext, tc_2_fit, alpha_2_fit, M02_fit, D0_int)
    fit_1 = nogse.M_nogse_mixto(tnogses[i], gs[i], n, x_fit, tc_1_fit, alpha_1_fit, M01_fit, D0_ext)
    fit_2 = nogse.M_nogse_mixto(tnogses[i], gs[i], n, x_fit, tc_2_fit, alpha_2_fit, M02_fit, D0_int)

    nogse.plot_nogse_vs_x_fit(ax1, "fit ext", modelo, xs[i], x_fit, fs[i], fit_1, tnogses[i], gs[i], n, slic, color = 'orange')
    nogse.plot_nogse_vs_x_fit(ax1, "fit int", modelo, xs[i], x_fit, fs[i], fit_2, tnogses[i], gs[i], n, slic, color = 'green')
    nogse.plot_nogse_vs_x_fit(ax, num_grads[i], modelo, xs[i], x_fit, fs[i], fit, tnogse, gs[i], n, slic, color = palette[i]) 
    nogse.plot_nogse_vs_x_fit(ax1, num_grads[i], modelo, xs[i], x_fit, fs[i], fit, tnogse, gs[i], n, slic, color = palette[i])

    table = np.vstack((x_fit, fit))
    np.savetxt(f"{directory}/{roi}_ajuste_nogse_vs_x_tnogse={tnogse}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={gs[i]}_N={int(n)}_exp={exp}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={gs[i]}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig1)


fig.tight_layout()
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_N={int(n)}_exp={exp}.pdf")
fig.savefig(f"{directory}/nogse_vs_x_tnogse={tnogse}_N={int(n)}_exp={exp}.png", dpi=600)
plt.close(fig)



# with open(f"{directory}/parameters_tnogse={tnogse}_N={int(n)}.txt", "a") as a:
#     print(roi,  " - tc_1 = ", tc_1_fits[0], "+-", tc_1_errors[0], file=a)
#     print("     ",  " - alpha_1 = ", alpha_1_fits[0], "+-", alpha_1_errors[0], file=a)
#     print("     ",  " - M01 = ", M01_fit, "+-", error_M01, file=a)
#     print("     ",  " - D0_ext = ", D0_ext, file=a)
#     print("     ",  " - tc_2 = ", tc_2_fits[0], "+-", tc_2_errors[0], file=a)
#     print("     ",  " - alpha_2 = ", alpha_2_fits[0], "+-", alpha_2_errors[0], file=a)
#     print("     ",  " - M02 = ", M02_fit, "+-", error_M02, file=a)
#     print("     ",  " - D0_int = ", D0_int, file=a)