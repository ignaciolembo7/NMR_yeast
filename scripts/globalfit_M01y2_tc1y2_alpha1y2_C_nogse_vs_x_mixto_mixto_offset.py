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
folder = "globalfit_M01y2_tc1y2_alpha1y2_C_nogse_vs_x_mixto_rest_offset"
A0 = "sin_A0"
D0_ext = 2.3e-12 # extra
D0_int = 0.7e-12 # intra
n = 2
exp = 1 #int(input('exp: '))
slic = 0 # slice que quiero ver
modelo = "Mixto+Rest-offset"

palette = [
    "#1f77b4",  # Azul
    "#9467bd",  # Púrpura
    #"#e377c2",  # Rosa
    #"#7f7f7f",  # Gris
    "#8c564b",  # Marrón
    #"#f1c40f",  # Amarillo
    "#d62728",  # Rojo
]

num_grads = ["G1","G2","G3","G4"]
rois = ["ROI1","ROI1","ROI1","ROI1"]

tnogses = [15.0,15.0,15.0,15.0]
gs=[100.0,275.0,600.0,1000.0]
tnogses = [17.5, 17.5, 17.5, 17.5]
gs = [105.0, 210.0, 405.0, 800.0]
tnogses = [21.5,21.5,21.5,21.5] 
gs = [75.0,160.0,300.0,700.0]
tnogses = [25.0, 25.0,25.0,25.0]
gs = [60.0,120.0,210.0,600.0]
# tnogses = [27.5, 27.5, 27.5,27.5]
# gs = [55.0, 110.0, 190.0, 550.0]
# tnogses = [30.0,30.0,30.0,30.0]
# gs = [50.0,100.0,170.0,500.0]
# tnogses = [35.0,35.0,35.0,35.0]
# gs = [40.0,80.0,130.0,400.0]
# tnogses = [37.5,37.5,37.5,37.5]
# gs = [35.0,75.0,120.0,375.0]
# tnogses = [40.0,40.0,40.0,40.0]
# gs = [30.0,70.0,110.0,350.0]

# Create directory if it doesn't exist
directory = f"../results_{file_name}/{folder}/tnogse={tnogses[0]}_N={int(n)}_exp={exp}"
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
#for i in range(len(xs)):

params.add(f'tc1', value= 3.1, min=0.1, max=50.0, vary = 1)
params.add(f'alpha1', value=0.45, min=0.1, max=1.0, vary = 1)
params.add(f'tc2', value=1.6, min=1.0, max=15.0, vary = 1)
params.add(f'alpha2', value=0.0, min=0.0, max=1.0, vary = 0)
params.add('M01', value=2500, min=0, max=10000, vary=1)
params.add('M02', value=2000, min=0, max=10000, vary=1)
params.add(f'C', value=100, min=0, vary = 1)
#params.add(f'D0_1', value=D0_ext, min = D0_int, max = D0_ext, vary=False)
#params.add(f'D0_2', value=D0_int, min = D0_int, max = D0_ext, vary=False)

def objective_function(params, x_list=xs, fs_list=fs):
    residuals = []
    for i, (x, fs_data) in enumerate(zip(x_list, fs_list)):
        tc1_fit = params[f'tc1'].value
        alpha1_fit = params[f'alpha1'].value
        M01_fit = params[f'M01'].value
        tc2_fit = params[f'tc2'].value
        alpha2_fit = params[f'alpha2'].value
        M02_fit = params[f'M02'].value
        C_fit = params[f'C'].value  
        model = nogse.fit_nogse_vs_x_mixto_mixto_offset(tnogses[i], gs[i], n, x, tc1_fit, alpha1_fit, M01_fit, D0_ext, tc2_fit, alpha2_fit, M02_fit, D0_int, C_fit)

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

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

for i in range(len(xs)):
    
    fig1, ax1 = plt.subplots(figsize=(8,6)) 
    fig2, ax2 = plt.subplots(figsize=(8,6)) 


    M01_fit = result.params[f'M01'].value
    M01_error = result.params[f'M01'].stderr
    M02_fit = result.params[f'M02'].value
    M02_error = result.params[f'M02'].stderr
    tc1_fit = result.params[f'tc1'].value
    tc1_error = result.params[f'tc1'].stderr
    alpha1_fit = result.params[f'alpha1'].value
    alpha1_error = result.params[f'alpha1'].stderr
    tc2_fit = result.params[f'tc2'].value
    tc2_error = result.params[f'tc2'].stderr
    alpha2_fit = result.params[f'alpha2'].value
    alpha2_error = result.params[f'alpha2'].stderr
    C_fit = result.params[f'C'].value
    C_error = result.params[f'C'].stderr

    x_fit = np.linspace( np.min(x), np.max(x), num=1000)
    fit = nogse.fit_nogse_vs_x_mixto_mixto_offset(tnogses[i], gs[i], n, x_fit, tc1_fit, alpha1_fit, M01_fit, D0_ext, tc2_fit, alpha2_fit, M02_fit, D0_int, C_fit)
    fit_1 = nogse.M_nogse_mixto(tnogses[i], gs[i], n, x_fit, tc1_fit, alpha1_fit, M01_fit, D0_ext)
    fit_2 = nogse.M_nogse_mixto(tnogses[i], gs[i], n, x_fit, tc2_fit, alpha2_fit, M02_fit, D0_int)

    nogse.plot_nogse_vs_x_fit(ax1, "fit ext", modelo, xs[i], x_fit, fs[i], fit_1, tnogses[i], gs[i], n, slic, color = 'orange')
    nogse.plot_nogse_vs_x_fit(ax1, "fit int", modelo, xs[i], x_fit, fs[i], fit_2, tnogses[i], gs[i], n, slic, color = 'green')
    nogse.plot_nogse_vs_x_fit(ax, num_grads[i], modelo, xs[i], x_fit, fs[i], fit, tnogse, gs[i], n, slic, color = palette[i]) 
    nogse.plot_nogse_vs_x_fit(ax1, num_grads[i], modelo, xs[i], x_fit, fs[i], fit, tnogse, gs[i], n, slic, color = palette[i])
    nogse.plot_nogse_vs_x_fit(ax2, num_grads[i], modelo, xs[i], x_fit, fs[i], fit, tnogse, gs[i], n, slic, color = palette[i]) 

    table = np.vstack((x_fit, fit))
    np.savetxt(f"{directory}/{roi}_fit_nogse_vs_x_tnogse={tnogse}_g={gs[i]}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={gs[i]}_N={int(n)}_exp={exp}.pdf")
    fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={gs[i]}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig1)
    
    fig2.tight_layout()
    fig2.savefig(f"../results_{file_name}/{folder}/{roi}_nogse_vs_x_tnogse={tnogse}_g={gs[i]}_N={int(n)}_exp={exp}.pdf")
    fig2.savefig(f"../results_{file_name}/{folder}/{roi}_nogse_vs_x_tnogse={tnogse}_g={gs[i]}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig2)

with open(f"{directory}/parameters_tnogse={tnogse}_N={int(n)}.txt", "a") as a:
    print(roi,  " - tc_1 = ", tc1_fit, "+-", tc1_error, file=a)
    print("     ",  " - alpha_1 = ", alpha1_fit, "+-", alpha1_error, file=a)
    print("     ",  " - M01 = ", M01_fit, "+-", M01_error, file=a)
    print("     ",  " - D0_ext = ", D0_ext, file=a)
    print("     ",  " - tc_2 = ", tc2_fit, "+-", tc2_error, file=a)
    print("     ",  " - alpha_2 = ", alpha2_fit, "+-", alpha2_error, file=a)
    print("     ",  " - M02 = ", M02_fit, "+-", M02_error, file=a)
    print("     ",  " - C = ", C_fit, "+-", C_error, file=a)
    print("     ",  " - D0_int = ", D0_int, file=a)
    print("     ",  " - Chi cuadrado = ", result.chisqr, file=a)
    print("     ",  " - Reduced chi cuadrado = ", result.redchi, file=a)

with open(f"../results_{file_name}/{folder}/{roi}_parameters_vs_tnogse.txt", "a") as a:
    print(tnogse, tc1_fit, tc1_error, alpha1_fit, alpha1_error, M01_fit, M01_error, tc2_fit, tc2_error, alpha2_fit, alpha2_error, M02_fit, M02_error, C_fit, C_error, file=a) 

fig.tight_layout()
fig.savefig(f"../results_{file_name}/{folder}/nogse_vs_x_tnogse={tnogse}_N={int(n)}_exp={exp}.pdf")
fig.savefig(f"../results_{file_name}/{folder}/nogse_vs_x_tnogse={tnogse}_N={int(n)}_exp={exp}.png", dpi=600)
plt.close(fig)