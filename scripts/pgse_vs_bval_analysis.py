#NMRSI - Ignacio Lembo Ferrari - 23/05/2024

import numpy as np
import matplotlib.pyplot as plt
from protocols import pgse
from lmfit import Model
import os
import seaborn as sns
from tqdm import tqdm
sns.set_theme(context='paper')
sns.set_style("whitegrid")

DwGradSep = float(input('Grad Separation (Delta) [ms]: ')) #ms
DwGradDur = float(input('Grad Duration (delta) [ms]: ')) #ms

file_name = "levaduras_20220829"
folder = "pgse_vs_bval_expmodel"
slic = 0 # slice que quiero ver
modelo = "expmodel"  # nombre carpeta modelo libre/rest/tort
D0_ext = 0.0023 #2.3e-12 # extra
D0_int = 0.0007 #0.7e-12 # intra

D0=D0_int

fig, ax = plt.subplots(figsize=(8,6)) 
rois = ["ROI1"] #,"ROI2", "ROI3","ROI4","ROI5"]
palette = sns.color_palette("tab20", len(rois)) # Generar una paleta de colores única (ej: husl, Set3, tab10, tab20)

# Create directory if it doesn't exist
directory = f"../results_{file_name}/pgse_vs_bvalue_exp/slice={slic}/DwGradDur={round(DwGradDur,2)}_DwGradSep={round(DwGradSep,2)}/"
os.makedirs(directory, exist_ok=True)

for roi, color in tqdm(zip(rois,palette)):

    data = np.loadtxt(f"../results_{file_name}/pgse_vs_bvalue_data/slice={slic}/DwGradDur={round(DwGradDur,2)}_DwGradSep={round(DwGradSep,2)}/{roi}_Datos_pgse_vs_bvalue_DwGradDur={round(DwGradDur,2)}_DwGradSep={round(DwGradSep,2)}.txt")

    bval = data[:, 0]
    f = data[:, 1]
    # Combinar los vectores usando zip()
    vectores_combinados = zip(bval, f)
    # Ordenar los vectores combinados basándote en vector_g
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    # Separar los vectores nuevamente
    bval, f = zip(*vectores_ordenados)

    #modelo M_pgse_exp
    model = Model(pgse.M_pgse_exp, independent_vars=["bval"], param_names=["M0", "D0"])
    model.set_param_hint("M0", value=100)
    model.set_param_hint("D0", value = D0_int)
    params = model.make_params()
    #params["M0"].vary = False # fijo M0 en 1, los datos estan normalizados y no quiero que varíe
    params["M0"].vary = 1
    params["D0"].vary = 1
    
    result = model.fit(f[-15:], params, bval=bval[-15:])
    M0_fit = result.params["M0"].value
    error_M0 = result.params["M0"].stderr
    D0_fit = result.params["D0"].value
    error_D0 = result.params["D0"].stderr

    print(result.fit_report())

    bval_fit = np.linspace(np.min(bval[-15:]), np.max(bval[-15:]), num=1000)
    fit = pgse.M_pgse_exp(bval_fit, M0_fit, D0_fit)

    with open(f"{directory}/parameters_{roi}_pgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.txt", "a") as a:
        print(roi,  " - M0 = ", M0_fit, "+-", error_M0, file=a)
        print("    ",  " - D0 = ", D0_fit, "+-", error_D0, file=a)
    
        
    fig1, ax1 = plt.subplots(figsize=(8,6))

    pgse.plot_pgse_vs_bval_rest(ax, roi, modelo, bval, bval_fit, f, fit, D0_fit, DwGradDur, DwGradSep, slic, color)
    pgse.plot_pgse_vs_bval_rest(ax1, roi, modelo, bval, bval_fit, f, fit, D0_fit, DwGradDur, DwGradSep, slic, color)

    table = np.vstack((bval_fit, fit))
    np.savetxt(f"{directory}/{roi}_ajuste_pgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.txt", table.T, delimiter=' ', newline='\n')
    
    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_pgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.pdf")
    fig1.savefig(f"{directory}/{roi}_pgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.png", dpi=600)
    plt.close(fig1)

fig.tight_layout()
fig.savefig(f"{directory}/pgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.pdf")
fig.savefig(f"{directory}/pgse_vs_bvalue_DwGradDur={round(DwGradDur,6)}_DwGradSep={round(DwGradSep,6)}.png", dpi=600)
plt.close(fig)