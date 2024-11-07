import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import lmfit
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

tnogses = [15.0,17.5,21.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0]
num_grads = ["G1", "G2", "G3", "G4"]
gss = [[100.0, 275.0, 600.0, 1000.0], [105.0, 210.0, 405.0, 800.0], [75.0, 160.0, 300.0, 700.0], [60.0, 120.0, 210.0, 600.0], [55.0, 110.0, 190.0, 550.0], [50.0, 100.0, 170.0, 500.0], [45.0, 90.0, 150.0, 450.0], [40.0, 80.0, 130.0, 400.0], [35.0, 75.0, 120.0, 375.0], [30.0, 70.0, 110.0, 350.0]]
alphas_centrales = [0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55]  # Valores centrales de α para cada tnogse
entorno_alpha = 0.05  # Rango de variación alrededor de los valores centrales de α

parameters = {(tnogse, num_grad): {'tc1_fit': [], 'tc1_error': [], 'alpha1': [], 'C': [], 'C_error': [], 'chi': []} for tnogse in tnogses for num_grad in num_grads}

file_name = "levaduras_20240622"
A0 = "sin_A0"
exp = 1
n = 2
slic = 0
modelo = "Mixto-rest"
D0_ext = 2.3e-12
D0_int = 0.7e-12
roi = "ROI1"

palette = [
"#1f77b4",  # Azul
"#9467bd",  # Púrpura
#"#e377c2",  # Rosa
##"#7f7f7f",  # Gris
#"#8c564b",  # Marrón
"#f1c40f",  # Amarillo
"#d62728",  # Rojo
]   
sns.set_palette(palette)

for tnogse, gs, alpha_central in zip(tnogses, gss, alphas_centrales):

    alphas1 = [alpha_central - entorno_alpha, alpha_central, alpha_central + entorno_alpha]

    for num_grad, g, color in zip(num_grads, gs, palette):

        fig, ax = plt.subplots(figsize=(8, 6))

        folder = f"fit_nogse_vs_x_mixto-rest_sinoffset_bestalpha_M01=0.75_D02={D0_int}"
        directory = f"../results_{file_name}/{folder}/{A0}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}"
        os.makedirs(directory, exist_ok=True)

        data = np.loadtxt(f"../results_{file_name}/nogse_vs_x_data/slice={slic}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}/{roi}_data_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt")
        x = data[:, 0]
        f = data[:, 1]
        vectores_combinados = zip(x, f)
        vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
        x, f = zip(*vectores_ordenados)

        # Graficar puntos experimentales
        ax.scatter(x, f, marker='o', label="Datos experimentales", zorder=3, color = color)

        fit_palette = sns.color_palette("tab20", len(alphas1))

        for i, alpha1 in enumerate(alphas1):

            model = lmfit.Model(nogse.fit_nogse_vs_x_mixto_mixto_offset, independent_vars=["TE", "G", "N", "x", "D0_1", "D0_2"], param_names=["tc_1", "alpha_1", "M0_1", "tc_2", "alpha_2", "M0_2", "C"])
            #model.set_param_hint("M0_1", value=1000.0, min=0.0, max=5000.0, vary=1)
            model.set_param_hint("M0_2", value=1000.0, min=0.0, max=5000.0, vary=1)
            model.set_param_hint("M0_1", expr="(0.75/0.25)*M0_2")
            model.set_param_hint("tc_1", value=10.0, min=0.1, max=50.0, vary=1)
            model.set_param_hint("tc_2", value=2.23, min=0.1, max=50.0, vary=0)
            model.set_param_hint("alpha_1", value=alpha1, min=0.0, max=1.0, vary=0)
            model.set_param_hint("alpha_2", value=0, min=0.0, max=1.0, vary=0)
            model.set_param_hint("C", value=0, min=0, max=5000.0, vary=0)
            params = model.make_params()

            result = model.fit(f, params, TE=float(tnogse), G=float(g), N=n, x=x, D0_1=D0_ext, D0_2=D0_int) 

            M01_fit = result.params["M0_1"].value
            M02_fit = result.params["M0_2"].value
            tc1_fit = result.params["tc_1"].value
            tc2_fit = result.params["tc_2"].value
            alpha2_fit = result.params["alpha_2"].value
            C_fit = result.params["C"].value
            M01_error = result.params["M0_1"].stderr
            M02_error = result.params["M0_2"].stderr
            tc1_error = result.params["tc_1"].stderr
            tc2_error = result.params["tc_2"].stderr
            alpha2_error = result.params["alpha_2"].stderr
            C_error = result.params["C"].stderr
            
            parameters[(tnogse, num_grad)]['tc1_fit'].append(tc1_fit)
            parameters[(tnogse, num_grad)]['alpha1'].append(alpha1)
            parameters[(tnogse, num_grad)]['tc1_error'].append(tc1_error)
            parameters[(tnogse, num_grad)]['C'].append(C_fit)
            parameters[(tnogse, num_grad)]['C_error'].append(C_error)
            parameters[(tnogse, num_grad)]['chi'].append(result.chisqr)

            x_fit = np.linspace(np.min(x), np.max(x), num=1000)
            fit = nogse.fit_nogse_vs_x_mixto_mixto_offset(tnogse, g, n, x_fit, tc1_fit, alpha1, M01_fit, D0_ext, tc2_fit, alpha2_fit, M02_fit, D0_int, C_fit)

            with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
                print(f"Ajuste {i + 1} - alpha1 = {alpha1}", file=a)
                print(roi,  " - tc_1 = ", tc1_fit, "+-", tc1_error, file=a)
                print("    "  " - M0_1 = ", M01_fit, "+-", M01_error, file=a)
                print("    ",  " - D0_1 = ", D0_ext, file=a)
                print("    ",  " - alpha_2 = ", alpha2_fit, "+-", alpha2_error, file=a)
                print("    ",  " - tc_2 = ", tc2_fit, "+-", tc2_error, file=a)
                print("    ",  " - M0_2 = ", M02_fit, "+-", M02_error, file=a)
                print("    ",  " - D0_2 = ", D0_int, file=a)
                print("    ",  " - Chi cuadrado = ", result.chisqr, file=a)
                print("    ",  " - Reduced chi cuadrado = ", result.redchi, file=a)

            # Graficar con línea punteada para α central y línea sólida para los otros valores
            linestyle = '--' if alpha1 == alpha_central else '-'
            #si tc1_error es None entonces tc1_error = 0
            if tc1_error == None:
                tc1_error = 0
            ax.plot(x_fit, fit, linewidth=2, label=f"$\\alpha$ = {round(alpha1,2)} - $\\tau_{{c,ext}}$ = ({round(tc1_fit,2)} $\\pm$ {round(tc1_error,2)}) ms", linestyle=linestyle, color=fit_palette[i])

        ax.legend(title_fontsize=10, fontsize=10, loc='best')
        ax.set_xlabel("Tiempo de modulación x [ms]", fontsize=27)
        ax.set_ylabel("Señal $\mathrm{NOGSE}$", fontsize=27)
        ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
        ax.tick_params(axis='x', rotation=0, labelsize=16, color='black')
        ax.tick_params(axis='y', labelsize=16, color='black')
        ax.set_title(f"{modelo} | $T_\\mathrm{{NOGSE}}$ = {tnogse} ms | {num_grad} = {g} mT/m | $N$ = {n} | slice = {slic}", fontsize=15)

        fig.tight_layout()
        fig.savefig(f"../results_{file_name}/{folder}/{A0}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
        fig.savefig(f"../results_{file_name}/{folder}/{A0}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
        plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))

# Itera sobre todos los gradientes
for num_grad, color in zip(num_grads, palette):
    tcs = []  # Lista para almacenar los valores de tc para cada tnogse
    tcs_error = []  # Lista para almacenar los errores de tc para cada tnogse
    
    # Itera sobre todos los tnogses
    for tnogse in tnogses:
        tcs.append(parameters[(tnogse, num_grad)]['tc1_fit'][1])
        tcs_error.append(parameters[(tnogse, num_grad)]['tc1_error'][1])
    
    # Graficar tc vs tnogse con barras de error
    ax.errorbar(tnogses, tcs, yerr=0, fmt='o-', markersize=3, linewidth=2, capsize=5, label=f"{num_grad}", color=color)

ax.set_xlabel(f"Tiempo de difusión $T_\\mathrm{{NOGSE}}$ [ms]", fontsize=27)
ax.set_ylabel(f"Tiempo de correlación $\\tau_{{c,ext}}$ [ms]", fontsize=27)
ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax.tick_params(axis='x', rotation=0, labelsize=16, color='black')
ax.tick_params(axis='y', labelsize=16, color='black')
ax.legend(title="Gradiente", title_fontsize=15, fontsize=15, loc='best')

fig.tight_layout()
fig.savefig(f"../results_{file_name}/{folder}/{A0}/tc1_vs_tnogse.pdf")
fig.savefig(f"../results_{file_name}/{folder}/{A0}/tc1_vs_tnogse.png", dpi=600)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))

# Itera sobre todos los gradientes
for num_grad, color in zip(num_grads, palette):
    Cs = []  # Lista para almacenar los valores de C para cada tnogse
    Cs_error = []  # Lista para almacenar los errores de C para cada tnogse
    
    # Itera sobre todos los tnogses
    for tnogse in tnogses:
        Cs.append(parameters[(tnogse, num_grad)]['C'][1])
        Cs_error.append(parameters[(tnogse, num_grad)]['C_error'][1])
    
    # Graficar C vs tnogse con barras de error
    ax.errorbar(tnogses, Cs, yerr=0, fmt='o-', markersize=3, linewidth=2, capsize=5, label=f"{num_grad}", color=color)

ax.set_xlabel("Tiempo de difusión $T_\\mathrm{NOGSE}$ [ms]", fontsize=27)
ax.set_ylabel("Offset señal $C$", fontsize=27)
ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
ax.tick_params(axis='x', rotation=0, labelsize=16, color='black')
ax.tick_params(axis='y', labelsize=16, color='black')
ax.legend(title="Gradiente", title_fontsize=15, fontsize=15, loc='best')

fig.tight_layout()
fig.savefig(f"../results_{file_name}/{folder}/{A0}/C_vs_tnogse.pdf")
fig.savefig(f"../results_{file_name}/{folder}/{A0}/C_vs_tnogse.png", dpi=600)
plt.close(fig)