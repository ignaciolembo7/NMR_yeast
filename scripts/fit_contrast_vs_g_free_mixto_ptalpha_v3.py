import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import lmfit
import os
import seaborn as sns
sns.set_theme(context='paper')
sns.set_style("whitegrid")

tnogses = [15.0,17.5,21.5,25.0,27.5,30.0,35.0,40.0]
alphass = [[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
# Listas para almacenar t_c y alpha para cada combinación de tnogse y num_grad
alpha_vs_tc = {(tnogse): {'alpha': [], 'chi': []} for tnogse in tnogses}

file_name = "levaduras_20240613"
A0 = "con_A0"
exp = 1
n = 2
slic = 0
modelo = "Free-Rest"

D0_ext = 2.3e-12
D0_int = 0.7e-12
roi = "ROI1"
palette = [
"#17becf", # Azul claro
"#1f77b4",  # Azul
"#9467bd",  # Púrpura
"#e377c2",  # Rosa
"#7f7f7f",  # Gris
"#8c564b",  # Marrón
"#f1c40f",  # Amarillo
"#d62728",  # Rojo
]   
sns.set_palette(palette)

# for roi, color in zip(rois, palette):

for tnogse, alphas, color in zip(tnogses, alphass, palette):

        fig, ax = plt.subplots(figsize=(8, 6)) 

        folder = f"fit_contrast_vs_g_free_rest_pt_alpha_M01=0.75_D02={D0_int}"

        directory = f"../results_{file_name}/{folder}/{A0}/tnogse={tnogse}_N={int(n)}_exp={exp}"
        os.makedirs(directory, exist_ok=True)

        data = np.loadtxt(f"../results_{file_name}/plot_contrast_vs_g_data/{A0}/tnogse={tnogse}_N={int(n)}_exp={exp}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={int(n)}.txt")

        g = data[:, 0]
        f = data[:, 1]
        vectores_combinados = zip(g, f)
        vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
        g, f = zip(*vectores_ordenados)

        fit_palette = sns.color_palette("tab20", len(alphas))  # Colores para los ajustes

        for i, alpha in enumerate(alphas):

            #remover los elementos en la posicion _ de x y f 
            g = np.delete(g, [0])
            f = np.delete(f, [0])

            model = lmfit.Model(nogse.fit_contrast_vs_g_free_mixto, independent_vars=["TE", "G", "N", "D0_1", "D0_2"], param_names=["alpha_1", "M0_1", "tc_2", "alpha_2", "M0_2"])
            model.set_param_hint("M0_1", value=0.75, min=0, max = 1.0, vary = 0) #extracelular
            model.set_param_hint("M0_2", expr="1 - M0_1") 
            #model.set_param_hint("M0_2", value=0.15, min=0, max = 1.0, vary = 1) #intracelular
            model.set_param_hint("tc_2", value=2.23, min = 0, max = 10.0, vary = 0) #intracelular
            model.set_param_hint("alpha_1", value=alpha, min = 0, max=1, vary = 0) #extracelular
            model.set_param_hint("alpha_2", value=0.0, min = 0, max=1, vary = 0) #intracelular

            params = model.make_params()

            result = model.fit(f, params, TE=float(tnogse), G=g, N=n, D0_1=D0_ext, D0_2=D0_int) 

            print(f"Ajuste {i + 1} para alpha = {alpha}")
            print(result.params.pretty_print())
            print(f"Chi cuadrado = {result.chisqr}")
            print(f"Reduced chi cuadrado = {result.redchi}")

            M01_fit = result.params["M0_1"].value
            M02_fit = result.params["M0_2"].value
            M02_error = result.params["M0_1"].stderr
            M01_error = result.params["M0_2"].stderr
            tc2_fit = result.params["tc_2"].value
            tc2_error = result.params["tc_2"].stderr
            alpha2_fit = result.params["alpha_2"].value
            alpha2_error = result.params["alpha_2"].stderr

            #si t_c=0.1 pasar a la siguiente iteración 
            # if tc_fit < 0.15:
            #     continue

            # Guardar los valores ajustados de t_c y alpha para graficar más tarde
            alpha_vs_tc[(tnogse)]['alpha'].append(alpha)
            alpha_vs_tc[(tnogse)]['chi'].append(result.chisqr)

            g_fit = np.linspace(0, np.max(g), num=1000)
            fit = nogse.fit_contrast_vs_g_free_mixto(tnogse, g_fit, n, alpha, M01_fit, D0_ext, tc2_fit, alpha2_fit, M02_fit, D0_int)

            with open(f"{directory}/parameters_tnogse={tnogse}_N={int(n)}.txt", "a") as a:
                print(f"Ajuste {i + 1} - alpha1 = {alpha}", file=a)
                print(roi,  " - M0_1 = ", M01_fit, "+-", M01_error, file=a)
                print("    ",  " - D0_1 = ", D0_ext, file=a)
                print("    ",  " - tc_2 = ", tc2_fit, "+-", tc2_error, file=a)
                print("    ",  " - alpha_2 = ", alpha2_fit, "+-", alpha2_error, file=a)
                print("    ",  " - M0_2 = ", M02_fit, "+-", M02_error, file=a)
                print("    ",  " - D0_2 = ", D0_int, file=a)
                print("    ",  " - Chi cuadrado = ", result.chisqr, file=a)
                print("    ",  " - Reduced chi cuadrado = ", result.redchi, file=a)
                
            ax.plot(g, f, "o", markersize=7, linewidth=2, color = color)
            ax.plot(g_fit, fit, linewidth=2, label = alpha, color = fit_palette[i])
            ax.legend(title = f"$\\alpha_{{ext}}$", title_fontsize=10, fontsize=10, loc='best')
            ax.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=27)
            ax.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
            ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
            ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
            ax.tick_params(axis='y', labelsize=16, color='black')
            title = ax.set_title(f"{modelo} | $T_\mathrm{{NOGSE}}$ = {tnogse} ms | $N$ = {n} | slice = {slic} ", fontsize=15)

            # table = np.vstack((x_fit, fit))
            # np.savetxt(f"{directory}/{roi}_fit_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_tc={tc}.txt", table.T, delimiter=' ', newline='\n')
            #fig1, ax1 = plt.subplots(figsize=(8, 6)) 
            #nogse.plot_nogse_vs_x_fit(ax1, roi, modelo, x, x_fit, f, fit, tnogse, g, n, num_grad, slic, tc, fit_palette[i])
            # fig1.tight_layout()
            # fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_tc={tc}.pdf")
            # fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_tc={tc}.png", dpi=600)
            # plt.close(fig1)

        fig.tight_layout()
        fig.savefig(f"../results_{file_name}/{folder}/{A0}/contrast_vs_g_tnogse={tnogse}_N={int(n)}.pdf")
        fig.savefig(f"../results_{file_name}/{folder}/{A0}/contrast_vs_g_tnogse={tnogse}_N={int(n)}.png", dpi=600)
        plt.close(fig)