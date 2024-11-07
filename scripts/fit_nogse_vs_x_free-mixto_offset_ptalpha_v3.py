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
alphass1 = [[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,0.75,0.8,0.85,0.9], [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,0.75,0.8,0.85,0.9],[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,0.75,0.8,0.85,0.9],[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,0.75,0.8,0.85,0.9],[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,0.75,0.8,0.85,0.9], [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,0.75,0.8,0.85,0.9]]

# M0_ints = [1482, 1288, 1064, 685, 366, 384, 179, 218, 115, 133]

parameters = {(tnogse, num_grad): {'alpha_1': [], 'C': [], 'C_error': [], 'chi': []} for tnogse in tnogses for num_grad in num_grads}

file_name = "levaduras_20240622"
A0 = "sin_A0"
exp = 1
n = 2
slic = 0
modelo = "Free-Rest"

D0_ext = 2.3e-12
D0_int = 0.7e-12
roi = "ROI1"
palette = [
"#1f77b4",  # Azul
"#9467bd",  # Púrpura
#"#e377c2",  # Rosa
#"#7f7f7f",  # Gris
#"#8c564b",  # Marrón
"#f1c40f",  # Amarillo
"#d62728",  # Rojo
]   
sns.set_palette(palette)

# for roi, color in zip(rois, palette):

for tnogse, gs in zip(tnogses, gss):

    for num_grad, g, alphas1, color in zip(num_grads, gs, alphass1, palette):

        fig, ax = plt.subplots(figsize=(8, 6)) 

        folder = f"fit_nogse_vs_x_free-rest_sinoffset_pt_alpha_M01=0.75_D02={D0_int}"

        directory = f"../results_{file_name}/{folder}/{A0}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}"
        os.makedirs(directory, exist_ok=True)

        data = np.loadtxt(f"../results_{file_name}/nogse_vs_x_data/slice={slic}/tnogse={tnogse}_g={g}_N={int(n)}_exp={exp}/{roi}_data_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.txt")

        x = data[:, 0]
        f = data[:, 1]
        vectores_combinados = zip(x, f)
        vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
        x, f = zip(*vectores_ordenados)

        fit_palette = sns.color_palette("tab20", len(alphas1))  # Colores para los ajustes

        for i, alpha1 in enumerate(alphas1):
            #def fit_nogse_vs_x_free_mixto_offset(TE, G, N, x, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2, C): #alpha es 1/alpha
            model = lmfit.Model(nogse.fit_nogse_vs_x_free_mixto_offset, independent_vars=["TE", "G", "N", "x", "D0_1", "D0_2"], param_names=["alpha_1", "M0_1", "tc_2", "alpha_2", "M0_2", "C"])
            #model.set_param_hint("M0_1", value=1000.0, min=0.0, max=5000.0, vary=1)
            model.set_param_hint("M0_2", value=1000.0, min=0.0, max=5000.0, vary=1)
            model.set_param_hint("M0_1", expr="(0.75/0.25)*M0_2")
            model.set_param_hint("tc_2", value=2.23, min=0.1, max=50.0, vary=0)
            model.set_param_hint("alpha_1", value=alpha1, min=0.0, max=1.0, vary=0)
            model.set_param_hint("alpha_2", value=0, min=0.0, max=1.0, vary=0)
            model.set_param_hint("C", value=0, min=0, max=5000.0, vary=0)
            params = model.make_params()

            result = model.fit(f, params, TE=float(tnogse), G=float(g), N=n, x=x, D0_1=D0_ext, D0_2=D0_int) 

            print(f"Ajuste {i + 1} para alpha_1 = {alpha1}")
            print(result.params.pretty_print())
            print(f"Chi cuadrado = {result.chisqr}")
            print(f"Reduced chi cuadrado = {result.redchi}")

            M01_fit = result.params["M0_1"].value
            M02_fit = result.params["M0_2"].value
            tc2_fit = result.params["tc_2"].value
            alpha2_fit = result.params["alpha_2"].value
            C_fit = result.params["C"].value
            M01_error = result.params["M0_1"].stderr
            M02_error = result.params["M0_2"].stderr
            tc2_error = result.params["tc_2"].stderr
            alpha2_error = result.params["alpha_2"].stderr
            C_error = result.params["C"].stderr

            #si t_c=0.1 pasar a la siguiente iteración 
            # if tc_fit < 0.15:
            #     continue

            # Guardar los valores ajustados de t_c y alpha para graficar más tarde
            parameters[(tnogse, num_grad)]['alpha_1'].append(alpha1)
            parameters[(tnogse, num_grad)]['C'].append(C_fit)
            parameters[(tnogse, num_grad)]['C_error'].append(C_error)
            parameters[(tnogse, num_grad)]['chi'].append(result.chisqr)

            x_fit = np.linspace(np.min(x), np.max(x), num=1000)
            fit = nogse.fit_nogse_vs_x_free_mixto_offset(tnogse, g, n, x_fit, alpha1, M01_fit, D0_ext, tc2_fit, alpha2_fit, M02_fit, D0_int, C_fit)

            with open(f"{directory}/parameters_tnogse={tnogse}_g={g}_N={int(n)}.txt", "a") as a:
                print(f"Ajuste {i + 1} - alpha = {alpha1}", file=a)
                print(roi,  " - M01 = ", M01_fit, "+-", M01_error, file=a)
                print("    ",  " - M02 = ", M02_fit, "+-", M02_error, file=a)
                print("    ",  " - tc2 = ", tc2_fit, "+-", tc2_error, file=a)
                print("    ",  " - C = ", C_fit, "+-", C_error, file=a)
                print("    ",  " - Chi cuadrado = ", result.chisqr, file=a)
                print("    ",  " - Reduced chi cuadrado = ", result.redchi, file=a)
                
            ax.plot(x, f, "o", markersize=7, linewidth=2, color = color)
            ax.plot(x_fit, fit, linewidth=2, label = alpha1, color = fit_palette[i])
            ax.legend(title = "$\\alpha$", title_fontsize=10, fontsize=10, loc='best')
            ax.set_xlabel("Tiempo de modulación x [ms]", fontsize=27)
            ax.set_ylabel("Señal $\mathrm{NOGSE}$", fontsize=27)
            ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
            ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
            ax.tick_params(axis='y', labelsize=16, color='black')
            title = ax.set_title(f"{modelo} | $T_\mathrm{{NOGSE}}$ = {tnogse} ms | {num_grad} = {g} mT/m | $N$ = {n} | slice = {slic} ", fontsize=15)

            # table = np.vstack((x_fit, fit))
            # np.savetxt(f"{directory}/{roi}_fit_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_tc={tc}.txt", table.T, delimiter=' ', newline='\n')
            #fig1, ax1 = plt.subplots(figsize=(8, 6)) 
            #nogse.plot_nogse_vs_x_fit(ax1, roi, modelo, x, x_fit, f, fit, tnogse, g, n, num_grad, slic, tc, fit_palette[i])
            # fig1.tight_layout()
            # fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_tc={tc}.pdf")
            # fig1.savefig(f"{directory}/{roi}_nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}_tc={tc}.png", dpi=600)
            # plt.close(fig1)

        fig.tight_layout()
        fig.savefig(f"../results_{file_name}/{folder}/{A0}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.pdf")
        fig.savefig(f"../results_{file_name}/{folder}/{A0}/nogse_vs_x_tnogse={tnogse}_g={g}_N={int(n)}.png", dpi=600)
        plt.close(fig)

for tnogse in tnogses:
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for num_grad, color in zip(num_grads, palette):
        C_values = parameters[(tnogse, num_grad)]['C']
        alpha_values = parameters[(tnogse, num_grad)]['alpha_1']
        ax2.plot(alpha_values, C_values, "-", linewidth = 2,  color=color, label=num_grad)
        #color del punto asociado al valor del chi cuadradado para ese ajuste con barra de de color al costado de la figura
        chi_values = parameters[(tnogse, num_grad)]['chi']
        ax2.scatter(alpha_values, C_values, c=chi_values, cmap='plasma', s=100, edgecolor='black', linewidth=0.5)

    cbar = plt.colorbar(ax2.scatter(alpha_values, C_values, c=chi_values, cmap='plasma', s=100, edgecolor='black', linewidth=0.5))
    cbar.set_label('Chi cuadrado', fontsize=15)
    cbar.ax.tick_params(labelsize=15)

    ax2.legend(title="Gradientes", title_fontsize=15, fontsize=15, loc='best')
    ax2.set_ylabel("Offset C", fontsize=27)
    ax2.set_xlabel("$\\alpha$", fontsize=27)
    ax2.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax2.tick_params(axis='x', rotation=0, labelsize=16, color='black')
    ax2.tick_params(axis='y', labelsize=16, color='black')
    title = ax2.set_title(f"{modelo} | $T_\\mathrm{{NOGSE}}$ = {tnogse} ms | $N$ = {n} | slice = {slic}", fontsize=15)

    fig2.tight_layout()
    fig2.savefig(f"../results_{file_name}/{folder}/{A0}/C_vs_alpha1_tnogse={tnogse}.pdf")
    fig2.savefig(f"../results_{file_name}/{folder}/{A0}/C_vs_alpha1_tnogse={tnogse}.png", dpi=600)
    plt.close(fig2)

palette = sns.color_palette("tab10", len(tnogses))
for num_grad in num_grads:
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for tnogse, color in zip(tnogses, palette):
        C_values = parameters[(tnogse, num_grad)]['C']
        alpha_values = parameters[(tnogse, num_grad)]['alpha_1']
        ax3.plot(alpha_values, C_values, "-", linewidth = 2, color=color, label=tnogse)
        #color del punto asociado al valor del chi cuadradado para ese ajuste con barra de de color al costado de la figura
        chi_values = parameters[(tnogse, num_grad)]['chi']
        ax3.scatter(alpha_values, C_values, c=chi_values, cmap='plasma', s=100, edgecolor='black', linewidth=0.5)
        
    cbar = plt.colorbar(ax3.scatter(alpha_values, C_values, c=chi_values, cmap='plasma', s=100, edgecolor='black', linewidth=0.5))
    cbar.set_label('Chi cuadrado', fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    
    ax3.legend(title="$T_\\mathrm{NOGSE}$ [ms]", title_fontsize=15, fontsize=15, loc='best')
    ax3.set_ylabel("Offset C", fontsize=27)
    ax3.set_xlabel("$\\alpha$", fontsize=27)
    ax3.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax3.tick_params(axis='x', rotation=0, labelsize=16, color='black')
    ax3.tick_params(axis='y', labelsize=16, color='black')
    title = ax3.set_title(f"{modelo} | {num_grad} | $N$ = {n} | slice = {slic}", fontsize=15)

    fig3.tight_layout()
    fig3.savefig(f"../results_{file_name}/{folder}/{A0}/C_vs_alpha1_G={num_grad}.pdf")
    fig3.savefig(f"../results_{file_name}/{folder}/{A0}/C_vs_alpha1_G={num_grad}.png", dpi=600)
    plt.close(fig3)
