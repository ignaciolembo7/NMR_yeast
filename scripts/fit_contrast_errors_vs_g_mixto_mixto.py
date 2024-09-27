import numpy as np
import matplotlib.pyplot as plt
from protocols import nogse
import lmfit
import os
import seaborn as sns

# Configuración de seaborn
sns.set_theme(context='paper')
sns.set_style("whitegrid")

# Definir parámetros y rutas de archivo
file_name = "levaduras_20240613"
folder = "fit_contrast_errors_vs_g_mixto_mixto"
A0 = "con_A0"
D0_ext = 2.3e-12  # Extra
D0_int = 0.7e-12  # Intra
exp = 1  # Número de experimento
slic = 0  # Slice que quiero ver
modelo = "Mixto+Mixto"  # Nombre carpeta modelo libre/rest/tort
tnogse = float(input('Tnogse [ms]: '))  # Tiempo tnogse en ms
n = 2  # Valor de N
rois = ["ROI1"]  # Lista de ROIs

# Crear directorio para almacenar resultados si no existe
directory = f"../results_{file_name}/{folder}/{A0}/tnogse={tnogse}_N={int(n)}_exp={exp}"
os.makedirs(directory, exist_ok=True)

# Inicializar la figura
fig, ax = plt.subplots(figsize=(8, 6))
palette = sns.color_palette("tab10", len(rois))  # Generar una paleta de colores única

for roi, color in zip(rois, palette):

    # Crear subfiguras
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Cargar datos de contraste y errores
    data = np.loadtxt(f"../results_{file_name}/contrast_vs_g_data/{A0}/tnogse={tnogse}_N={int(n)}_exp={exp}/{roi}_data_contrast_vs_g_tnogse={tnogse}_N={int(n)}.txt")
    g = data[:, 0]  # Valores de gradiente
    f = data[:, 1]  # Valores de contraste observados
    f_err = data[:, 2]  # Errores en los valores observados (incertidumbre en f)

    # Ordenar los vectores por gradiente g
    vectores_combinados = zip(g, f, f_err)
    vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    g, f, f_err = zip(*vectores_ordenados)

    # Eliminar el primer valor si es necesario
    g = np.delete(g, [0])
    f = np.delete(f, [0])
    f_err = np.delete(f_err, [0])

    # Definir el modelo
    model = lmfit.Model(nogse.fit_contrast_vs_g_mixto_mixto, independent_vars=["TE", "G", "N", "D0_1", "D0_2"], param_names=["tc_1", "alpha_1", "M0_1", "tc_2", "alpha_2", "M0_2"])
    model.set_param_hint("M0_1", value=0.6, min=0, max=1.0, vary=1)
    model.set_param_hint("M0_2", value=0.4, min=0, max=1.0, vary=0)
    model.set_param_hint("tc_1", value=5.0, min=0, max=20.0, vary=1)
    model.set_param_hint("tc_2", value=2.1, min=0, max=10.0, vary=0)
    model.set_param_hint("alpha_1", value=0.2, min=0, max=1, vary=0)
    model.set_param_hint("alpha_2", value=0.0, min=0, max=1, vary=0)
    params = model.make_params()

    # Ajustar el modelo utilizando los errores (weights)
    result = model.fit(f, params, TE=float(tnogse), G=g, N=n, D0_1=D0_ext, D0_2=D0_int, weights=1/f_err)

    # Imprimir resultados del ajuste
    print(result.params.pretty_print())
    print(f"Chi cuadrado = {result.chisqr}")
    print(f"Reduced chi cuadrado = {result.redchi}")

    # Obtener parámetros ajustados y sus errores
    M01_fit = result.params["M0_1"].value
    error_M01 = result.params["M0_1"].stderr
    tc_1_fit = result.params["tc_1"].value
    error_tc_1 = result.params["tc_1"].stderr
    alpha_1_fit = result.params["alpha_1"].value
    error_alpha_1 = result.params["alpha_1"].stderr
    M02_fit = result.params["M0_2"].value
    error_M02 = result.params["M0_2"].stderr
    tc_2_fit = result.params["tc_2"].value
    error_tc_2 = result.params["tc_2"].stderr
    alpha_2_fit = result.params["alpha_2"].value
    error_alpha_2 = result.params["alpha_2"].stderr

    # Calcular el valor de chi-cuadrado y guardarlo
    chi_square = result.chisqr

    with open(f"{directory}/parameters_tnogse={tnogse}_N={int(n)}.txt", "a") as a:
        print(roi,  " - tc_1 = ", tc_1_fit, "+-", error_tc_1, file=a)
        print("     ",  " - alpha_1 = ", alpha_1_fit, "+-", error_alpha_1, file=a)
        print("     ",  " - M01 = ", M01_fit, "+-", error_M01, file=a)
        print("     ",  " - D0_ext = ", D0_ext, file=a)
        print("     ",  " - tc_2 = ", tc_2_fit, "+-", error_tc_2, file=a)
        print("     ",  " - alpha_2 = ", alpha_2_fit, "+-", error_alpha_2, file=a)
        print("     ",  " - M02 = ", M02_fit, "+-", error_M02, file=a)
        print("     ",  " - D0_int = ", D0_int, file=a)
        print("     ",  " - Chi cuadrado = ", chi_square, file=a)

    # Generar la curva de ajuste
    g_fit = np.linspace(np.min(g), np.max(g), num=1000)
    fit = nogse.fit_contrast_vs_g_mixto_mixto(tnogse, g_fit, n, tc_1_fit, alpha_1_fit, M01_fit, D0_ext, tc_2_fit, alpha_2_fit, M02_fit, D0_int)

    # Graficar y guardar
    nogse.plot_contrast_vs_g_fit(ax, roi, modelo, g, g_fit, f, fit, tnogse, n, slic, color)

    # Guardar los resultados en un archivo de texto
    table = np.vstack((g_fit, fit))
    np.savetxt(f"{directory}/{roi}_fit_contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.txt", table.T, delimiter=' ', newline='\n')

    fig1.tight_layout()
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.pdf")
    fig1.savefig(f"{directory}/{roi}_contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.png", dpi=600)
    plt.close(fig1)

fig.tight_layout()
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.pdf")
fig.savefig(f"{directory}/contrast_vs_g_tnogse={tnogse}_N={int(n)}_exp={exp}.png", dpi=600)
plt.close(fig)
