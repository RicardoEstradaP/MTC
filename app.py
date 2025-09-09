import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, skew

st.set_page_config(page_title="Tendencia central: media, mediana y moda", layout="wide")

st.title("Tendencia central: media, mediana y moda — con sesgo y atipicos")
st.caption("Explora como cambian la media, la mediana y la moda dependiendo de la forma de la distribucion, el sesgo y la presencia de valores atipicos.")

with st.sidebar:
    st.header("Configuracion")
    data_mode = st.radio("Fuente de datos", ["Generar datos sinteticos", "Cargar archivo (CSV)"], index=0)
    np.random.seed(st.number_input("Semilla aleatoria", min_value=0, max_value=10000, value=42, step=1))

    if data_mode == "Generar datos sinteticos":
        dist = st.selectbox(
            "Tipo de distribucion",
            [
                "Normal",
                "Sesgada a la derecha (lognormal)",
                "Sesgada a la izquierda (lognormal invertida)",
                "Uniforme",
                "Bimodal (mixtura de 2 normales)",
                "Normal + atipico(s)"
            ],
            index=0
        )
        n = st.slider("n (tamano de muestra)", min_value=20, max_value=5000, value=500, step=10)

        if dist in ["Normal", "Normal + atipico(s)"]:
            mu = st.number_input("Media objetivo (mu)", value=50.0, step=1.0)
            sigma = st.number_input("Desviacion estandar (sigma)", value=10.0, step=1.0, min_value=0.1)
        elif dist in ["Sesgada a la derecha (lognormal)", "Sesgada a la izquierda (lognormal invertida)"]:
            mu = st.number_input("Parametro mu de la lognormal (en log-espacio)", value=3.7, step=0.1)
            sigma = st.number_input("Parametro sigma de la lognormal (en log-espacio)", value=0.5, step=0.05, min_value=0.05)
        elif dist == "Uniforme":
            a = st.number_input("Minimo (a)", value=0.0, step=1.0)
            b = st.number_input("Maximo (b)", value=100.0, step=1.0)
            if b <= a:
                st.warning("El maximo (b) debe ser mayor que el minimo (a). Ajusta los valores.")
        elif dist == "Bimodal (mixtura de 2 normales)":
            mu1 = st.number_input("Media 1 (mu1)", value=40.0, step=1.0)
            sigma1 = st.number_input("Desviacion 1 (sigma1)", value=8.0, step=0.5, min_value=0.1)
            mu2 = st.number_input("Media 2 (mu2)", value=65.0, step=1.0)
            sigma2 = st.number_input("Desviacion 2 (sigma2)", value=6.0, step=0.5, min_value=0.1)
            weight = st.slider("Peso de la componente 1 (w)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        # Atipicos solo si corresponde
        add_outliers = False
        if dist in ["Normal + atipico(s)"]:
            add_outliers = st.checkbox("Agregar atipico(s)", value=True)
            outlier_value = st.number_input("Valor de atipico", value=200.0, step=1.0)
            outlier_count = st.slider("Cantidad de atipicos", min_value=1, max_value=50, value=3, step=1)

    else:
        uploaded = st.file_uploader("Sube un CSV con una columna llamada 'valor' o selecciona la columna a usar.", type=["csv"])

# --- Generacion / carga de datos ---
data = None
source_msg = ""
if data_mode == "Generar datos sinteticos":
    if dist == "Normal":
        data = np.random.normal(mu, sigma, n)
    elif dist == "Normal + atipico(s)":
        base = np.random.normal(mu, sigma, n)
        if add_outliers:
            base = np.concatenate([base, np.full(outlier_count, outlier_value)])
        data = base
    elif dist == "Sesgada a la derecha (lognormal)":
        data = np.random.lognormal(mean=mu, sigma=sigma, size=n)
    elif dist == "Sesgada a la izquierda (lognormal invertida)":
        data = -np.random.lognormal(mean=mu, sigma=sigma, size=n)
        data = data - data.min()  # desplazar a positivos si se desea, pero aqui dejamos posibles negativos
    elif dist == "Uniforme":
        b_eff = b if 'b' in locals() else 100.0
        a_eff = a if 'a' in locals() else 0.0
        data = np.random.uniform(a_eff, b_eff, n)
    elif dist == "Bimodal (mixtura de 2 normales)":
        k = int(n * weight)
        comp1 = np.random.normal(mu1, sigma1, k)
        comp2 = np.random.normal(mu2, sigma2, n - k)
        data = np.concatenate([comp1, comp2])
    source_msg = f"Datos sinteticos • {dist}"
else:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'valor' in df.columns:
            series = df['valor'].dropna().astype(float)
        elif len(numeric_cols) >= 1:
            col = st.selectbox("Selecciona la columna numerica", numeric_cols, index=0, key="colsel")
            series = df[col].dropna().astype(float)
        else:
            st.error("El CSV no tiene columnas numericas. Agrega una columna 'valor' o numeros.")
            st.stop()
        data = series.values
        source_msg = f"Datos de archivo • {uploaded.name}"
    else:
        st.info("Sube un archivo CSV para continuar.")
        st.stop()

data = np.array(data, dtype=float)
n_obs = len(data)
if n_obs == 0:
    st.error("No hay datos para mostrar.")
    st.stop()

# --- Calculo de media, mediana, moda (estimada) ---
media = float(np.mean(data))
mediana = float(np.median(data))

# Moda para datos continuos: aproximamos con KDE o histograma
try:
    kde = gaussian_kde(data)
    xs = np.linspace(np.min(data), np.max(data), 1024)
    dens = kde(xs)
    moda_x = float(xs[np.argmax(dens)])
    moda_method = "KDE"
except Exception:
    # Respaldo: histograma
    counts, bins = np.histogram(data, bins="auto")
    centers = 0.5 * (bins[:-1] + bins[1:])
    moda_x = float(centers[np.argmax(counts)])
    moda_method = "Histograma"

sesgo = float(skew(data))

# --- Grafico interactivo ---
nbins = int(np.clip(np.sqrt(n_obs), 10, 80))  # regla simple: ~sqrt(n)
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=data,
    nbinsx=nbins,
    histnorm="probability density",
    name="Frecuencia",
    opacity=0.6
))

# Suavizado con KDE (si posible)
try:
    xs = np.linspace(np.min(data), np.max(data), 512)
    kde = gaussian_kde(data)
    dens = kde(xs)
    fig.add_trace(go.Scatter(x=xs, y=dens, mode="lines", name="Densidad (KDE)"))
except Exception:
    pass

# Lineas verticales de media, mediana, moda
fig.add_vline(x=media, line_width=2, line_dash="dash", annotation_text=f"Media: {media:.2f}", annotation_position="top")
fig.add_vline(x=mediana, line_width=2, line_dash="dash", annotation_text=f"Mediana: {mediana:.2f}", annotation_position="top")
fig.add_vline(x=moda_x, line_width=2, line_dash="dash", annotation_text=f"Moda*: {moda_x:.2f}", annotation_position="top")

fig.update_layout(
    bargap=0.05,
    title=f"Distribucion ({source_msg}) — n = {n_obs}",
    xaxis_title="Valor",
    yaxis_title="Densidad",
    legend_title="Capas",
)

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Estadisticos")
    st.metric("Media (x-bar)", f"{media:,.3f}")
    st.metric("Mediana (Q2)", f"{mediana:,.3f}")
    st.metric("Moda* (aprox.)", f"{moda_x:,.3f}")
    st.metric("Sesgo (skewness)", f"{sesgo:,.3f}")
    st.caption("*En datos continuos la moda se estima por KDE o histograma (pico de mayor densidad).")

    # Mensaje sobre el orden tipico segun el sesgo
    if sesgo > 0.1:
        st.info("Distribucion sesgada a la derecha (cola hacia valores grandes). Suele cumplirse: Moda < Mediana < Media.")
    elif sesgo < -0.1:
        st.info("Distribucion sesgada a la izquierda (cola hacia valores pequenos). Suele cumplirse: Media < Mediana < Moda.")
    else:
        st.info("Distribucion aproximadamente simetrica. Suele cumplirse: Media ~ Mediana ~ Moda.")

st.markdown("---")
with st.expander("Que nos dice cada medida?"):
    st.markdown(
        "- Media: punto de equilibrio que se ajusta a todos los valores. Es sensible a atipicos.\n"
        "- Mediana: el centro de los datos (50% abajo / 50% arriba). Es robusta a atipicos.\n"
        "- Moda: el valor (o zona) mas frecuente. En distribuciones multimodales puede haber varias modas.\n"
        "- Sesgo: indica la asimetria. Positivo -> cola a la derecha; negativo -> cola a la izquierda."
    )

with st.expander("Actividades sugeridas para clase"):
    st.markdown(
        "1) Genera una normal y agrega atipicos extremos. Compara como cambia la media vs. la mediana.\n"
        "2) Usa la opcion Bimodal y discute por que la moda puede ser multiple.\n"
        "3) Ajusta el sesgo con la lognormal e identifica el orden tipico entre moda, mediana y media.\n"
        "4) Carga un CSV de tu curso (una columna numerica) y analiza los resultados."
    )

st.caption("Hecho en Streamlit • Autor: Tu nombre • Licencia: MIT")