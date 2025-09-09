# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, skew

st.set_page_config(page_title="Tendencia central: media, mediana y moda", layout="wide")

st.title("📊 Tendencia central: media, mediana y moda")
st.caption("Explora cómo cambian la media, la mediana y la moda según la forma de la distribución y el sesgo.")

# -----------------------------
# Controles
# -----------------------------
with st.sidebar:
    st.header("⚙️ Configuración de datos")
    dist = st.selectbox(
        "Tipo de distribución",
        [
            "Normal",
            "Uniforme",
            "Sesgada a la derecha",
            "Sesgada a la izquierda",
        ],
        index=0
    )

    n = st.slider("n (tamaño de muestra)", min_value=20, max_value=5000, value=500, step=10)

    if dist == "Normal":
        mu = st.number_input("Media (μ)", value=50.0, step=1.0)
        sigma = st.number_input("Desviación estándar (σ)", value=10.0, step=0.5, min_value=0.1)
    elif dist == "Uniforme":
        a = st.number_input("Mínimo (a)", value=0.0, step=1.0)
        b = st.number_input("Máximo (b)", value=100.0, step=1.0)
        if b <= a:
            st.warning("El máximo (b) debe ser mayor que el mínimo (a). Ajusta los valores.")
    elif dist in ["Sesgada a la derecha", "Sesgada a la izquierda"]:
        skew_intensity = st.slider(
            "Intensidad de sesgo (baja → alta)",
            min_value=0.1, max_value=1.5, value=0.6, step=0.05,
            help="Controla cuán asimétrica es la distribución."
        )
        # Centro objetivo solo para que no 'se vaya' muy lejos al graficar
        centro = st.number_input("Centro aproximado (para ubicar la masa principal)", value=50.0, step=1.0)

# -----------------------------
# Generación de datos
# -----------------------------
if dist == "Normal":
    data = np.random.normal(mu, sigma, n)

elif dist == "Uniforme":
    a_eff = a
    b_eff = b
    if b_eff <= a_eff:
        st.stop()
    data = np.random.uniform(a_eff, b_eff, n)

elif dist == "Sesgada a la derecha":
    # Lognormal con media log=0 y sigma=skew_intensity, luego reescalamos al 'centro' indicado
    raw = np.random.lognormal(mean=0.0, sigma=skew_intensity, size=n)
    # centramos cerca de 'centro' manteniendo forma (ajuste lineal a [centro-20, centro+20])
    lo, hi = np.percentile(raw, [5, 95])
    span = max(hi - lo, 1e-6)
    data = (raw - lo) / span * 40 + (centro - 20)

elif dist == "Sesgada a la izquierda":
    # Negativo de lognormal para sesgo a la izquierda, luego reescalamos alrededor de 'centro'
    raw = -np.random.lognormal(mean=0.0, sigma=skew_intensity, size=n)
    lo, hi = np.percentile(raw, [5, 95])
    span = max(hi - lo, 1e-6)
    data = (raw - lo) / span * 40 + (centro - 20)

data = np.array(data, dtype=float)
n_obs = len(data)
if n_obs == 0:
    st.error("No hay datos para mostrar.")
    st.stop()

# -----------------------------
# Cálculos de tendencia central
# -----------------------------
media = float(np.mean(data))
mediana = float(np.median(data))

# Moda aproximada para datos continuos por KDE (respaldo: histograma)
try:
    kde = gaussian_kde(data)
    xs = np.linspace(np.min(data), np.max(data), 1024)
    dens = kde(xs)
    moda_x = float(xs[np.argmax(dens)])
    moda_method = "KDE"
except Exception:
    counts, bins = np.histogram(data, bins="auto")
    centers = 0.5 * (bins[:-1] + bins[1:])
    moda_x = float(centers[np.argmax(counts)])
    moda_method = "Histograma"

# Sesgo (skewness) e IQR para contextualizar
sesgo = float(skew(data))
q1, q3 = np.percentile(data, [25, 75])
iqr = float(q3 - q1) if q3 > q1 else float(np.std(data))
diff_mm = media - mediana

# -----------------------------
# Gráfico
# -----------------------------
nbins = int(np.clip(np.sqrt(n_obs), 10, 80))  # regla ~sqrt(n)
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=data,
    nbinsx=nbins,
    histnorm="probability density",
    name="Frecuencia",
    opacity=0.6
))

# KDE si es posible
try:
    xs_plot = np.linspace(np.min(data), np.max(data), 512)
    kde_plot = gaussian_kde(data)
    dens_plot = kde_plot(xs_plot)
    fig.add_trace(go.Scatter(x=xs_plot, y=dens_plot, mode="lines", name="Densidad (KDE)"))
except Exception:
    pass

# Líneas verticales
fig.add_vline(x=media, line_width=2, line_dash="dash", annotation_text=f"Media: {media:.2f}", annotation_position="top")
fig.add_vline(x=mediana, line_width=2, line_dash="dash", annotation_text=f"Mediana: {mediana:.2f}", annotation_position="top")
fig.add_vline(x=moda_x, line_width=2, line_dash="dash", annotation_text=f"Moda*: {moda_x:.2f}", annotation_position="top")

fig.update_layout(
    bargap=0.05,
    title=f"Distribución — n = {n_obs}",
    xaxis_title="Valor",
    yaxis_title="Densidad",
    legend_title="Capas",
)

# -----------------------------
# Layout y métricas
# -----------------------------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📌 Estadísticos")
    st.metric("Media (x̄)", f"{media:,.3f}")
    st.metric("Mediana (Q2)", f"{mediana:,.3f}")
    st.metric("Moda* (aprox.)", f"{moda_x:,.3f}")
    st.metric("Sesgo (skewness)", f"{sesgo:,.3f}")
    st.caption("*En datos continuos la moda se estima por KDE o histograma (pico de mayor densidad).")

# -----------------------------
# Interpretación guiada (dinámica)
# -----------------------------
st.markdown("---")
st.subheader("🧠 ¿Qué nos dicen tus medidas *en estos datos*?")
interpretaciones = []

# Media vs Mediana
if abs(diff_mm) < 1e-9:
    interpretaciones.append("- **Media y mediana son prácticamente iguales** → la distribución parece bastante **simétrica**.")
elif diff_mm > 0:
    interpretaciones.append("- **Media > Mediana** → hay más **cola hacia valores grandes** (tendencia a **sesgo a la derecha**).")
else:
    interpretaciones.append("- **Media < Mediana** → hay más **cola hacia valores pequeños** (tendencia a **sesgo a la izquierda**).")

# Orden típico según sesgo
if sesgo > 0.1:
    interpretaciones.append("- Orden típico esperado con sesgo a la derecha: **Moda < Mediana < Media**.")
elif sesgo < -0.1:
    interpretaciones.append("- Orden típico esperado con sesgo a la izquierda: **Media < Mediana < Moda**.")
else:
    interpretaciones.append("- Orden típico con simetría: **Media ≈ Mediana ≈ Moda**.")

# Magnitud relativa de la diferencia media-mediana respecto al IQR
if iqr > 0:
    rel = abs(diff_mm) / iqr
    if rel < 0.05:
        interpretaciones.append(f"- La diferencia |Media−Mediana| es **muy pequeña** respecto al IQR ({rel:.2%}) → **asimetría leve**.")
    elif rel < 0.20:
        interpretaciones.append(f"- La diferencia |Media−Mediana| es **moderada** respecto al IQR ({rel:.2%}) → **asimetría moderada**.")
    else:
        interpretaciones.append(f"- La diferencia |Media−Mediana| es **grande** respecto al IQR ({rel:.2%}) → **asimetría marcada**.")

# Lectura conceptual de cada medida con sus valores
interpretaciones.append(
    f"- **Media ({media:.2f})**: punto de *equilibrio* que se ajusta a todos los valores; **sensible** a valores extremos."
)
interpretaciones.append(
    f"- **Mediana ({mediana:.2f})**: el **centro** (50% debajo y 50% arriba); **robusta** ante valores extremos."
)
interpretaciones.append(
    f"- **Moda aprox. ({moda_x:.2f})**: zona de **mayor frecuencia** observada (estimación por {moda_method})."
)

st.markdown("\n".join(interpretaciones))
