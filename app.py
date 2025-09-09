# app.py
import streamlit as st
import numpy as np
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
        ["Normal", "Uniforme", "Sesgada a la derecha", "Sesgada a la izquierda"],
        index=0
    )
    n = st.slider("n (tamaño de muestra)", 20, 5000, 500, step=10)

    if dist == "Normal":
        mu = st.number_input("Media (μ)", value=50.0, step=1.0)
        sigma = st.number_input("Desviación estándar (σ)", value=10.0, step=0.5, min_value=0.1)
    elif dist == "Uniforme":
        a = st.number_input("Mínimo (a)", value=0.0, step=1.0)
        b = st.number_input("Máximo (b)", value=100.0, step=1.0)
    else:
        skew_intensity = st.slider("Intensidad de sesgo", 0.1, 1.5, 0.6, step=0.05)
        centro = st.number_input("Centro aproximado", value=50.0, step=1.0)

# -----------------------------
# Generación de datos
# -----------------------------
if dist == "Normal":
    data = np.random.normal(mu, sigma, n)
elif dist == "Uniforme":
    data = np.random.uniform(a, b, n)
elif dist == "Sesgada a la derecha":
    raw = np.random.lognormal(mean=0.0, sigma=skew_intensity, size=n)
    lo, hi = np.percentile(raw, [5, 95])
    span = max(hi - lo, 1e-6)
    data = (raw - lo) / span * 40 + (centro - 20)
else:  # Sesgada a la izquierda
    raw = -np.random.lognormal(mean=0.0, sigma=skew_intensity, size=n)
    lo, hi = np.percentile(raw, [5, 95])
    span = max(hi - lo, 1e-6)
    data = (raw - lo) / span * 40 + (centro - 20)

data = np.array(data, dtype=float)
xmin, xmax = float(np.min(data)), float(np.max(data))

# -----------------------------
# Estadísticos
# -----------------------------
media = float(np.mean(data))
mediana = float(np.median(data))
try:
    kde = gaussian_kde(data)
    xs = np.linspace(xmin, xmax, 1024)
    dens = kde(xs)
    moda_x = float(xs[np.argmax(dens)])
    moda_method = "KDE"
except Exception:
    counts, bins = np.histogram(data, bins="auto")
    centers = 0.5 * (bins[:-1] + bins[1:])
    moda_x = float(centers[np.argmax(counts)])
    moda_method = "Histograma"
sesgo = float(skew(data))

# -----------------------------
# Gráfico con ejes bloqueados
# -----------------------------
nbins = int(np.clip(np.sqrt(len(data)), 10, 80))
hist_density, _ = np.histogram(data, bins=nbins, density=True)
peak = float(hist_density.max())

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=data, nbinsx=nbins, histnorm="probability density",
    name="Frecuencia", opacity=0.6
))
fig.add_vline(x=media, line_width=2, line_dash="dash", line_color="red",
              annotation_text=f"Media: {media:.2f}", annotation_position="top")
fig.add_vline(x=mediana, line_width=2, line_dash="dash", line_color="blue",
              annotation_text=f"Mediana: {mediana:.2f}", annotation_position="top")
fig.add_vline(x=moda_x, line_width=2, line_dash="dash", line_color="green",
              annotation_text=f"Moda*: {moda_x:.2f}", annotation_position="top")

# Ejes fijos
fig.update_layout(
    width=900, height=520,
    bargap=0.05,
    title=f"Distribución — n = {len(data)}",
    xaxis_title="Valor",
    yaxis_title="Densidad",
)
fig.update_xaxes(range=[xmin, xmax])     # 🔒 bloquea eje X
fig.update_yaxes(range=[0, peak * 1.15])  # 🔒 bloquea eje Y

# -----------------------------
# Mostrar
# -----------------------------
col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(fig, use_container_width=False)
with col2:
    st.subheader("📌 Estadísticos")
    st.metric("Media", f"{media:.2f}")
    st.metric("Mediana", f"{mediana:.2f}")
    st.metric("Moda*", f"{moda_x:.2f}")
    st.metric("Sesgo", f"{sesgo:.2f}")
