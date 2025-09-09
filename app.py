# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, skew
import streamlit.components.v1 as components

st.set_page_config(page_title="Tendencia central: media, mediana y moda", layout="wide")

st.title("📊 Tendencia central: media, mediana y moda")
st.caption("Vista con tamaño fijo y scroll horizontal opcional para evitar recortes en Moodle/iframes.")

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

    # Permite ajustar el ancho sin editar código
    fig_w = st.slider("Ancho del gráfico (px)", 700, 1400, 980, step=10)
    fig_h = 520

    params = {"dist": dist, "n": int(n)}

    if dist == "Normal":
        mu = st.number_input("Media (μ)", value=50.0, step=1.0)
        sigma = st.number_input("Desviación estándar (σ)", value=10.0, step=0.5, min_value=0.1)
        params.update({"mu": float(mu), "sigma": float(sigma)})
    elif dist == "Uniforme":
        a = st.number_input("Mínimo (a)", value=0.0, step=1.0)
        b = st.number_input("Máximo (b)", value=100.0, step=1.0)
        if b <= a:
            st.warning("El máximo (b) debe ser mayor que el mínimo (a).")
        params.update({"a": float(a), "b": float(b)})
    else:
        skew_intensity = st.slider("Intensidad de sesgo (baja → alta)", 0.1, 1.5, 0.6, step=0.05)
        centro = st.number_input("Centro aproximado", value=50.0, step=1.0)
        params.update({"skew_intensity": float(skew_intensity), "centro": float(centro)})

    generar = st.button("🔄 Generar/Actualizar muestra")

# -----------------------------
# Generación de datos
# -----------------------------
def generate_sample(p):
    n = p["n"]
    if p["dist"] == "Normal":
        data = np.random.normal(p["mu"], p["sigma"], n)
    elif p["dist"] == "Uniforme":
        if p["b"] <= p["a"]:
            return np.array([])
        data = np.random.uniform(p["a"], p["b"], n)
    elif p["dist"] == "Sesgada a la derecha":
        raw = np.random.lognormal(mean=0.0, sigma=p["skew_intensity"], size=n)
        lo, hi = np.percentile(raw, [5, 95]); span = max(hi - lo, 1e-6)
        data = (raw - lo) / span * 40 + (p["centro"] - 20)
    else:  # Sesgada a la izquierda
        raw = -np.random.lognormal(mean=0.0, sigma=p["skew_intensity"], size=n)
        lo, hi = np.percentile(raw, [5, 95]); span = max(hi - lo, 1e-6)
        data = (raw - lo) / span * 40 + (p["centro"] - 20)
    return np.array(data, dtype=float)

if "data" not in st.session_state:
    st.session_state.data = np.array([])
if "last_params" not in st.session_state:
    st.session_state.last_params = None

if generar or st.session_state.data.size == 0:
    st.session_state.data = generate_sample(params)
    st.session_state.last_params = params.copy()

data = st.session_state.data
if data.size == 0:
    st.stop()

# -----------------------------
# Estadísticos
# -----------------------------
media = float(np.mean(data))
mediana = float(np.median(data))

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

sesgo = float(skew(data))

# -----------------------------
# Bins y ejes fijos
# -----------------------------
nbins = int(np.clip(np.sqrt(data.size), 10, 80))
xmin, xmax = float(np.min(data)), float(np.max(data))
if xmax == xmin:
    xmax = xmin + 1.0

bin_edges = np.linspace(xmin, xmax, nbins + 1)
hist_density, _ = np.histogram(data, bins=bin_edges, density=True)
peak_hist = float(hist_density.max()) if len(hist_density) else 1.0

# KDE pico (opcional)
peak_kde = 0.0
try:
    xs_dense = np.linspace(xmin, xmax, 1024)
    kde_for_peak = gaussian_kde(data)
    dens_for_peak = kde_for_peak(xs_dense)
    peak_kde = float(np.max(dens_for_peak))
except Exception:
    xs_dense, dens_for_peak = None, None

ymax = max(peak_hist, peak_kde) * 1.15
if ymax <= 0:
    ymax = 1.0

# -----------------------------
# Figura Plotly (no responsive)
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=data,
    histnorm="probability density",
    xbins=dict(start=xmin, end=xmax, size=(xmax - xmin) / nbins),
    name="Frecuencia",
    opacity=0.6
))
if xs_dense is not None:
    fig.add_trace(go.Scatter(x=xs_dense, y=dens_for_peak, mode="lines", name="Densidad (KDE)"))

# Líneas verticales (colores)
fig.add_vline(x=media, line_width=2, line_dash="dash", line_color="red")
fig.add_vline(x=mediana, line_width=2, line_dash="dash", line_color="blue")
fig.add_vline(x=moda_x, line_width=2, line_dash="dash", line_color="green")

# Anotaciones fuera del área de trazado (evita cortes)
annotations = [
    dict(x=media, y=1.02, xref="x", yref="paper", text=f"Media: {media:.2f}", showarrow=False, font=dict(size=12, color="red")),
    dict(x=mediana, y=1.06, xref="x", yref="paper", text=f"Mediana: {mediana:.2f}", showarrow=False, font=dict(size=12, color="blue")),
    dict(x=moda_x, y=1.10, xref="x", yref="paper", text=f"Moda*: {moda_x:.2f}", showarrow=False, font=dict(size=12, color="green")),
]

fig.update_layout(
    width=fig_w, height=fig_h, autosize=False,
    bargap=0.05,
    title=f"Distribución — n = {data.size}",
    xaxis_title="Valor", yaxis_title="Densidad",
    margin=dict(l=70, r=160, t=80, b=70),  # margen derecho amplio
    paper_bgcolor="white", plot_bgcolor="white",
    annotations=annotations,
)
fig.update_xaxes(range=[xmin, xmax], fixedrange=True)
fig.update_yaxes(range=[0, ymax], fixedrange=True)

# -----------------------------
# Render HTML fijo con scroll horizontal si hace falta
# -----------------------------
html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                   config={"responsive": False, "displaylogo": False})

outer_w = fig_w + 80   # ancho mínimo del contenedor interno
components.html(
    f"""
    <div style="width:100%; overflow-x:auto; margin:auto;">
      <div style="
        width:{outer_w}px;
        max-width:none;
        margin:0 auto;
        padding-right:40px;   /* espacio extra lado derecho */
      ">
        {html}
      </div>
    </div>
    """,
    height=fig_h + 120,
    scrolling=False
)

# -----------------------------
# Métricas e interpretación
# -----------------------------
st.subheader("📌 Estadísticos")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Media", f"{media:.2f}")
c2.metric("Mediana", f"{mediana:.2f}")
c3.metric("Moda*", f"{moda_x:.2f}")
c4.metric("Sesgo", f"{sesgo:.2f}")
st.caption("*En datos continuos la moda se estima por KDE o histograma.")
