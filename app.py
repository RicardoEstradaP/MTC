# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, skew
import hashlib, json

st.set_page_config(page_title="Tendencia central: media, mediana y moda", layout="wide")

st.title("📊 Tendencia central: media, mediana y moda")
st.caption("Gráfico responsive: se ajusta al ancho disponible (ideal para incrustar en Moodle).")

# -----------------------------
# Sidebar: controles
# -----------------------------
with st.sidebar:
    st.header("⚙️ Configuración de datos")
    dist = st.selectbox(
        "Tipo de distribución",
        ["Normal", "Uniforme", "Sesgada a la derecha", "Sesgada a la izquierda"],
        index=0
    )
    n = st.slider("n (tamaño de muestra)", 20, 5000, 500, step=10)

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

# -----------------------------
# Utilidades (RNG determinista)
# -----------------------------
def rng_from_params(p: dict):
    s = json.dumps(p, sort_keys=True).encode()
    seed = int(hashlib.md5(s).hexdigest()[:8], 16)
    return np.random.default_rng(seed)

def generate_sample(p: dict) -> np.ndarray:
    r = rng_from_params(p)
    n = p["n"]
    if p["dist"] == "Normal":
        data = r.normal(p["mu"], p["sigma"], n)
    elif p["dist"] == "Uniforme":
        if p["b"] <= p["a"]:
            return np.array([])
        data = r.uniform(p["a"], p["b"], n)
    elif p["dist"] == "Sesgada a la derecha":
        raw = r.lognormal(mean=0.0, sigma=p["skew_intensity"], size=n)
        lo, hi = np.percentile(raw, [5, 95]); span = max(hi - lo, 1e-6)
        data = (raw - lo) / span * 40 + (p["centro"] - 20)
    else:  # Sesgada a la izquierda
        raw = -r.lognormal(mean=0.0, sigma=p["skew_intensity"], size=n)
        lo, hi = np.percentile(raw, [5, 95]); span = max(hi - lo, 1e-6)
        data = (raw - lo) / span * 40 + (p["centro"] - 20)
    return np.array(data, dtype=float)

# -----------------------------
# Datos y estadísticos
# -----------------------------
data = generate_sample(params)
if data.size == 0:
    st.stop()

media = float(np.mean(data))
mediana = float(np.median(data))

# Moda aproximada (KDE con respaldo histograma)
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
# Binning y ejes
# -----------------------------
nbins = int(np.clip(np.sqrt(data.size), 10, 80))
xmin, xmax = float(np.min(data)), float(np.max(data))
if xmax == xmin:  # evita rango cero
    xmax = xmin + 1.0

# Para estabilidad visual usamos densidad
hist_density, bin_edges = np.histogram(data, bins=nbins, density=True)
peak_hist = float(hist_density.max()) if len(hist_density) else 1.0

peak_kde = 0.0
try:
    xs_dense = np.linspace(xmin, xmax, 512)
    kde_for_peak = gaussian_kde(data)
    dens_for_peak = kde_for_peak(xs_dense)
    peak_kde = float(np.max(dens_for_peak))
except Exception:
    xs_dense, dens_for_peak = None, None

ymax = max(peak_hist, peak_kde) * 1.15
if ymax <= 0:
    ymax = 1.0

# -----------------------------
# Figura Plotly (responsive)
# -----------------------------
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=data,
    nbinsx=nbins,
    histnorm="probability density",
    name="Frecuencia",
    opacity=0.6
))
if xs_dense is not None:
    fig.add_trace(go.Scatter(x=xs_dense, y=dens_for_peak, mode="lines", name="Densidad (KDE)"))

# Líneas verticales con colores
fig.add_vline(x=media,   line_width=2, line_dash="dash", line_color="red")
fig.add_vline(x=mediana, line_width=2, line_dash="dash", line_color="blue")
fig.add_vline(x=moda_x,  line_width=2, line_dash="dash", line_color="green")

# Anotaciones dentro del área (para no salirse al hacer más chico)
fig.update_layout(
    autosize=True,  # 🔑 permite que Plotly/Streamlit lo haga responsive
    height=420,     # alto base (se ajusta un poco con el contenedor)
    bargap=0.05,
    title=f"Distribución — n = {data.size}",
    xaxis_title="Valor",
    yaxis_title="Densidad",
    margin=dict(l=50, r=50, t=60, b=50),
    paper_bgcolor="white",
    plot_bgcolor="white",
)
fig.update_xaxes(range=[xmin, xmax])       # mantenemos el mismo rango de datos
fig.update_yaxes(range=[0, ymax])          # y el mismo rango de densidad

# 🔑 Render responsive dentro de Streamlit (usa el ancho del contenedor)
st.plotly_chart(fig, use_container_width=True, theme=None)

# -----------------------------
# Métricas e interpretación
# -----------------------------
st.subheader("📌 Estadísticos")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Media",   f"{media:.2f}")
c2.metric("Mediana", f"{mediana:.2f}")
c3.metric("Moda*",   f"{moda_x:.2f}")
c4.metric("Sesgo",   f"{sesgo:.2f}")
st.caption("*En datos continuos la moda se estima por KDE o histograma (pico de mayor densidad).")

st.markdown("---")
st.subheader("🧠 Interpretación en estos datos")
diff_mm = media - mediana
msgs = []
if abs(diff_mm) < 1e-9:
    msgs.append("- Media y mediana son prácticamente iguales → distribución **simétrica**.")
elif diff_mm > 0:
    msgs.append("- Media > Mediana → cola hacia valores grandes (**sesgo a la derecha**).")
else:
    msgs.append("- Media < Mediana → cola hacia valores pequeños (**sesgo a la izquierda**).")

if sesgo > 0.1:
    msgs.append("- Orden típico: Moda < Mediana < Media.")
elif sesgo < -0.1:
    msgs.append("- Orden típico: Media < Mediana < Moda.")
else:
    msgs.append("- Orden típico: Media ≈ Mediana ≈ Moda.")

msgs.append(f"- **Media ({media:.2f})**: punto de equilibrio, sensible a valores extremos.")
msgs.append(f"- **Mediana ({mediana:.2f})**: centro, robusta ante valores extremos.")
msgs.append(f"- **Moda ({moda_x:.2f})**: zona de mayor frecuencia (estimada por {moda_method}).")

st.markdown("\n".join(msgs))
