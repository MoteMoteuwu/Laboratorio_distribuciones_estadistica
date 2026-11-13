import hashlib
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from matplotlib.figure import Figure
from scipy import stats
from scipy.integrate import trapezoid
from scipy.stats import gaussian_kde

st.set_page_config(
    page_title="Laboratorio de distribuciones · Estadística I",
    page_icon="📊",
    layout="wide",
)

SALT = "estadistica1-2026"
TABLE = "ranking"
NAVY, TEAL, ORANGE = "#0f2747", "#1f7a8c", "#e8833a"

# ─────────────────────────── Banco de casos ───────────────────────────
CASES = {
    "C01": dict(
        title="Tiempo de ensamble de una unidad electrónica",
        area="Ingeniería de Métodos y Tiempos",
        context="En una planta automatizada de manufactura de equipos de precisión, la división de Ingeniería de Métodos lleva a cabo un estudio analítico sobre el tiempo de ciclo operativo. Se mide el tiempo (en minutos) que toma a un técnico completar el ensamble completo de una unidad de control tras la estandarización del proceso. Las observaciones se recolectaron durante condiciones de estabilidad operativa sin interrupciones técnicas.",
        variable="Tiempo de ensamble (min)",
        kind="normal", n=300, p={"mu": 18.0, "sigma": 2.4}, seed=8121,
    ),
    "C02": dict(
        title="Duración de reparaciones en equipos críticos",
        area="Gestión del Mantenimiento Industrial",
        context="El departamento de Gestión de Mantenimiento registra la duración (en horas) de las intervenciones correctivas no programadas efectuadas sobre el sistema principal de bombeo industrial. La variabilidad observada presenta una concentración inicial en fallas menores de rápida solución, junto con una cola superior extendida originada por diagnósticos complejos y sustitución de piezas de gran envergadura.",
        variable="Duración de reparación (h)",
        kind="lognormal", n=500, p={"mu_log": 1.15, "sigma_log": 0.55}, seed=9284,
    ),
    "C03": dict(
        title="Vida útil de componentes mecánicos",
        area="Ingeniería de Confiabilidad y Calidad",
        context="En la unidad de Ensayos Destructivos y Confiabilidad, se evalúa el tiempo transcurrido hasta el fallo (en horas de trabajo continuo) de rodamientos de alta precisión sometidos a un protocolo de fatiga acelerada. La recolección empírica busca caracterizar la tasa de falla del componente crítico previa a su implementación masiva.",
        variable="Tiempo hasta la falla (h)",
        kind="weibull", n=400, p={"shape": 2.4, "scale": 1200.0}, seed=10473,
    ),
    "C04": dict(
        title="Intervalo entre llegadas a la estación de control",
        area="Control Estadístico de Procesos",
        context="Dentro del flujo logístico interno de una planta metalmecánica, se evalúa el tiempo (en minutos) transcurrido entre arribos consecutivos de piezas moldeadas a la celda de inspección dimensional automatizada. Se asume una tasa constante de flujo operacional y eventos aleatorios independientes en el tiempo.",
        variable="Intervalo de llegada (min)",
        kind="exponential", n=350, p={"scale": 4.8}, seed=11762,
    ),
    "C05": dict(
        title="Consumo diario de insumos industriales",
        area="Administración de Inventarios y Logística",
        context="La sección de Control de Inventarios realiza el seguimiento del consumo diario (en kilogramos) de polímero granulado consumido en la máquina de moldeo por inyección. Las variaciones registradas se encuentran acotadas dentro de un rango estrictamente definido por los límites de capacidad teórica y lotes de producción estándar.",
        variable="Consumo diario (kg)",
        kind="uniform", n=450, p={"low": 22.0, "high": 38.0}, seed=12991,
    ),
    "C06": dict(
        title="Duración de una actividad de montaje en proyectos",
        area="Planeación y Programación de Proyectos",
        context="La oficina de Gestión de Proyectos analiza la duración (en días hábiles) de la actividad de montaje e instalación de líneas de producción en proyectos de ampliación de planta. Los registros históricos muestran un valor mínimo y un valor máximo claramente delimitados por las condiciones contractuales y técnicas, con una duración más frecuente desplazada hacia el extremo inferior del rango.",
        variable="Duración de la actividad (días)",
        kind="triangular", n=380, p={"low": 4.0, "mode": 7.0, "high": 13.0}, seed=13457,
    ),
    "C07": dict(
        title="Peso neto de llenado de empaques",
        area="Control de Calidad en Procesos de Envasado",
        context="En una línea de envasado de alimentos, el área de Aseguramiento de Calidad registra el peso neto (en gramos) de los empaques producidos por una llenadora volumétrica recientemente calibrada. Las desviaciones respecto al valor objetivo se atribuyen a la acumulación de múltiples fuentes pequeñas e independientes de variación (densidad del producto, vibración y tiempos de dosificación).",
        variable="Peso neto (g)",
        kind="normal", n=420, p={"mu": 502.0, "sigma": 3.5}, seed=14821,
    ),
    "C08": dict(
        title="Tiempo de entrega de proveedores",
        area="Gestión de la Cadena de Suministro",
        context="El departamento de Compras estudia el tiempo (en días calendario) transcurrido entre la emisión de una orden de compra y la recepción del material en el almacén central. Los tiempos son estrictamente positivos; la mayoría de las entregas se concentra en pocos días, pero se presentan retrasos ocasionales prolongados asociados a trámites aduaneros y a la disponibilidad de transporte.",
        variable="Tiempo de entrega (días)",
        kind="lognormal", n=360, p={"mu_log": 2.0, "sigma_log": 0.4}, seed=15903,
    ),
    "C09": dict(
        title="Vida útil de herramientas de corte",
        area="Ingeniería de Manufactura",
        context="En un centro de mecanizado CNC, el área de Ingeniería de Manufactura registra el tiempo de corte efectivo (en minutos) que soportan los insertos de carburo antes de alcanzar el criterio de desgaste que obliga a su reemplazo. El mecanismo de desgaste progresivo hace que la probabilidad de reemplazo aumente con el tiempo de uso acumulado.",
        variable="Vida de la herramienta (min)",
        kind="weibull", n=330, p={"shape": 1.7, "scale": 95.0}, seed=16388,
    ),
    "C10": dict(
        title="Tiempo entre fallas de una banda transportadora",
        area="Mantenimiento Basado en Confiabilidad",
        context="El equipo de Confiabilidad de un centro de distribución registra las horas de operación transcurridas entre paradas no programadas de la banda transportadora principal. Las fallas se originan en causas externas aleatorias (atascos, objetos extraños, cortes de energía) que ocurren de forma independiente y sin efecto de envejecimiento apreciable del equipo.",
        variable="Tiempo entre fallas (h)",
        kind="exponential", n=300, p={"scale": 36.0}, seed=17542,
    ),
}

LABEL = {
    "normal": "Normal", "lognormal": "Lognormal", "weibull": "Weibull",
    "exponential": "Exponencial", "uniform": "Uniforme", "triangular": "Triangular",
}

# (clave, etiqueta, valor mínimo)
PARAMS = {
    "normal": [("mu", "Media μ", None), ("sigma", "Desviación estándar σ", 0.1)],
    "lognormal": [("mu_log", "Media logarítmica μ_log", None), ("sigma_log", "Desviación logarítmica σ_log", 0.1)],
    "weibull": [("shape", "Forma k", 0.1), ("scale", "Escala λ", 0.1)],
    "exponential": [("scale", "Escala (media) β", 0.1)],
    "uniform": [("low", "Límite inferior a", None), ("high", "Límite superior b", None)],
    "triangular": [("low", "Límite inferior a", None), ("mode", "Moda c", None), ("high", "Límite superior b", None)],
}

# Valores iniciales estándar de cada familia (no dependen de la muestra)
STD_DEFAULTS = {
    "normal": [0.0, 1.0],
    "lognormal": [0.0, 1.0],
    "weibull": [1.0, 1.0],
    "exponential": [1.0],
    "uniform": [0.0, 1.0],
    "triangular": [0.0, 0.5, 1.0],
}


# ─────────────────────────── Lógica estadística ───────────────────────────
@st.cache_data(show_spinner=False)
def sample(cid: str) -> np.ndarray:
    c = CASES[cid]
    r, p, n, k = np.random.default_rng(c["seed"]), c["p"], c["n"], c["kind"]
    if k == "normal":
        x = r.normal(p["mu"], p["sigma"], n)
    elif k == "lognormal":
        x = r.lognormal(p["mu_log"], p["sigma_log"], n)
    elif k == "weibull":
        x = p["scale"] * r.weibull(p["shape"], n)
    elif k == "exponential":
        x = r.exponential(p["scale"], n)
    elif k == "triangular":
        x = r.triangular(p["low"], p["mode"], p["high"], n)
    else:
        x = r.uniform(p["low"], p["high"], n)
    return np.round(x, 2)


def assigned(code: str) -> list[str]:
    d = hashlib.sha256((SALT + code).encode()).digest()
    r = np.random.default_rng(int.from_bytes(d[:8], "big"))
    return sorted(r.choice(list(CASES), 2, replace=False).tolist())


def validate(d: str, p: dict) -> str | None:
    if not all(np.isfinite(list(p.values()))):
        return "Hay valores no numéricos."
    if d == "normal" and p["sigma"] <= 0:
        return "σ debe ser mayor que 0."
    if d == "lognormal" and p["sigma_log"] <= 0:
        return "σ_log debe ser mayor que 0."
    if d == "weibull" and (p["shape"] <= 0 or p["scale"] <= 0):
        return "k y λ deben ser mayores que 0."
    if d == "exponential" and p["scale"] <= 0:
        return "β debe ser mayor que 0."
    if d == "uniform" and p["high"] <= p["low"]:
        return "El límite superior b debe ser mayor que a."
    if d == "triangular" and not (p["low"] < p["high"] and p["low"] <= p["mode"] <= p["high"]):
        return "Debe cumplirse a ≤ c ≤ b, con a < b."
    return None


def frozen(d: str, p: dict):
    if d == "normal":
        return stats.norm(loc=p["mu"], scale=p["sigma"])
    if d == "lognormal":
        return stats.lognorm(s=p["sigma_log"], scale=np.exp(p["mu_log"]))
    if d == "weibull":
        return stats.weibull_min(c=p["shape"], scale=p["scale"])
    if d == "exponential":
        return stats.expon(scale=p["scale"])
    if d == "uniform":
        return stats.uniform(loc=p["low"], scale=p["high"] - p["low"])
    w = p["high"] - p["low"]
    return stats.triang(c=(p["mode"] - p["low"]) / w, loc=p["low"], scale=w)


def overlap(dist_a, dist_b) -> float:
    """Coeficiente de solapamiento OVL = ∫ min(f, g) dx, en [0, 1].
    La malla se construye con los cuantiles de ambas distribuciones,
    así queda densa donde cada una concentra su masa."""
    q = np.linspace(1e-7, 1 - 1e-7, 20001)
    with np.errstate(all="ignore"):
        pts = np.concatenate([dist_a.ppf(q), dist_b.ppf(q)])
        pts = np.unique(pts[np.isfinite(pts)])
        f = np.nan_to_num(dist_a.pdf(pts), nan=0.0, posinf=0.0, neginf=0.0)
        g = np.nan_to_num(dist_b.pdf(pts), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.clip(trapezoid(np.minimum(f, g), pts), 0.0, 1.0))


def score_case(cid: str, d: str, p: dict) -> float:
    c = CASES[cid]
    return round(100 * overlap(frozen(c["kind"], c["p"]), frozen(d, p)), 1)


def fmt_params(d: str, p: dict) -> str:
    return " · ".join(f"{lbl.split()[-1]} = {p[k]:.1f}" for k, lbl, _ in PARAMS[d])


def short(cid: str) -> str:
    return f"C{int(cid[1:])}" if cid and cid[1:].isdigit() else str(cid)


# ─────────────────────────── Gráficas ───────────────────────────
def _style(ax, title, xl, yl):
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=NAVY, loc="left", pad=8)
    ax.set_xlabel(xl, fontsize=8.5, color="#4a5a70")
    ax.set_ylabel(yl, fontsize=8.5, color="#4a5a70")
    ax.tick_params(labelsize=8, colors="#4a5a70")
    ax.grid(True, alpha=0.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9d3df")


def fig_density(x, dist, var):
    fig = Figure(figsize=(7, 3.0), dpi=110)
    ax = fig.subplots()
    lo, hi = float(x.min()), float(x.max())
    pad = 0.04 * (hi - lo)
    xx = np.linspace(lo - pad, hi + pad, 400)
    counts, _, _ = ax.hist(x, bins="fd", density=True, color=TEAL, alpha=0.18,
                           edgecolor="white", label="Histograma")
    top = float(counts.max())
    try:
        k = gaussian_kde(x)(xx)
        ax.plot(xx, k, color=NAVY, lw=2, label="KDE muestral")
        top = max(top, float(k.max()))
    except Exception:
        pass
    with np.errstate(all="ignore"):
        y = dist.pdf(xx)
    y = np.where(np.isfinite(y), y, np.nan)
    ax.plot(xx, y, color=ORANGE, lw=2.2, ls="--", label="Modelo candidato")
    if np.all(np.isnan(y)) or np.nanmax(y) > top * 1.8:
        ax.set_ylim(0, top * 1.8)
    _style(ax, "Densidad teórica vs. KDE observada", var, "Densidad")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def fig_qq(x, dist):
    fig = Figure(figsize=(7, 2.8), dpi=110)
    ax = fig.subplots()
    n = len(x)
    q = (np.arange(1, n + 1) - 0.5) / n
    sx = np.sort(x)
    with np.errstate(all="ignore"):
        tq = dist.ppf(q)
    m = np.isfinite(tq)
    ax.scatter(tq[m], sx[m], s=10, alpha=0.55, color=TEAL, edgecolors="none")
    if m.sum() > 1:
        lo, hi = min(tq[m].min(), sx.min()), max(tq[m].max(), sx.max())
        ax.plot([lo, hi], [lo, hi], ls="--", color=ORANGE, lw=1.8)
    _style(ax, "Gráfico Q-Q", "Cuantiles teóricos", "Cuantiles observados")
    fig.tight_layout()
    return fig


# ─────────────────────────── Supabase ───────────────────────────
def sb_conf():
    try:
        url = str(st.secrets.get("SUPABASE_URL", "")).strip().rstrip("/")
        key = str(st.secrets.get("SUPABASE_KEY", "")).strip()
    except Exception:
        return None
    if not url or not key:
        return None
    if url.endswith("/rest/v1"):
        url = url[: -len("/rest/v1")]
    return url, key


def sb_headers(key: str, prefer: str | None = None) -> dict:
    h = {"apikey": key, "Content-Type": "application/json"}
    if key.startswith("eyJ"):  # claves anon/service_role antiguas (JWT)
        h["Authorization"] = f"Bearer {key}"
    if prefer:
        h["Prefer"] = prefer
    return h


@st.cache_data(ttl=30, show_spinner=False)
def fetch_submission(code: str):
    url, key = sb_conf()
    r = requests.get(f"{url}/rest/v1/{TABLE}", headers=sb_headers(key),
                     params={"select": "*", "student_code": f"eq.{code}", "limit": "1"}, timeout=15)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    rows = r.json()
    return rows[0] if rows else None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_ranking():
    url, key = sb_conf()
    r = requests.get(
        f"{url}/rest/v1/{TABLE}", headers=sb_headers(key),
        params={"select": "student_code,student_name,score,score_1,score_2,case_1,case_2,submitted_at",
                "order": "score.desc,submitted_at.asc", "limit": "1000"},
        timeout=15,
    )
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def get_submission(code):
    if sb_conf() is None:
        return st.session_state.local_rows.get(code), None
    try:
        return fetch_submission(code), None
    except Exception as e:
        return None, str(e)


def load_ranking():
    if sb_conf() is None:
        return pd.DataFrame(list(st.session_state.local_rows.values())), None
    try:
        return pd.DataFrame(fetch_ranking()), None
    except Exception as e:
        return pd.DataFrame(), str(e)


def insert_submission(row: dict):
    conf = sb_conf()
    if conf is None:
        if row["student_code"] in st.session_state.local_rows:
            return "duplicate", ""
        st.session_state.local_rows[row["student_code"]] = {
            **row, "submitted_at": datetime.now(timezone.utc).isoformat()}
        return "ok", ""
    url, key = conf
    try:
        r = requests.post(f"{url}/rest/v1/{TABLE}", headers=sb_headers(key, "return=minimal"),
                          json=row, timeout=15)
    except Exception as e:
        return "error", f"No fue posible conectar con Supabase: {e}"
    if r.status_code in (200, 201, 204):
        return "ok", ""
    if r.status_code == 409 or "23505" in r.text:
        return "duplicate", ""
    return "error", f"HTTP {r.status_code}: {r.text[:300]}"


def clear_caches():
    fetch_submission.clear()
    fetch_ranking.clear()


# ─────────────────────────── Estilos ───────────────────────────
st.markdown("""
<style>
.block-container{max-width:1200px;padding-top:2rem}
.hero{padding:1.6rem 2rem;border-radius:20px;color:#fff;margin-bottom:1.2rem;
  background:linear-gradient(120deg,#0f2747 0%,#1f5f7a 60%,#1f7a8c 100%);
  box-shadow:0 10px 30px rgba(15,39,71,.18)}
.hero h1{font-size:1.9rem;margin:0;color:#fff;padding:0}
.hero p{margin:.35rem 0 0;color:#d6e7ef}
.badge{display:inline-block;padding:.15rem .7rem;border-radius:999px;background:#e3f1f4;
  color:#1f7a8c;font-weight:600;font-size:.78rem;margin-bottom:.3rem}
.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:.5rem;margin:.4rem 0 1rem}
.stat{background:#f6f8fb;border:1px solid #e6ebf2;border-radius:12px;padding:.5rem .7rem}
.stat span{display:block;font-size:.68rem;color:#5b6b80;text-transform:uppercase;letter-spacing:.04em}
.stat b{font-size:1rem;color:#0f2747}
.total{background:linear-gradient(120deg,#0f2747,#1f7a8c);color:#fff;border-radius:18px;
  padding:1.3rem 1.6rem;margin-bottom:1rem}
.total .big{font-size:2.6rem;font-weight:700;line-height:1}
.total small{color:#d6e7ef}
@media (max-width:800px){.stats{grid-template-columns:repeat(3,1fr)}}
</style>
""", unsafe_allow_html=True)

st.session_state.setdefault("local_rows", {})
connected = sb_conf() is not None

st.markdown("""
<div class="hero">
  <h1>📊 Laboratorio de distribuciones</h1>
  <p>Estadística I · Ingeniería Industrial · Modelación teórica y ajuste empírico</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── Barra lateral ───────────────────────────
with st.sidebar:
    st.markdown("### 👤 Identificación")
    name = re.sub(r"\s+", " ", st.text_input("Nombre completo", placeholder="Ej. Carlos Ramírez", max_chars=80)).strip()
    code = st.text_input("Código estudiantil", placeholder="Ej. 2194852", max_chars=12).strip()

    name_ok = len(name) >= 3
    code_ok = re.fullmatch(r"\d{4,12}", code) is not None
    if name and not name_ok:
        st.error("Escribe tu nombre completo.")
    if code and not code_ok:
        st.error("El código debe contener solo dígitos (entre 4 y 12).")
    st.caption("Tus casos se asignan según tu código. Verifícalo bien: solo se permite **un envío**.")

    st.divider()
    if connected:
        st.success("Resultados guardados en la nube.", icon="☁️")
    else:
        st.warning("No se encontraron credenciales de Supabase. Los envíos se guardan solo en esta sesión del navegador.", icon="💾")

identified = name_ok and code_ok
ids = assigned(code) if identified else []

if st.session_state.pop("just_submitted", False):
    st.balloons()
    st.toast("¡Envío registrado con éxito!", icon="✅")

tab_act, tab_rank, tab_guide = st.tabs(["🧪 Actividad", "🏆 Ranking", "📘 Guía"])


# ─────────────────────────── Componentes ───────────────────────────
def render_case(ix: int, cid: str) -> dict:
    c, x = CASES[cid], sample(cid)
    with st.container(border=True):
        st.markdown(f'<span class="badge">Caso {ix} · {short(cid)}</span>', unsafe_allow_html=True)
        st.markdown(f"#### {c['title']}")
        st.caption(f"{c['area']} · Variable: {c['variable']}")
        st.write(c["context"])

        st.markdown(f"""
        <div class="stats">
          <div class="stat"><span>n</span><b>{len(x)}</b></div>
          <div class="stat"><span>Media</span><b>{np.mean(x):.2f}</b></div>
          <div class="stat"><span>Desv. est.</span><b>{np.std(x, ddof=1):.2f}</b></div>
          <div class="stat"><span>Mínimo</span><b>{x.min():.2f}</b></div>
          <div class="stat"><span>Máximo</span><b>{x.max():.2f}</b></div>
          <div class="stat"><span>Asimetría</span><b>{stats.skew(x):.2f}</b></div>
        </div>""", unsafe_allow_html=True)

        with st.expander("📄 Ver o descargar la muestra"):
            df = pd.DataFrame({c["variable"]: x})
            st.dataframe(df, height=200, width="stretch")
            st.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                               f"{cid}_muestra.csv", "text/csv", key=f"dl_{cid}")

        left, right = st.columns([1, 1.6], gap="large")
        with left:
            d = st.selectbox("Distribución candidata", list(LABEL), format_func=LABEL.get, key=f"dist_{cid}")
            st.markdown("**Parámetros** · una cifra decimal")
            st.caption("Los valores iniciales son los estándar de la familia: ajústalos a los datos.")
            p = {}
            for (k, lbl, mn), dv in zip(PARAMS[d], STD_DEFAULTS[d]):
                val = st.number_input(lbl, value=float(dv), min_value=mn, step=0.1,
                                      format="%.1f", key=f"{cid}_{d}_{k}")
                p[k] = round(float(val), 1)
            err = validate(d, p)
            if err:
                st.warning(err, icon="⚠️")
            else:
                st.caption("✅ Parámetros válidos. Observa el ajuste en las gráficas.")
        with right:
            if err:
                st.info("Corrige los parámetros para ver las gráficas.")
            else:
                dist = frozen(d, p)
                st.pyplot(fig_density(x, dist, c["variable"]))
                st.pyplot(fig_qq(x, dist))
    return {"dist": d, "params": p, "valid": err is None}


def render_result(sub: dict):
    when = str(sub.get("submitted_at", ""))[:16].replace("T", " ")
    total = float(sub["score"])
    st.markdown(f"""
    <div class="total">
      <small>Envío registrado · {sub.get('student_name', '')} · {when} UTC</small>
      <div class="big">{total:.1f} <span style="font-size:1.1rem;font-weight:400">/ 200</span></div>
      <small>Tu respuesta quedó guardada y ya no puede modificarse.</small>
    </div>""", unsafe_allow_html=True)

    detail = sub.get("detail") or {}
    cols = st.columns(2, gap="large")
    for col, cid, pts in zip(cols, (sub["case_1"], sub["case_2"]), (sub["score_1"], sub["score_2"])):
        pts = float(pts)
        with col, st.container(border=True):
            st.markdown(f'<span class="badge">{short(cid)}</span>', unsafe_allow_html=True)
            st.markdown(f"**{CASES[cid]['title']}**")
            ans = detail.get(cid, {})
            if ans:
                st.caption(f"{LABEL.get(ans['dist'], ans['dist'])} · {fmt_params(ans['dist'], ans['params'])}")
            st.metric("Solapamiento con el modelo real", f"{pts:.1f} / 100")
            st.progress(min(max(pts / 100, 0.0), 1.0))
    st.info("Consulta tu posición en la pestaña **🏆 Ranking**.")


def do_submit(answers: dict):
    with st.spinner("Calculando y registrando tu envío…"):
        scores = {cid: score_case(cid, a["dist"], a["params"]) for cid, a in answers.items()}
        row = {
            "student_code": code,
            "student_name": name,
            "case_1": ids[0], "case_2": ids[1],
            "score_1": scores[ids[0]], "score_2": scores[ids[1]],
            "score": round(scores[ids[0]] + scores[ids[1]], 1),
            "detail": {cid: {"dist": answers[cid]["dist"], "params": answers[cid]["params"],
                             "ovl": scores[cid]} for cid in ids},
        }
        status, msg = insert_submission(row)
    if status == "ok":
        clear_caches()
        st.session_state.just_submitted = True
        st.rerun()
    elif status == "duplicate":
        clear_caches()
        st.warning("Ya existe un envío registrado con este código. Solo se permite uno por estudiante.")
    else:
        st.error(f"No se pudo registrar el envío. Detalle técnico: {msg}")


# ─────────────────────────── Pestaña: Actividad ───────────────────────────
with tab_act:
    if not identified:
        st.markdown("### Bienvenido 👋")
        st.write("Ingresa tu **nombre** y tu **código** en el panel lateral para habilitar tus dos casos asignados.")
        c1, c2, c3 = st.columns(3)
        c1.info("**1. Analiza** el contexto y la muestra de cada caso.", icon="🔎")
        c2.info("**2. Ajusta** la distribución y sus parámetros con las gráficas.", icon="🎯")
        c3.info("**3. Envía** tus dos casos. Solo tienes un intento.", icon="📤")
    else:
        sub, err = get_submission(code)
        if err:
            st.error(f"No se pudo consultar Supabase. Revisa la tabla y los permisos.\n\nDetalle: {err}")
        elif sub:
            render_result(sub)
        else:
            st.markdown(f"### Tus casos, {name.split()[0]}")
            st.caption("Las gráficas se actualizan al instante con cada cambio de parámetros.")
            answers = {cid: render_case(i, cid) for i, cid in enumerate(ids, 1)}

            st.markdown("### 📤 Envío final")
            with st.container(border=True):
                st.dataframe(pd.DataFrame([{
                    "Caso": f"{short(cid)} · {CASES[cid]['title']}",
                    "Distribución": LABEL[a["dist"]],
                    "Parámetros": fmt_params(a["dist"], a["params"]),
                    "Estado": "✅ Listo" if a["valid"] else "⚠️ Revisar",
                } for cid, a in answers.items()]), hide_index=True, width="stretch")

                all_valid = all(a["valid"] for a in answers.values())
                if not all_valid:
                    st.error("Hay parámetros inválidos. Corrígelos antes de enviar.")
                confirm = st.checkbox("Revisé mis dos casos y entiendo que **no podré modificar** mi respuesta.",
                                      key=f"confirm_{code}")
                if st.button("Enviar mis dos casos", type="primary", width="stretch",
                             disabled=not (confirm and all_valid)):
                    do_submit(answers)

# ─────────────────────────── Pestaña: Ranking ───────────────────────────
with tab_rank:
    h1, h2 = st.columns([4, 1], vertical_alignment="bottom")
    h1.markdown("### Clasificación general")
    if h2.button("🔄 Actualizar", width="stretch"):
        clear_caches()
        st.rerun()

    df, err = load_ranking()
    if err:
        st.error(f"No se pudo cargar el ranking. Detalle: {err}")
    elif df.empty:
        st.info("Aún no hay envíos registrados.")
    else:
        for col in ("score", "score_1", "score_2"):
            df[col] = pd.to_numeric(df[col], errors="coerce").round(1)
        df = df.sort_values(["score", "submitted_at"], ascending=[False, True]).reset_index(drop=True)
        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        df["pos"] = range(1, len(df) + 1)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Participantes", len(df))
        m2.metric("Promedio", f"{df['score'].mean():.1f}")
        m3.metric("Máximo", f"{df['score'].max():.1f}")
        mine = df.index[df["student_code"].astype(str) == code].tolist() if identified else []
        m4.metric("Tu posición", f"#{mine[0] + 1}" if mine else "—")

        view = pd.DataFrame({
            "Pos.": [f"{medals.get(p, '')} {p}".strip() for p in df["pos"]],
            "Estudiante": df["student_name"],
            "Puntaje total": df["score"],
            "Caso 1": df["case_1"].map(short),
            "Pts. caso 1": df["score_1"],
            "Caso 2": df["case_2"].map(short),
            "Pts. caso 2": df["score_2"],
        })
        st.dataframe(
            view, hide_index=True, width="stretch",
            column_config={
                "Puntaje total": st.column_config.ProgressColumn(
                    "Puntaje total", min_value=0, max_value=200, format="%.1f"),
                "Pts. caso 1": st.column_config.NumberColumn(format="%.1f"),
                "Pts. caso 2": st.column_config.NumberColumn(format="%.1f"),
            },
        )

# ─────────────────────────── Pestaña: Guía ───────────────────────────
with tab_guide:
    st.markdown("""
### Objetivo
Identificar la distribución teórica que mejor describe cada variable continua y estimar sus parámetros
mediante diagnóstico gráfico dual: **densidad vs. KDE** y **gráfico Q-Q**.

### Paso a paso
1. **Contexto:** lee la descripción del proceso y las restricciones de la variable.
2. **Muestra:** revisa el resumen estadístico, la tabla o descarga el CSV.
3. **Familia:** elige entre Normal, Lognormal, Weibull, Exponencial, Uniforme o Triangular.
4. **Parámetros:** parten de los valores estándar de la familia (por ejemplo, μ = 0 y σ = 1 en la Normal).
   Ajústalos con una cifra decimal hasta que el modelo describa los datos.
5. **Diagnóstico:** la curva punteada debe superponerse al KDE, y los puntos del Q-Q deben seguir la diagonal.
6. **Envío:** cuando estés seguro de ambos casos, envíalos. **Solo se permite un envío por código.**

### Calificación (por caso, máximo 100)
El puntaje mide qué tanto se **solapa** la densidad que propones, $g(x)$, con la densidad real
del proceso, $f(x)$:

$$\\text{Puntaje} = 100 \\times \\int \\min\\big(f(x),\\, g(x)\\big)\\,dx$$

Es el porcentaje de área que comparten ambas curvas:
- **100** significa que tu modelo coincide con el real.
- **0** significa que no comparten nada de área.

No hay penalizaciones abruptas. Un modelo de otra familia con una forma parecida recibe puntaje parcial,
pero la familia correcta con buenos parámetros siempre logra el mayor solapamiento.
""")

st.divider()
st.caption("Laboratorio de Modelación Probabilística · Estadística I · Ingeniería Industrial")
