"""Where Is The Leak? - Water Leakage Detection & Localization (BattLeDIM 2018).
Run:  streamlit run app.py
Needs artifacts/water_leakage_models.joblib (made by the notebook) next to this file or in its parent folder."""
import re
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

THEME_TOML = """[theme]
base = "dark"
primaryColor = "#4f8cff"
backgroundColor = "#0b0e14"
secondaryBackgroundColor = "#12161d"
textColor = "#e8ecf2"
font = "sans serif"

[browser]
gatherUsageStats = false
"""


def ensure_theme() -> bool:
    """Create .streamlit/config.toml in the folder streamlit is run from (Streamlit only reads it from there).
    Returns True if the file was just created/updated, i.e. the app must be restarted once to apply it."""
    try:
        f = Path.cwd() / ".streamlit" / "config.toml"
        if not f.exists() or f.read_text(encoding="utf-8") != THEME_TOML:
            f.parent.mkdir(exist_ok=True)
            f.write_text(THEME_TOML, encoding="utf-8")
            return True
    except OSError:
        pass
    return False


ensure_theme()
st.set_page_config(page_title="Where Is The Leak?", layout="wide", initial_sidebar_state="collapsed")
HERE = Path(__file__).resolve().parent
LEAK_THR = 0.05  # same label threshold as the notebook
FEAT_RE = re.compile(r"^(?P<col>.+)_(?P<kind>diff|mean|std|dev)(?P<w>\d+)?$")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{--bg:#0b0e14;--card:#12161d;--line:#232a35;--txt:#e8ecf2;--mut:#8a94a6;--acc:#4f8cff;--bad:#ff5c5c;--ok:#3ddc97;
--font:'Inter','Segoe UI',system-ui,-apple-system,Roboto,sans-serif}
.stApp,.stApp p,.stApp label,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp li,.stApp input,.stApp textarea,.stApp button p,
[data-testid="stMetricValue"],[data-testid="stMetricLabel"] p,button[data-baseweb="tab"] p{font-family:var(--font)}
.stApp{background:var(--bg);color:var(--txt)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stDecoration"]{display:none!important}
[data-testid="stToolbar"]{right:1rem}
[data-testid="stToolbar"] button,[data-testid="stToolbar"] a{color:var(--mut)}
[data-testid="stToolbar"] button:hover,[data-testid="stToolbar"] a:hover{color:var(--txt)}
[data-testid="stSidebar"]{background:var(--card);border-right:1px solid var(--line)}
[data-testid="stSidebar"] h2{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.12em;color:var(--mut)}
[data-testid="stSidebar"] h3{font-size:.85rem;font-weight:600;color:var(--txt)}
.block-container{padding-top:1.6rem;max-width:1440px}
h1,h2,h3,h4,label,p,li{color:var(--txt)}
[data-testid="stMarkdownContainer"] p{font-size:.9rem}
[data-testid="stCaptionContainer"],small{color:var(--mut)!important}
.hero{display:flex;align-items:center;gap:16px;padding:0;margin:0}
.rule{height:1px;background:var(--line);margin:10px 0 18px}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none!important}
[data-testid="stPopover"]{display:flex;justify-content:flex-end}
[data-testid="stPopover"] button{background:var(--card);color:var(--txt);border:1px solid var(--line);border-radius:8px;font-weight:500;padding:6px 18px}
[data-testid="stPopover"] button:hover{border-color:var(--acc);color:var(--acc)}
[data-testid="stPopoverBody"]{min-width:440px;max-height:78vh;overflow-y:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
.brand{display:flex;align-items:center;gap:14px}
.logo{width:36px;height:36px;border-radius:9px;background:var(--acc);display:grid;place-items:center;flex:none}
.hero-t{font-size:1.2rem;font-weight:650;letter-spacing:-.015em;color:var(--txt);line-height:1.25}
.hero-s{margin-top:2px;font-size:.8rem;color:var(--mut)}
.pill{flex:none;font-size:.68rem;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--mut);border:1px solid var(--line);background:var(--card);padding:6px 12px;border-radius:6px}
.status{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 14px;border-radius:8px;font-weight:600;font-size:.88rem;margin-bottom:14px;border:1px solid}
.status span:first-child::before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:currentColor;margin-right:10px;vertical-align:middle}
.status-t{font-weight:400;font-size:.76rem;color:var(--mut)}
.ok{background:rgba(61,220,151,.07);color:var(--ok);border-color:rgba(61,220,151,.30)}
.bad{background:rgba(255,92,92,.08);color:var(--bad);border-color:rgba(255,92,92,.35)}
div[data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);padding:16px 18px;border-radius:8px}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{color:var(--mut)!important;font-size:.7rem!important;font-weight:600!important;text-transform:uppercase;letter-spacing:.09em}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] *{color:var(--txt)!important;font-size:1.8rem!important;font-weight:600!important;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
button[data-baseweb="tab"]{padding:10px 2px;margin-right:22px;background:transparent}
button[data-baseweb="tab"] p{font-size:.85rem;font-weight:500;color:var(--mut)}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--txt)}
[data-baseweb="tab-highlight"]{background:var(--acc)!important;background-color:var(--acc)!important}
[data-baseweb="tab-border"]{background:var(--line)!important;background-color:var(--line)!important}
div[role="radiogroup"]{gap:8px;flex-wrap:wrap}
div[role="radiogroup"] label[data-baseweb="radio"]{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 16px;margin:0}
div[role="radiogroup"] label[data-baseweb="radio"]>div:first-child{display:none}
div[role="radiogroup"] label[data-baseweb="radio"] p{font-size:.85rem;font-weight:500;color:var(--mut)}
div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked){background:var(--acc);border-color:var(--acc)}
div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) p{color:#fff}
.timebar{display:flex;justify-content:space-between;font-size:.82rem;color:var(--mut)}
.timebar b{color:var(--txt);font-weight:600}
.sec{margin-bottom:8px}
.sec-t{font-size:.85rem;font-weight:600;color:var(--txt)}
.sec-s{font-size:.75rem;color:var(--mut);margin-top:2px}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:8px}
.stDownloadButton button,.stButton button{background:var(--card);color:var(--txt);border:1px solid var(--line);border-radius:8px;font-weight:500}
.stDownloadButton button:hover,.stButton button:hover{border-color:var(--acc);color:var(--acc)}
[data-testid="stFileUploaderDropzone"]{background:var(--bg)!important;border:1px dashed var(--line)!important;border-radius:8px}
[data-testid="stFileUploaderDropzone"] *{color:var(--mut)!important}
[data-testid="stFileUploaderDropzone"] button{background:var(--card)!important;border:1px solid var(--line)!important;color:var(--txt)!important;border-radius:8px}
[data-testid="stFileUploaderDropzone"] button *{color:var(--txt)!important}
div[data-baseweb="select"]>div,div[data-baseweb="input"],div[data-baseweb="base-input"]{background:var(--bg)!important;border-color:var(--line)!important}
.foot{margin-top:26px;padding-top:12px;border-top:1px solid var(--line);color:var(--mut);font-size:.78rem;text-align:center}
</style>""", unsafe_allow_html=True)

_t = go.layout.Template(pio.templates["plotly_dark"])
_t.layout.update(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                 font=dict(family="Inter, Segoe UI, Roboto, sans-serif", color="#aab4c3", size=12),
                 hoverlabel=dict(bgcolor="#12161d", bordercolor="#232a35", font=dict(color="#e8ecf2")),
                 colorway=["#4f8cff", "#2dd4bf", "#8a94a6", "#e0a73a", "#a78bfa", "#3ddc97"],
                 xaxis=dict(gridcolor="#1a212b", linecolor="#232a35", zeroline=False, tickfont=dict(size=11)),
                 yaxis=dict(gridcolor="#1a212b", linecolor="#232a35", zeroline=False, tickfont=dict(size=11)),
                 legend=dict(font=dict(size=11)))
pio.templates["leak_dark"] = _t
pio.templates.default = "leak_dark"


def show(fig, **_):
    try:  # newer Streamlit versions
        st.plotly_chart(fig, width="stretch", theme=None)
    except TypeError:  # older versions
        st.plotly_chart(fig, use_container_width=True, theme=None)


def section(title, sub=""):
    st.markdown(f'<div class="sec"><div class="sec-t">{title}</div><div class="sec-s">{sub}</div></div>', unsafe_allow_html=True)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


# ----------------------------- loading ---------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models(path: str):
    return joblib.load(path)


def find_models():
    for base in (HERE, HERE.parent):
        for p in (base / "artifacts" / "water_leakage_models.joblib", base / "water_leakage_models.joblib"):
            if p.exists():
                return p
    return None


def find_inp():
    for base in (HERE, HERE.parent, HERE / "Dataset", HERE.parent / "Dataset"):
        if (base / "L-TOWN.inp").exists():
            return base / "L-TOWN.inp"
    return None


def read_csv(src) -> pd.DataFrame:
    raw = src.getvalue().decode("utf-8-sig") if hasattr(src, "getvalue") else Path(src).read_text(encoding="utf-8-sig")
    df = pd.read_csv(StringIO(raw), sep=";", decimal=",")
    if df.shape[1] == 1:  # fallback: comma separated, dot decimal
        df = pd.read_csv(StringIO(raw), sep=",", decimal=".")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df.sort_values("Timestamp").reset_index(drop=True)


@st.cache_resource(show_spinner="Reading data...")
def load_sensor_data(press_b, flow_b, leak_b, folder):
    def rd(b, name):
        if b is not None:
            return read_csv(b)
        f = Path(folder) / name if folder else None
        return read_csv(f) if f and f.exists() else None
    p, f, l = rd(press_b, "2018_SCADA_Pressures.csv"), rd(flow_b, "2018_SCADA_Flows.csv"), rd(leak_b, "2018_Leakages.csv")
    if p is None:
        return None, None
    sensors = p if f is None else p.merge(f, on="Timestamp", how="inner")
    return sensors, l


# --------------------------- feature engineering ----------------------
def build_needed_features(df, needed, cfg):
    """Rebuild exactly the features the models use (same causal logic as the notebook)."""
    base = df.drop(columns="Timestamp").astype(float).reset_index(drop=True)
    out, bw = {}, cfg.get("BASELINE_WINDOW", 144)
    for name in needed:
        if name in base.columns:
            out[name] = base[name]; continue
        m = FEAT_RE.match(name)
        if not m or m["col"] not in base.columns:
            raise ValueError(f"Feature '{name}' cannot be built - missing sensor column.")
        s, w = base[m["col"]], int(m["w"] or 0)
        if m["kind"] == "diff":
            out[name] = s.diff().fillna(0)
        elif m["kind"] == "mean":
            out[name] = s.rolling(w, min_periods=1).mean()
        elif m["kind"] == "std":
            out[name] = s.rolling(w, min_periods=2).std().fillna(0)
        else:
            out[name] = s - s.rolling(w or bw, min_periods=1).mean()
    return pd.DataFrame(out)


@st.cache_resource(show_spinner="Running detection & localization...")
def run_inference(_models, _df, data_key, start, end):
    cfg = _models["config"]
    det_f, loc_f = _models["detection_features"], _models["localization_features"]
    X = build_needed_features(_df, sorted(set(det_f) | set(loc_f)), cfg)
    keep = (_df["Timestamp"] >= start) & (_df["Timestamp"] <= end) & (np.arange(len(_df)) >= cfg.get("BASELINE_WINDOW", 144))
    idx = np.where(keep)[0]
    Xs = X.iloc[idx].astype("float32")
    det = _models["detection_model"]
    p_leak = det.predict_proba(Xs[det_f])[:, list(det.classes_).index(1)]
    cols = []
    for est in _models["localization_model"].estimators_:
        pr = est.predict_proba(Xs[loc_f])
        cols.append(pr[:, list(est.classes_).index(1)] if 1 in est.classes_ else np.zeros(len(Xs)))
    loc = pd.DataFrame(np.column_stack(cols), columns=_models["link_names"])
    return _df["Timestamp"].iloc[idx].reset_index(drop=True), pd.Series(p_leak, name="p_leak"), loc, idx


# ------------------------------ 3D map ---------------------------------
def schematic_layout(links, sensors):
    """Schematic (NOT geographic) layout: pump in the centre, links on a ring, sensors on the outer ring."""
    rows = [("PUMP_1", "pump", 0.0, 0.0)]
    for i, l in enumerate(links):
        a = 2 * np.pi * i / len(links)
        rows.append((l, "link", 4 * np.cos(a), 4 * np.sin(a)))
    pres = [s for s in sensors if s.startswith("n")]
    for i, s in enumerate(pres):
        a = 2 * np.pi * (i + .5) / len(pres)
        rows.append((s, "sensor", 7.5 * np.cos(a), 7.5 * np.sin(a)))
    return pd.DataFrame(rows, columns=["id", "type", "x", "y"])


def parse_inp(text):
    sec, coords, pipes = None, {}, []
    for line in text.splitlines():
        line = line.split(";")[0].strip()
        if not line:
            continue
        if line.startswith("["):
            sec = line.upper(); continue
        p = line.split()
        if sec == "[COORDINATES]" and len(p) >= 3:
            coords[p[0]] = (float(p[1]), float(p[2]))
        elif sec in ("[PIPES]", "[PUMPS]", "[VALVES]") and len(p) >= 3:
            pipes.append((p[0], p[1], p[2]))
    return coords, pipes


def layout_from_inp(text, links, sensors):
    """Real network layout from an EPANET .inp (coordinates normalised to about +-9)."""
    coords, pipes = parse_inp(text)
    if not coords:
        raise ValueError("no [COORDINATES] section found")
    pts = np.array(list(coords.values()))
    c0, sc = (pts.min(0) + pts.max(0)) / 2, 18 / max(float(np.ptp(pts, axis=0).max()), 1e-9)
    nrm = lambda xy: (float((xy[0] - c0[0]) * sc), float((xy[1] - c0[1]) * sc))
    mid = {pid: ((coords[a][0] + coords[b][0]) / 2, (coords[a][1] + coords[b][1]) / 2) for pid, a, b in pipes if a in coords and b in coords}
    rows = []
    for sid in sensors:
        if sid == "PUMP_1" and sid in mid:
            rows.append((sid, "pump", *nrm(mid[sid])))
        elif sid in coords:
            rows.append((sid, "sensor", *nrm(coords[sid])))
        elif sid in mid:
            rows.append((sid, "sensor", *nrm(mid[sid])))
    rows += [(l, "link", *nrm(mid[l])) for l in links if l in mid]
    if not any(r[1] == "link" for r in rows):
        raise ValueError("none of the leak pipes were found in the file")
    edges = [(*nrm(coords[a]), *nrm(coords[b])) for _, a, b in pipes if a in coords and b in coords]
    return pd.DataFrame(rows, columns=["id", "type", "x", "y"]), edges


def map3d(layout, probs, thr, pressures, chronic=(), edges=None, kpis=None):
    fig = go.Figure()
    zmax = 10
    if edges:
        xs, ys = [], []
        for x1, y1, x2, y2 in edges:
            xs += [x1, x2, None]; ys += [y1, y2, None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=[None if v is None else 0 for v in xs], mode="lines",
                                   line=dict(color="#2b3138", width=2), name="Network", hoverinfo="skip"))
    s = layout[layout.type == "sensor"]
    fig.add_trace(go.Scatter3d(x=s.x, y=s.y, z=[0] * len(s), mode="markers", text=s.id,
        marker=dict(size=3, color="#6e7681", symbol="diamond"), name="Pressure sensors",
        customdata=[f"{pressures.get(i, np.nan):.1f} m" for i in s.id], hovertemplate="%{text}<br>Pressure: %{customdata}<extra></extra>"))
    pump = layout[layout.type == "pump"].head(1)
    fig.add_trace(go.Scatter3d(x=pump.x, y=pump.y, z=[0] * len(pump), mode="markers+text", text=["PUMP_1"] * len(pump), textposition="bottom center",
        textfont=dict(color="#e6edf3"), marker=dict(size=9, color="#e6edf3", symbol="square"), name="Pump"))
    lk = layout[layout.type == "link"].copy()
    lk["p"] = lk.id.map(probs).fillna(0)
    for r in lk.itertuples():  # stems
        fig.add_trace(go.Scatter3d(x=[r.x, r.x], y=[r.y, r.y], z=[0, r.p * zmax], mode="lines",
            line=dict(color="#8b949e" if r.id in chronic else "#f85149" if r.p >= thr else "#30363d", width=6), showlegend=False, hoverinfo="skip"))
    ch, lk = lk[lk.id.isin(chronic)], lk[~lk.id.isin(chronic)]
    fig.add_trace(go.Scatter3d(x=lk.x, y=lk.y, z=lk.p * zmax, mode="markers+text", text=lk.id, textposition="top center",
        textfont=dict(color="#e6edf3", size=10),
        marker=dict(size=7 + 14 * lk.p, color=lk.p, colorscale=[[0, "#30363d"], [.5, "#da3633"], [1, "#ffa198"]], cmin=0, cmax=1, showscale=True,
                    colorbar=dict(title="Confidence", len=.6, tickfont=dict(color="#8b949e")), line=dict(color="#0d1117", width=1)),
        customdata=(lk.p * 100).round(1), name="Pipes", hovertemplate="%{text}<br>Leak confidence: %{customdata}%<extra></extra>"))
    if len(ch):
        fig.add_trace(go.Scatter3d(x=ch.x, y=ch.y, z=ch.p * zmax, mode="markers+text", text=ch.id, textposition="top center",
            textfont=dict(color="#c9d1d9", size=10), marker=dict(size=10, color="#c9d1d9", symbol="diamond"), name="Chronic leak",
            customdata=(ch.p * 100).round(1), hovertemplate="%{text} (chronic)<br>Confidence: %{customdata}%<extra></extra>"))
    gx, gy = [layout.x.min() - 1, layout.x.max() + 1], [layout.y.min() - 1, layout.y.max() + 1]
    fig.add_trace(go.Surface(x=gx, y=gy, z=np.zeros((2, 2)), showscale=False, opacity=.35, colorscale=[[0, "#161b22"], [1, "#161b22"]], hoverinfo="skip"))
    m = max(gx[1] - gx[0], gy[1] - gy[0])
    fig.update_layout(height=720, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", x=0.01, y=0.02, bgcolor="rgba(18,22,29,.88)", bordercolor="#232a35", borderwidth=1, font=dict(size=11)),
        scene=dict(bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(title="Leak confidence", showticklabels=False, gridcolor="#21262d", backgroundcolor="rgba(0,0,0,0)"),
                   camera=dict(eye=dict(x=1.4, y=1.4, z=.9)), aspectmode="manual", aspectratio=dict(x=1.8 * (gx[1] - gx[0]) / m, y=1.8 * (gy[1] - gy[0]) / m, z=.6)))
    for k, (lab, val, col) in enumerate(kpis or []):
        fig.add_annotation(xref="paper", yref="paper", x=0, y=1, xanchor="left", yanchor="top", xshift=14 + k * 174, yshift=-14,
                           width=158, align="left", showarrow=False, bgcolor="rgba(18,22,29,.92)", bordercolor="#232a35", borderwidth=1, borderpad=10,
                           text=f"<span style='color:#8a94a6;font-size:11px'>{lab}</span><br><span style='color:{col};font-size:17px'><b>{val}</b></span>")
    return fig


# ------------------------------- main ----------------------------------
def main():
    h_l, h_r = st.columns([6, 1])
    h_l.markdown('<div class="hero"><div class="brand"><div class="logo"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.5s6.5 7 6.5 11.5a6.5 6.5 0 0 1-13 0C5.5 9.5 12 2.5 12 2.5z"/></svg></div><div><div class="hero-t">Where Is The Leak?</div><div class="hero-s">Leak detection and localization &middot; L-Town network &middot; BattLeDIM 2018</div></div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    with h_r:
        pop = st.popover("Settings")
    with pop:
        mp = find_models()
        mp_in = str(mp) if mp else st.text_input("Models file", "artifacts/water_leakage_models.joblib")
        st.subheader("Data")
        press_f = st.file_uploader("SCADA Pressures CSV", type="csv")
        flow_f = st.file_uploader("SCADA Flows CSV", type="csv")
        leak_f = st.file_uploader("Leakages CSV (optional ground truth)", type="csv")
        folder = next((str(d) for d in (HERE / "Dataset", HERE.parent / "Dataset", HERE, HERE.parent) if (d / "2018_SCADA_Pressures.csv").exists()), None)
        if folder and not (press_f or flow_f):
            st.caption(f"Using local folder: {folder}")
        st.subheader("Thresholds")
        det_thr = st.slider("Detection threshold", .05, .95, .50, .05)
        loc_thr = st.slider("Localization threshold", .05, .95, .50, .05)
        persist = st.slider("Alert persistence (steps of 5 min)", 1, 36, 6)
        inp_f = st.file_uploader("EPANET network file (.inp)", type="inp", help="Draws the real L-Town network from its coordinates.")
        layout_f = st.file_uploader("Network coordinates CSV (id,x,y)", type="csv", help="Optional. Replace the schematic layout with real coordinates.")

    if not Path(mp_in).exists():
        st.error("Models file not found. Run the notebook first, then place `artifacts/water_leakage_models.joblib` next to this app."); st.stop()
    models = load_models(mp_in)
    df, truth = load_sensor_data(press_f, flow_f, leak_f, None if (press_f or flow_f) else folder)  # uploads never mix with the demo folder
    if df is None:
        st.info("Open **Settings** (top right) and upload the SCADA pressure and flow files, or put the `Dataset` folder next to the app, to start."); st.stop()
    if len(df) < 200:
        st.error("At least 200 rows (≈17 h at 5 min) are needed to build the rolling features."); st.stop()

    need_cols = models["config"]["sensor_columns"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        st.error(f"The uploaded data is missing {len(missing)} sensor column(s) the models need: {', '.join(missing[:12])}"
                 f"{' ...' if len(missing) > 12 else ''}. Upload BOTH the Pressures and the Flows files of the L-Town network "
                 "(columns n1..n769, p227, p235, PUMP_1)."); st.stop()
    if df[need_cols].isna().any().any():
        df = df.copy(); df[need_cols] = df[need_cols].interpolate(limit_direction="both")
        pop.caption("Missing sensor values were interpolated.")
    step_min = df.Timestamp.diff().dropna().median() / pd.Timedelta(minutes=1)
    if abs(step_min - 5) > 0.01:
        st.warning(f"Sampling interval is {step_min:g} min; the models expect 5 min, so results may be unreliable.")
    t0, t1 = df.Timestamp.min().to_pydatetime(), df.Timestamp.max().to_pydatetime()
    default_start = max(t0, t1 - pd.Timedelta(days=14).to_pytimedelta())
    start, end = pop.slider("Analysis period", t0, t1, (default_start, t1), step=pd.Timedelta(hours=1).to_pytimedelta())
    dkey = (mp_in, len(df), str(df.Timestamp.iloc[0]), str(df.Timestamp.iloc[-1]), round(float(df.drop(columns="Timestamp").sum().sum()), 3))
    try:
        ts, p_leak, loc, idx = run_inference(models, df, dkey, pd.Timestamp(start), pd.Timestamp(end))
    except ValueError as e:
        st.error(str(e)); st.stop()
    if len(ts) == 0:
        st.warning("No rows in the selected period."); st.stop()

    above = (p_leak >= det_thr).astype(int)
    alert = above.rolling(persist, min_periods=persist).min().fillna(0).astype(bool)
    n_alert = int(alert.sum())
    cur = len(ts) - 1
    links = models["link_names"]
    chronic = [l for l in models["config"].get("DETECTION_EXCLUDE_LINKS", []) if l in links]
    reg = [l for l in links if l not in chronic]

    latest = bool(alert.iloc[-1])
    top_pipe = loc[reg].mean().idxmax()
    view = st.radio("View", ["Map", "Timeline", "Pipes", "Sensors", "Model"], horizontal=True, label_visibility="collapsed")

    if view == "Timeline":
        st.markdown(f'<div class="status {"bad" if latest else "ok"}"><span>{"Leak suspected" if latest else "Network operating normally"}</span><span class="status-t">As of {ts.iloc[-1]:%Y-%m-%d %H:%M}</span></div>', unsafe_allow_html=True)
        c = st.columns(4)
        c[0].metric("Steps analysed", f"{len(ts):,}")
        c[1].metric(f"Alert steps ({n_alert / len(ts) * 100:.1f}%)", f"{n_alert:,}")
        c[2].metric("Peak leak confidence", f"{p_leak.max() * 100:.0f}%")
        c[3].metric("Most suspected pipe", top_pipe)
        fig = go.Figure(go.Scatter(x=ts, y=p_leak, name="Leak confidence", line=dict(color="#4f8cff", width=1.6)))
        fig.add_hline(y=det_thr, line_dash="dash", line_color="#ff5c5c", annotation_text="threshold")
        if truth is not None:
            tr = truth.set_index("Timestamp").reindex(ts).drop(columns="p257", errors="ignore").gt(LEAK_THR).any(axis=1)
            fig.add_trace(go.Scatter(x=ts, y=tr.astype(int) * 0.05, name="Actual leak (excl. p257)", fill="tozeroy", line=dict(width=0, shape="hv"), fillcolor="rgba(248,81,73,.9)", hoverinfo="skip"))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(range=[0, 1.02], title="Confidence"), legend=dict(orientation="h", x=0, y=1.12))
        with st.container(border=True):
            section("Leak confidence over time", "Model confidence against the detection threshold" + (" and the documented leak periods" if truth is not None else ""))
            show(fig)
        if truth is not None:
            from sklearn.metrics import matthews_corrcoef, f1_score
            yt = tr.astype(int).values
            if len(set(yt)) == 2:
                m = st.columns(3)
                m[0].metric("MCC", f"{matthews_corrcoef(yt, above):.3f}")
                m[1].metric("Leak F1", f"{f1_score(yt, above, zero_division=0):.3f}")
                m[2].metric("Normal F1", f"{f1_score(yt, above, pos_label=0, zero_division=0):.3f}")

    if view == "Map":
        sensors = [c for c in df.columns if c != "Timestamp"]
        layout, edges = None, None
        auto = find_inp()
        inp_text = inp_f.getvalue().decode("utf-8", "ignore") if inp_f else (auto.read_text(encoding="utf-8", errors="ignore") if auto else None)
        if inp_text:
            try:
                layout, edges = layout_from_inp(inp_text, links, sensors)
            except Exception as e:
                st.warning(f"Could not use the .inp file: {e}")
        if layout is None and layout_f:
            layout = pd.read_csv(layout_f)
            layout["type"] = layout["id"].map(lambda v: "link" if v in links else "pump" if v == "PUMP_1" else "sensor")
        real = layout is not None
        if not real:
            layout = schematic_layout(links, sensors)
        kpis = [("STATUS", "Leak suspected" if latest else "Normal", "#ff5c5c" if latest else "#3ddc97"),
                ("ALERT STEPS", f"{n_alert:,} ({n_alert / len(ts) * 100:.0f}%)", "#e8ecf2"),
                ("TOP PIPE", top_pipe, "#e8ecf2")]
        map_box = st.container(border=True)
        with st.container(border=True):
            i = st.slider("Time", 0, cur, cur, key="mapt", label_visibility="collapsed") if cur > 0 else 0
            st.markdown(f'<div class="timebar"><span>{ts.iloc[i]:%Y-%m-%d %H:%M}</span><span>Detection confidence <b>{p_leak.iloc[i] * 100:.0f}%</b></span></div>', unsafe_allow_html=True)
        with map_box:
            show(map3d(layout, loc.iloc[i].to_dict(), loc_thr, df.iloc[idx[i]].to_dict(), chronic, edges, kpis))
        st.caption("Each stem is a monitored pipe; height and colour show the model's leak confidence at the chosen time (grey diamond = chronic leak). "
                   + ("" if real else "Note: the layout is schematic, not geographic - upload the EPANET .inp file for the real network."))
        if not real:
            st.download_button("Download layout template", layout.to_csv(index=False), "network_layout.csv")

    if view == "Pipes":
        if chronic:
            st.info(f"{', '.join(chronic)}: long-lasting leak (documented from 8 Jan 2018, never repaired) - excluded from the ranking so it does not hide the other pipes. "
                    f"Flagged in {loc[chronic].ge(loc_thr).mean().mean() * 100:.0f}% of the period.")
        mean_p = loc[reg].mean().sort_values(ascending=False)
        active = (loc[reg] >= loc_thr).mean().mul(100).round(1)
        rank = pd.DataFrame({"Mean confidence": mean_p.round(3), "Time active %": active[mean_p.index]})
        a, b = st.columns([1, 2])
        with a:
            with st.container(border=True):
                section("Pipe ranking", "Mean confidence and share of time above threshold")
                table(rank, height=480)
        step = max(1, len(loc) // 600)
        hm = px.imshow(loc[reg].iloc[::step].T.values, x=ts.iloc[::step], y=reg, aspect="auto", color_continuous_scale=[[0, "#161b22"], [.5, "#da3633"], [1, "#ffa198"]], zmin=0, zmax=1)
        hm.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0), coloraxis_colorbar=dict(title="Conf."))
        with b:
            with st.container(border=True):
                section("Confidence by pipe over time", "Each row is a monitored pipe")
                show(hm)

    if view == "Sensors":
        cols = st.multiselect("Sensors", [c for c in df.columns if c != "Timestamp"], default=["PUMP_1"] if "PUMP_1" in df else None)
        if cols:
            seg = df.loc[idx, ["Timestamp"] + cols]
            with st.container(border=True):
                section("Sensor readings", "Selected sensors over the analysed period")
                show(px.line(seg.melt("Timestamp"), x="Timestamp", y="value", color="variable").update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0)))

    if view == "Model":
        st.write(f"**Detection:** {models['detection_model_name']} · {len(models['detection_features'])} features")
        st.write(f"**Localization:** {models['localization_model_name']} · {len(models['localization_features'])} features · {len(links)} pipes")
        det = models["detection_model"]
        imp = pd.Series(det.feature_importances_, index=models["detection_features"]).nlargest(12).sort_values()
        with st.container(border=True):
            section("Top detection features", "Most important inputs to the detection model")
            show(px.bar(imp, orientation="h", labels={"value": "Importance", "index": ""}).update_layout(height=360, showlegend=False, margin=dict(l=0, r=0, t=10, b=0)))
        res = pd.concat([ts.rename("Timestamp"), p_leak.round(4), alert.rename("alert"), loc.round(4)], axis=1)
        st.download_button("Download predictions (CSV)", res.to_csv(index=False), "leak_predictions.csv", "text/csv")
        st.markdown('<div class="foot">Where Is The Leak? · BattLeDIM 2018 · Graduation Project</div>', unsafe_allow_html=True)
        st.caption("Scores are model confidence values (tree-ensemble votes), not calibrated probabilities. Models were trained on one year / one set of events; scores show recognition of known leak patterns, not guaranteed generalisation to new networks.")


if __name__ == "__main__":
    main()
