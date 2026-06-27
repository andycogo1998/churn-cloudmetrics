import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

st.set_page_config(page_title="Churn CloudMetrics", layout="wide", page_icon="📊")

TEAL = "#00B8A9"        # color de marca, positivo, cuentas activas
TEAL_OSCURO = "#0A8C82"
NAVY = "#16284B"        # texto y titulos
CORAL = "#FF6B6B"       # churn, negativo
AMBAR = "#F4A340"       # advertencia, suspendido
GRIS = "#9AA7B0"        # neutro, inactivo
# Alias para mantener compatibilidad con el resto del dashboard
VERDE = TEAL; ROJO = CORAL; AZUL = TEAL_OSCURO; NARANJA = AMBAR
ESCALA_CHURN = [TEAL, AMBAR, CORAL]

# Template propio: fondo blanco y texto navy SIEMPRE, asi el dashboard se ve
# igual en modo claro u oscuro (los graficos no heredan el tema del navegador).
pio.templates["alegra"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", color=NAVY, size=13),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=[TEAL, AMBAR, CORAL, TEAL_OSCURO, GRIS, NAVY],
        title=dict(font=dict(size=15, color=NAVY), x=0.0, xanchor="left", pad=dict(b=8)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#e6eaee",
                   ticks="outside", tickcolor="#e6eaee", title_font=dict(size=12, color="#7b8794")),
        yaxis=dict(showgrid=False, zeroline=False,
                   title_font=dict(size=12, color="#7b8794")),
        legend=dict(bgcolor="rgba(0,0,0,0)", title="", font=dict(size=12)),
        margin=dict(t=48, b=24, l=12, r=12),
    )
)
TPL = "alegra"


def estilo_barras(fig, fmt=None):
    """Pone los valores sobre las barras de forma consistente."""
    fig.update_traces(textposition="outside", cliponaxis=False,
                      textfont=dict(size=12, color=NAVY))
    if fmt:
        fig.update_traces(texttemplate=fmt)
    return fig

st.markdown(f"""<style>
.block-container {{padding-top: 1.5rem;}}
h1,h2,h3 {{color:{NAVY};}}
/* Centrar las tarjetas de metricas */
[data-testid="stMetric"] {{
    text-align: center;
    background: #ffffff;
    border: 1px solid #eef1f4;
    border-radius: 12px;
    padding: 14px 8px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}}
[data-testid="stMetric"] label {{justify-content: center; width: 100%;}}
[data-testid="stMetricValue"] {{font-size: 1.5rem; justify-content: center; color: {NAVY};}}
[data-testid="stMetricLabel"] {{justify-content: center;}}
[data-testid="stMetricLabel"] p {{color: #5b6b7b; font-weight: 600;}}
div[data-testid="stMetric"] > div {{justify-content: center; align-items: center;}}
.insight {{
    background: #f4f9f8;
    border-left: 4px solid {TEAL};
    border-radius: 8px;
    padding: 12px 16px;
    height: 100%;
}}
.insight .n {{font-size: 1.45rem; font-weight: 700; color: {NAVY}; line-height: 1.1;}}
.insight .t {{font-size: 0.86rem; color: #5b6b7b; margin-top: 4px;}}
.insight.warn {{background: #fef6ed; border-left-color: {AMBAR};}}
.insight.bad {{background: #fdf0f0; border-left-color: {CORAL};}}
</style>""", unsafe_allow_html=True)


def logo_cloudmetrics():
    # Logo inventado: nube + barras de metrica, en turquesa de marca. Compacto (sin saltos de linea).
    return (
        f'<svg width="140" height="140" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<path d="M20 42h24a11 11 0 0 0 1.5-21.9A15 15 0 0 0 17 24.5 10 10 0 0 0 20 42z" fill="{TEAL}" opacity="0.18"/>'
        f'<path d="M22 40h22a9 9 0 0 0 1-17.9A13 13 0 0 0 19 26 8 8 0 0 0 22 40z" fill="none" stroke="{TEAL}" stroke-width="2.4"/>'
        f'<rect x="25" y="31" width="3.4" height="7" rx="1.2" fill="{TEAL_OSCURO}"/>'
        f'<rect x="30.3" y="27" width="3.4" height="11" rx="1.2" fill="{TEAL}"/>'
        f'<rect x="35.6" y="23" width="3.4" height="15" rx="1.2" fill="{AMBAR}"/>'
        f'</svg>'
    )

CENTROIDES = {
    "Colombia": (4.57, -74.30), "México": (23.63, -102.55), "Costa Rica": (9.75, -83.75),
    "República Dominicana": (18.74, -70.16), "Panamá": (8.54, -80.78), "Ecuador": (-1.83, -78.18),
}


def listar_csvs():
    return sorted(glob.glob("**/*.csv", recursive=True))


@st.cache_data
def clasificar(firma):
    """Detecta cada fuente por sus columnas, no por el nombre del archivo."""
    base = chat = wa = tel = None
    detalle = []
    for f in firma:
        try:
            cols = set(pd.read_csv(f, nrows=0).columns)
        except Exception:
            continue
        tipo = "otro"
        if "estado_cuenta" in cols and ("tipo_retiro" in cols or "fecha_primer_login" in cols):
            base = f; tipo = "base unificada"
        elif "csat_score" in cols:
            chat = chat or f; tipo = "soporte chat"
        elif "sentimiento_usuario" in cols:
            wa = wa or f; tipo = "soporte whatsapp"
        elif "nps_post_llamada" in cols or "escalo_a_especialista" in cols:
            tel = tel or f; tipo = "soporte telefono"
        elif "estado_cuenta" in cols and base is None:
            base = f; tipo = "base (solo usuarios)"
        detalle.append((f, tipo))
    return base, chat, wa, tel, detalle


@st.cache_data
def cargar_base(path):
    df = pd.read_csv(path, dtype=str)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    for c in ["user_id.1", "user_id.2"]:
        if c in df.columns:
            df = df.drop(columns=c)
    # La base final puede traer el soporte ya agregado. El dashboard lo recalcula
    # desde los archivos de soporte crudos, asi que se descartan estas columnas
    # para que el merge posterior no genere colisiones de nombres (_x / _y).
    _sop_baked = ["chat_tickets", "chat_no_resueltos", "chat_reaperturas", "chat_csat_prom", "chat_tpr_prom",
                  "wa_tickets", "wa_deriva_agente", "wa_negativos", "tel_llamadas", "tel_escalados", "tel_nps_prom",
                  "usa_soporte", "soporte_tickets_total", "friccion", "canal_mas_usado"]
    df = df.drop(columns=[c for c in _sop_baked if c in df.columns])
    num = ["monto_mensual_usd", "nps_salida", "dias_primer_factura", "facturas_emitidas_mes1",
           "facturas_emitidas_mes3", "reportes_generados_mes3", "usuarios_adicionales", "sesiones_promedio_semana"]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["fecha_registro", "fecha_ultimo_pago", "fecha_retiro"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "fecha_primer_login" in df.columns:
        s = pd.to_numeric(df["fecha_primer_login"], errors="coerce")
        df["fecha_primer_login"] = pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
    df["mrr"] = df["monto_mensual_usd"]
    df["churn_estado"] = df["estado_cuenta"].fillna("").str.lower().ne("activo")
    df["churn_retiro"] = df["tipo_retiro"].notna()
    df["activado"] = df["facturas_emitidas_mes1"].fillna(0) > 0
    sino = lambda s: df[s].fillna("").str.lower().eq("si")
    # Indice de activacion: 5 senales fuertes de configuracion temprana (True = falla)
    _falsa = pd.Series(False, index=df.index)
    _sig = [
        (~sino("configuracion_empresa_completa")) if "configuracion_empresa_completa" in df.columns else _falsa,
        (df["facturas_emitidas_mes1"].fillna(0).eq(0)) if "facturas_emitidas_mes1" in df.columns else _falsa,
        (~sino("plan_cuentas_configurado")) if "plan_cuentas_configurado" in df.columns else _falsa,
        (~sino("empleados_cargados")) if "empleados_cargados" in df.columns else _falsa,
        (~sino("modulo_cxc_activo")) if "modulo_cxc_activo" in df.columns else _falsa,
    ]
    df["fallas_activacion"] = sum(s.astype(int) for s in _sig)
    df["flag_estado_sin_retiro"] = df["churn_estado"] & ~df["churn_retiro"]
    df["flag_fecha_invertida"] = df["fecha_ultimo_pago"] < df["fecha_registro"]
    df["flag_pago_geo"] = ((df["metodo_pago"] == "PSE") & (df["pais"] != "Colombia")) | \
                          ((df["metodo_pago"] == "OXXO") & (df["pais"] != "México"))
    df["flag_login_no_activo"] = df["churn_estado"] & sino("login_ultimos_30_dias")
    if "primer_factura_emitida" in df.columns:
        df["flag_activacion_contradice"] = df["primer_factura_emitida"].fillna("").str.lower().eq("no") & \
                                           (df["facturas_emitidas_mes1"].fillna(0) > 0)
    else:
        df["flag_activacion_contradice"] = False
    df["flag_retiro_fecha_invertida"] = df["churn_retiro"] & (df["fecha_retiro"] < df["fecha_ultimo_pago"])
    return df


@st.cache_data
def cargar_soporte(pc, pw, pt):
    chat = pd.read_csv(pc, dtype=str) if pc else None
    wa = pd.read_csv(pw, dtype=str) if pw else None
    tel = pd.read_csv(pt, dtype=str) if pt else None
    if chat is not None:
        for c in ["tiempo_primera_respuesta_min", "tiempo_resolucion_hs", "csat_score"]:
            chat[c] = pd.to_numeric(chat[c], errors="coerce")
    if wa is not None:
        wa["tiempo_primera_respuesta_min"] = pd.to_numeric(wa["tiempo_primera_respuesta_min"], errors="coerce")
    if tel is not None:
        for c in ["duracion_llamada_min", "nps_post_llamada"]:
            tel[c] = pd.to_numeric(tel[c], errors="coerce")
    return chat, wa, tel


def agregar_usuario(chat, wa, tel):
    frames = []
    if chat is not None:
        g = chat.groupby("user_id").agg(
            chat_tickets=("ticket_id", "count"),
            chat_csat=("csat_score", "mean"),
            chat_reaperturas=("reapertura", lambda s: (s == "si").sum()),
            chat_no_resueltos=("resuelto", lambda s: (s == "no").sum())).reset_index()
        frames.append(g)
    if wa is not None:
        g = wa.groupby("user_id").agg(
            wa_tickets=("ticket_id", "count"),
            wa_negativos=("sentimiento_usuario", lambda s: (s == "negativo").sum()),
            wa_deriva=("deriva_a_agente", lambda s: (s == "si").sum())).reset_index()
        frames.append(g)
    if tel is not None:
        g = tel.groupby("user_id").agg(
            tel_tickets=("ticket_id", "count"),
            tel_nps=("nps_post_llamada", "mean"),
            tel_escalo=("escalo_a_especialista", lambda s: (s == "si").sum())).reset_index()
        frames.append(g)
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="user_id", how="outer")
    return out


FIRMA = tuple(listar_csvs())
DATA_PATH, P_CHAT, P_WA, P_TEL, DETALLE = clasificar(FIRMA)
if DATA_PATH is None:
    st.error("No se encontro la base unificada (ningun CSV con columna estado_cuenta). Archivos vistos: "
             + ", ".join(FIRMA) if FIRMA else "ningun CSV en el repo.")
    st.stop()

df = cargar_base(DATA_PATH)
chat, wa, tel = cargar_soporte(P_CHAT, P_WA, P_TEL)
agg = agregar_usuario(chat, wa, tel)
HAY_SOPORTE = agg is not None

if HAY_SOPORTE:
    df = df.merge(agg, on="user_id", how="left")
    for c in ["chat_tickets", "wa_tickets", "tel_tickets", "chat_reaperturas",
              "chat_no_resueltos", "wa_negativos", "wa_deriva", "tel_escalo"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    df["tickets_total"] = df.get("chat_tickets", 0) + df.get("wa_tickets", 0) + df.get("tel_tickets", 0)
    df["usa_soporte"] = df["tickets_total"] > 0
    df["friccion"] = (df.get("chat_reaperturas", 0) > 0) | (df.get("chat_no_resueltos", 0) > 0) | \
                     (df.get("tel_escalo", 0) > 0) | (df.get("wa_negativos", 0) > 0)


# ---------------- Sidebar ----------------
st.sidebar.title("Filtros")
definicion = st.sidebar.radio("Definicion de churn", ["estado_cuenta (recomendada)", "registro de retiro"],
    help="estado_cuenta marca 440 cuentas no activas. Retiros solo 140.")
col_churn = "churn_estado" if definicion.startswith("estado") else "churn_retiro"

def multi(label, col):
    opts = sorted(df[col].dropna().unique().tolist())
    return st.sidebar.multiselect(label, opts, default=opts)

f_estado = multi("Estado de cuenta", "estado_cuenta")
f_seg = multi("Segmento", "segmento")
f_pais = multi("Pais", "pais")
f_plan = multi("Plan", "plan")
d = df[df.estado_cuenta.isin(f_estado) & df.segmento.isin(f_seg) & df.pais.isin(f_pais) & df.plan.isin(f_plan)]
st.sidebar.markdown("---")
st.sidebar.caption(f"Mostrando {len(d):,} de {len(df):,} cuentas")


# ---------------- Cabecera ----------------
_motivo = ("Este tablero centraliza la información clave de churn en CloudMetrics para medirlo de forma "
           "consistente, identificar sus principales causas y facilitar acciones continuas desde Customer Experience. "
          )
_header = (
    '<div style="text-align:center; margin-bottom:0.2rem;">'
    '<div style="display:flex; justify-content:center; align-items:center; gap:12px;">'
    f'{logo_cloudmetrics()}'
    f'<h1 style="margin:0; color:{NAVY};">CloudMetrics - Centro de Control de Churn</h1>'
    '</div>'
    f'<p style="max-width:820px; margin:0.6rem auto 0; color:#5b6b7b; font-size:0.98rem;">{_motivo}</p>'
    '</div>'
)
st.markdown(_header, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

total = len(d); churned = int(d[col_churn].sum()); tasa = churned / total if total else 0
churn_teorico = d["churn_retiro"].mean() if total else 0
activos = d[d.estado_cuenta == "activo"]
mrr_total = d.mrr.sum(); mrr_activo = activos.mrr.sum(); mrr_riesgo = d.loc[d[col_churn], "mrr"].sum()
arpu = mrr_activo / len(activos) if len(activos) else 0
tasa_activacion = d["activado"].mean() if total else 0
revenue_churn = mrr_riesgo / mrr_total if mrr_total else 0
bajas_con_ficha = (d["churn_retiro"].sum() / churned) if churned else 0

st.markdown("---")

nombres_tabs = ["General", "Churn", "Negocio", "Causa raiz", "Soporte", "Calidad de datos"]
tabs = st.tabs(nombres_tabs)
T = dict(zip(nombres_tabs, tabs))


def churn_por(data, col):
    g = data.groupby(col).agg(cuentas=("user_id", "count"), churn=(col_churn, "sum")).reset_index()
    g["tasa"] = g["churn"] / g["cuentas"]
    return g.sort_values("tasa", ascending=False)


# ---------------- General ----------------
with T["General"]:
    act = d[d.estado_cuenta == "activo"]

    # ---- Diagnostico automatico de la base activa (reglas) ----
    n_act = len(act)
    mrr_act = act.mrr.sum()
    riesgo_mask = act["fallas_activacion"] >= 2
    en_riesgo = int(riesgo_mask.sum())
    pct_riesgo = en_riesgo / n_act if n_act else 0
    alertas = []
    if n_act:
        por_pais = act.assign(r=riesgo_mask).groupby("pais")["r"].mean()
        if len(por_pais) and por_pais.max() > pct_riesgo + 0.08:
            alertas.append(f"{por_pais.idxmax()} concentra mas cuentas con arranque parcial o critico que el promedio: {por_pais.max():.0%} de sus activas en riesgo frente al {pct_riesgo:.0%} general.")
        por_plan = act.assign(r=riesgo_mask).groupby("plan")["r"].mean()
        if len(por_plan) and por_plan.max() > pct_riesgo + 0.08:
            alertas.append(f"El plan {por_plan.idxmax()} arranca peor que el resto: {por_plan.max():.0%} de sus activas en riesgo.")
        if "friccion" in act.columns and act["friccion"].mean() > 0.45:
            alertas.append(f"La friccion de soporte en la base activa es alta ({act['friccion'].mean():.0%} tuvo alguna mala experiencia), un acelerador conocido del churn.")
    nivel = "ok" if pct_riesgo < 0.20 else ("warn" if pct_riesgo < 0.35 else "bad")
    _bordes = {"ok": TEAL, "warn": AMBAR, "bad": CORAL}
    cuerpo = (f"<b>La base activa es de {n_act:,} cuentas con MRR ${mrr_act:,.0f}.</b> "
              f"{en_riesgo:,} de ellas ({pct_riesgo:.0%}) estan en arranque parcial o critico, los dos niveles mas bajos "
              f"del termometro, es decir con dos o mas de los cinco "
              f"pasos clave de arranque sin completar (configurar la empresa, emitir la primera factura, cargar el "
              f"plan de cuentas, cargar empleados y activar cuentas por cobrar), "
              f"que es el grupo con mas riesgo de churn.")
    if alertas:
        cuerpo += "<br><b>Anomalias detectadas:</b><ul style='margin:4px 0 0 0; padding-left:18px;'>" + "".join(f"<li>{a}</li>" for a in alertas) + "</ul>"
    else:
        cuerpo += "<br>No se detectaron anomalias por pais, plan ni soporte en este corte."
    st.markdown(
        f"<div style='background:#f6fafa; border-left:4px solid {_bordes[nivel]}; border-radius:8px; padding:12px 16px; margin-bottom:6px;'>"
        f"<div style='font-size:0.78rem; font-weight:700; color:{_bordes[nivel]}; letter-spacing:.04em;'>DIAGNOSTICO AUTOMATICO DE LA BASE ACTIVA</div>"
        f"<div style='font-size:0.92rem; color:{NAVY}; margin-top:4px;'>{cuerpo}</div></div>", unsafe_allow_html=True)

    # ---- Tarjetas por estado, cada una con su MRR ----
    estados_cfg = [
        ("Cuentas", None, GRIS, "", "Cuentas que se encuentran en el filtro actual."),
        ("Activas", "activo", TEAL, "\u25b2 ", "Cuentas que hoy siguen activas y pagando."),
        ("Suspendidas", "suspendido", CORAL, "\u25bc ", "Cuentas cortadas, normalmente por falta de pago. Recuperables con cobranza."),
        ("Inactivas", "inactivo", AMBAR, "\u25bc ", "Cuentas que dejaron de tener actividad sin darse de baja formal."),
        ("Canceladas", "cancelado", CORAL, "\u25bc ", "Bajas definitivas de la cuenta."),
    ]
    kc = st.columns(5)
    for (lbl, estado, color, arrow, ayuda), c in zip(estados_cfg, kc):
        sub = d if estado is None else d[d.estado_cuenta == estado]
        card = (
            f'<div title="{ayuda}" style="background:#fff;border:1px solid #eef1f4;border-radius:12px;'
            f'padding:14px 8px;text-align:center;box-shadow:0 1px 2px rgba(0,0,0,0.03);">'
            f'<div style="font-size:0.8rem;color:#7b8794;">{lbl}</div>'
            f'<div style="font-size:1.7rem;font-weight:700;color:{NAVY};line-height:1.25;">{len(sub):,}</div>'
            f'<div style="font-size:0.82rem;font-weight:700;color:{color};">{arrow}MRR ${sub.mrr.sum():,.0f}</div>'
            f'</div>'
        )
        c.markdown(card, unsafe_allow_html=True)

    st.markdown("---")

    # ---- Cuentas activas hoy ----
    st.subheader("Cuentas activas hoy")
    st.caption("Como se componen la base de usuarios que sigue pagando actualmente.")

    cseg = st.columns([1, 2, 1])
    with cseg[1]:
        g = act["segmento"].value_counts().reset_index()
        g.columns = ["segmento", "cuentas"]
        fig = px.pie(g, names="segmento", values="cuentas", template=TPL, hole=0.55,
                     color_discrete_sequence=[TEAL, NAVY])
        fig.update_traces(textinfo="label+percent", textfont=dict(color="white", size=14))
        fig.update_layout(title="Por segmento", height=330, margin=dict(t=46, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(3)
    for col, titulo, cont in [("plan", "Por plan", cols[0]), ("pais", "Por pais", cols[1]), ("metodo_pago", "Por metodo de pago", cols[2])]:
        with cont:
            g = act[col].value_counts().reset_index()
            g.columns = [col, "cuentas"]
            g = g.sort_values("cuentas", ascending=True)
            tope = g["cuentas"].max() * 1.18 if len(g) else 1
            fig = px.bar(g, x="cuentas", y=col, orientation="h", template=TPL,
                         text="cuentas", color_discrete_sequence=[TEAL])
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(title=titulo, height=300, showlegend=False, margin=dict(t=40, b=10, l=8, r=10),
                              yaxis_title="", xaxis_title="",
                              xaxis=dict(showgrid=False, range=[0, tope]),
                              yaxis=dict(showgrid=False, automargin=True))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---- Cuentas registradas por mes ----
    st.markdown("##### Cuentas registradas por mes")
    st.caption("Proporcion de la cohorte que sigue activa vs los que entran al churn.")
    reg = d.dropna(subset=["fecha_registro"]).copy()
    reg["mes"] = reg["fecha_registro"].dt.to_period("M").dt.to_timestamp()
    reg["situacion"] = np.where(reg[col_churn], "Churn", "Activa")
    g = reg.groupby(["mes", "situacion"]).size().reset_index(name="cuentas")
    tot_mes = g.groupby("mes")["cuentas"].transform("sum")
    g["pct"] = g["cuentas"] / tot_mes
    fig = px.bar(g, x="mes", y="cuentas", color="situacion", template=TPL, barmode="stack",
                 text=g["pct"].map("{:.0%}".format), color_discrete_map={"Activa": TEAL, "Churn": CORAL})
    fig.update_traces(textposition="inside", textfont=dict(color="white", size=11), insidetextanchor="middle")
    fig.update_layout(height=360, xaxis_title="mes de registro", yaxis_title="cuentas",
                      legend=dict(orientation="h", y=1.1, x=0),
                      xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---- Termometro de arranque de la base activa ----
    st.subheader("Termometro de arranque de la base activa")
    st.caption("A cada cuenta activa le damos un puntaje de arranque segun cuantos pasos clave del onboarding "
               "completo: configurar la empresa, emitir la primera factura, cargar el plan de cuentas, cargar "
               "empleados y activar cuentas por cobrar. Cuanto peor arranca una cuenta, mas chances tiene de "
               "terminar yendose. Este grafico muestra como esta hoy la base activa segun ese arranque.")
    def _tier(n):
        return "Arranque completo" if n == 0 else "Buen arranque" if n == 1 else "Arranque parcial" if n == 2 else "Arranque critico"
    sa = act.copy()
    sa["tier"] = sa["fallas_activacion"].apply(_tier)
    orden = ["Arranque completo", "Buen arranque", "Arranque parcial", "Arranque critico"]
    colmap = {"Arranque completo": TEAL, "Buen arranque": "#5FBDB3", "Arranque parcial": AMBAR, "Arranque critico": CORAL}
    g = sa["tier"].value_counts().reindex(orden, fill_value=0).reset_index()
    g.columns = ["tier", "cuentas"]
    fig = px.bar(g, x="tier", y="cuentas", template=TPL, text="cuentas", color="tier",
                 color_discrete_map=colmap, category_orders={"tier": orden})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(height=320, showlegend=False, margin=dict(t=10, b=10),
                      yaxis_title="cuentas activas", xaxis_title="",
                      xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"<div style='font-size:0.85rem; color:{NAVY}; line-height:1.7;'>"
        f"<b style='color:{TEAL};'>Arranque completo</b>: completo los 5 pasos. &nbsp;|&nbsp; "
        f"<b style='color:#5FBDB3;'>Buen arranque</b>: 4 de 5, le falta 1. &nbsp;|&nbsp; "
        f"<b style='color:{AMBAR};'>Arranque parcial</b>: 3 de 5, le faltan 2. &nbsp;|&nbsp; "
        f"<b style='color:{CORAL};'>Arranque critico</b>: 2 o menos, le faltan 3 o mas."
        f"</div>", unsafe_allow_html=True)
    flojo = int((sa["fallas_activacion"] >= 2).sum())
    st.caption(f"{flojo:,} cuentas activas estan en arranque parcial o critico. Son las que conviene acompanar con onboarding antes de que se vayan.")



# ---------------- Churn ----------------
with T["Churn"]:
    PALETA = [TEAL, CORAL, AMBAR, NAVY, GRIS, TEAL_OSCURO, AZUL]

    # --- Donde se concentra el churn ---
    st.subheader("Donde se concentra el churn")
    cseg = st.columns([1, 2, 1])
    with cseg[1]:
        g = d[d[col_churn]]["segmento"].value_counts().reset_index()
        g.columns = ["segmento", "cuentas"]
        fig = px.pie(g, names="segmento", values="cuentas", template=TPL, hole=0.55,
                     color_discrete_sequence=[CORAL, AMBAR])
        fig.update_traces(textinfo="label+percent", textfont=dict(color="white", size=14))
        fig.update_layout(title="Churn por segmento", height=320, margin=dict(t=46, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    cols = st.columns(2)
    for col, titulo, cont in [("plan", "Tasa de churn por plan", cols[0]), ("pais", "Tasa de churn por pais", cols[1])]:
        with cont:
            g = churn_por(d, col)
            tope = g["tasa"].max() * 1.18 if len(g) else 1
            fig = px.bar(g, x=col, y="tasa", text=g["tasa"].map("{:.0%}".format), template=TPL,
                         color=col, color_discrete_sequence=PALETA)
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(title=titulo, yaxis_tickformat=".0%", height=320, margin=dict(t=40, b=10), showlegend=False,
                              yaxis_title="", xaxis_title="", yaxis=dict(range=[0, tope], automargin=True))
            st.plotly_chart(fig, use_container_width=True)

    # --- Retiros totales ---
    st.markdown("---")
    st.subheader("Retiros totales")
    st.caption("Todas las cuentas que hoy no estan activas, dejen o no una ficha de retiro. Es el churn completo medido por estado de cuenta.")
    no_act = d[d.estado_cuenta != "activo"]
    n_no = len(no_act)
    con_ficha = int(no_act["churn_retiro"].sum())
    mrr_total = d.mrr.sum()
    mt = st.columns(3)
    mt[0].metric("Cuentas en churn", f"{n_no:,}", f"{(n_no/len(d) if len(d) else 0):.0%} de la base", delta_color="off",
                 help="Cuentas en estado distinto de activo dentro del filtro.")
    mt[1].metric("MRR perdido / mes", f"${no_act.mrr.sum():,.0f}", f"{(no_act.mrr.sum()/mrr_total if mrr_total else 0):.0%} del MRR", delta_color="off",
                 help="Ingreso recurrente mensual de las cuentas que ya no estan activas.")
    mt[2].metric("Con ficha de retiro", f"{con_ficha:,}", f"{(con_ficha/n_no if n_no else 0):.0%} del churn", delta_color="off",
                 help="Cuantas de esas bajas dejaron un registro formal con el motivo.")
    cc1, cc2 = st.columns(2)
    cmap_estado = {"suspendido": AMBAR, "inactivo": GRIS, "cancelado": CORAL}
    with cc1:
        st.caption("Cuentas en churn por estado")
        g = no_act.estado_cuenta.value_counts().reset_index(); g.columns = ["estado", "cuentas"]
        g = g.sort_values("cuentas", ascending=True)
        tope = g["cuentas"].max() * 1.18 if len(g) else 1
        fig = px.bar(g, x="cuentas", y="estado", orientation="h", template=TPL, text="cuentas",
                     color="estado", color_discrete_map=cmap_estado)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=280, showlegend=False, margin=dict(t=10, b=10), yaxis_title="", xaxis_title="cuentas",
                          xaxis=dict(range=[0, tope]), yaxis=dict(automargin=True))
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        st.caption("MRR perdido por estado")
        g = no_act.groupby("estado_cuenta")["mrr"].sum().reset_index(); g.columns = ["estado", "mrr"]
        g = g.sort_values("mrr", ascending=True)
        tope = g["mrr"].max() * 1.28 if len(g) else 1
        fig = px.bar(g, x="mrr", y="estado", orientation="h", template=TPL, text=g["mrr"].map("${:,.0f}".format),
                     color="estado", color_discrete_map=cmap_estado)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=280, showlegend=False, margin=dict(t=10, b=10), yaxis_title="", xaxis_title="MRR USD",
                          xaxis=dict(range=[0, tope]), yaxis=dict(automargin=True))
        st.plotly_chart(fig, use_container_width=True)

    # --- Retiros identificados ---
    st.markdown("---")
    st.subheader("Retiros identificados")
    ret = d[d["churn_retiro"]].copy()
    if len(ret) == 0:
        st.info("No hay registros de retiro dentro del filtro actual.")
    else:
        voluntario = ret["tipo_retiro"].fillna("").str.lower().eq("voluntario").mean()
        nps_med = ret["nps_salida"].mean()
        m = st.columns(3)
        m[0].metric("Cuentas en churn con motivo identificado", f"{len(ret):,}", delta_color="off",
                    help="Cuentas en churn que dejaron un registro formal con el motivo de baja.")
        m[1].metric("Bajas voluntarias", f"{voluntario:.0%}", delta_color="off",
                    help="Voluntario = el cliente decide irse. Involuntario suele ser corte por falta de pago.")
        m[2].metric("NPS de salida", f"{nps_med:.1f}" if pd.notna(nps_med) else "s/d", delta_color="off",
                    help="Promedio del NPS al momento de la baja. La mitad de las fichas lo tienen vacio y la escala es 1 a 6.")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Motivo principal de baja")
            g = ret["motivo_principal"].dropna().value_counts().reset_index()
            g.columns = ["motivo", "casos"]; g = g.sort_values("casos", ascending=True)
            tope = g["casos"].max() * 1.18 if len(g) else 1
            fig = px.bar(g, x="casos", y="motivo", orientation="h", template=TPL, text="casos", color_discrete_sequence=[CORAL])
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(height=360, yaxis_title="", xaxis_title="casos", xaxis=dict(range=[0, tope]), yaxis=dict(automargin=True))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.caption("Voluntario vs involuntario")
            g = ret["tipo_retiro"].fillna("sin dato").str.capitalize().value_counts().reset_index()
            g.columns = ["tipo", "casos"]
            fig = px.pie(g, names="tipo", values="casos", template=TPL, hole=0.5,
                         color="tipo", color_discrete_map={"Voluntario": CORAL, "Involuntario": AMBAR, "Sin dato": GRIS})
            fig.update_traces(textinfo="label+percent", textfont=dict(color="white", size=13))
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            st.caption("Con que producto lo reemplazan")
            g = ret["lo_reemplaza_con"].dropna()
            g = g[~g.str.lower().isin(["", "ninguno", "nada", "n/a"])].value_counts().reset_index()
            g.columns = ["producto", "casos"]; g = g.sort_values("casos", ascending=True)
            if len(g):
                tope = g["casos"].max() * 1.18
                fig = px.bar(g, x="casos", y="producto", orientation="h", template=TPL, text="casos", color_discrete_sequence=[TEAL_OSCURO])
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(height=320, yaxis_title="", xaxis_title="casos", xaxis=dict(range=[0, tope]), yaxis=dict(automargin=True))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sin datos de reemplazo en el filtro.")
        with c4:
            st.caption("NPS al momento de la salida")
            g = ret["nps_salida"].dropna()
            if len(g):
                gg = g.value_counts().sort_index().reset_index()
                gg.columns = ["nps", "casos"]
                fig = px.bar(gg, x="nps", y="casos", template=TPL, text="casos",
                             color="nps", color_continuous_scale=[CORAL, AMBAR, TEAL])
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(height=320, yaxis_title="cuentas", xaxis_title="NPS de salida", coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sin NPS de salida en el filtro.")

    # --- Churn por pais (mapa, al final) ---
    st.markdown("---")
    st.subheader("Churn por pais")
    g = d.groupby("pais").agg(cuentas=("user_id", "count"), churn=(col_churn, "sum")).reset_index()
    g["tasa"] = g["churn"] / g["cuentas"]
    mrr_churn = d[d[col_churn]].groupby("pais")["mrr"].sum()
    g["mrr_churn"] = g["pais"].map(mrr_churn).fillna(0)
    g["lat"] = g["pais"].map(lambda p: CENTROIDES.get(p, (None, None))[0])
    g["lon"] = g["pais"].map(lambda p: CENTROIDES.get(p, (None, None))[1])
    g = g.dropna(subset=["lat", "lon"])
    fig = px.scatter_geo(g, lat="lat", lon="lon", size="cuentas", color="tasa", hover_name="pais",
                         hover_data={"cuentas": True, "tasa": ":.1%", "mrr_churn": ":$,.0f", "lat": False, "lon": False},
                         color_continuous_scale=ESCALA_CHURN, size_max=55, template=TPL)
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="#cccccc", showland=True,
                    landcolor="#f7f7f7", showocean=True, oceancolor="#eaf2f8")
    fig.update_layout(height=460, margin=dict(t=10, b=10), coloraxis_colorbar_title="churn")
    st.plotly_chart(fig, use_container_width=True)
    g2 = g[["pais", "cuentas", "churn", "tasa", "mrr_churn"]].sort_values("tasa", ascending=False).copy()
    g2["tasa"] = g2["tasa"].map("{:.1%}".format); g2["mrr_churn"] = g2["mrr_churn"].map("${:,.0f}".format)
    g2.columns = ["Pais", "Cuentas", "Churn", "Tasa churn", "MRR en churn"]
    sty = (g2.style.hide(axis="index")
           .set_properties(**{"text-align": "center"})
           .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]))
    st.table(sty)

    # --- Detalle de cuentas en churn ---
    st.markdown("---")
    st.subheader("Detalle de cuentas en churn")
    st.caption("Las cuentas consideradas churn segun la definicion elegida en el filtro lateral. Se puede ordenar por cualquier columna y descargar.")
    chd = d[d[col_churn]].copy()
    cols_show = [c for c in ["user_id", "nombre_empresa", "segmento", "pais", "plan", "mrr",
                             "estado_cuenta", "fecha_ultimo_pago", "fallas_activacion"] if c in chd.columns]
    tabla = chd[cols_show].rename(columns={"mrr": "mrr_usd", "fallas_activacion": "fallas_arranque"})
    st.dataframe(tabla, hide_index=True, use_container_width=True)
    st.download_button("Descargar CSV de cuentas en churn",
                       tabla.to_csv(index=False).encode("utf-8"),
                       "cuentas_en_churn.csv", "text/csv")



# ---------------- Negocio ----------------
with T["Negocio"]:
    st.subheader("Metricas de negocio")
    st.caption("Salud de la base que hoy paga. Todo en esta seccion se calcula sobre las cuentas activas.")
    act = d[d.estado_cuenta == "activo"].copy()
    n_act = len(act)

    # --- KPIs de la base activa ---
    k = st.columns(4)
    k[0].metric("MRR activo", f"${act.mrr.sum():,.0f}", delta_color="off",
                help="Ingreso recurrente mensual de las cuentas activas.")
    k[1].metric("ARPU", f"${(act.mrr.mean() if n_act else 0):,.1f}", delta_color="off",
                help="Ingreso promedio por cuenta activa (MRR / cuentas activas).")
    if "sesiones_promedio_semana" in act.columns:
        ses = pd.to_numeric(act["sesiones_promedio_semana"], errors="coerce").mean()
        k[2].metric("Sesiones / semana", f"{ses:.1f}", delta_color="off",
                    help="Sesiones promedio por semana de las cuentas activas. Mide enganche.")
    if "usuarios_adicionales" in act.columns:
        ua = pd.to_numeric(act["usuarios_adicionales"], errors="coerce").mean()
        k[3].metric("Usuarios adicionales", f"{ua:.1f}", delta_color="off",
                    help="Promedio de usuarios extra por cuenta activa. Proxy de expansion dentro de la cuenta.")

    st.markdown("---")

    # --- MRR por plan y por segmento (solo activas) ---
    c1, c2 = st.columns(2)
    with c1:
        st.caption("MRR por plan (cuentas activas)")
        g = act.groupby("plan")["mrr"].sum().reset_index().sort_values("mrr")
        tope = g["mrr"].max() * 1.25 if len(g) else 1
        fig = px.bar(g, x="mrr", y="plan", orientation="h", template=TPL, text=g["mrr"].map("${:,.0f}".format),
                     color="plan", color_discrete_sequence=[TEAL, CORAL, AMBAR, NAVY, GRIS, TEAL_OSCURO])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10), xaxis_title="MRR USD", yaxis_title="",
                          xaxis=dict(range=[0, tope]), yaxis=dict(automargin=True))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.caption("MRR por segmento (cuentas activas)")
        g = act.groupby("segmento")["mrr"].sum().reset_index()
        fig = px.pie(g, names="segmento", values="mrr", template=TPL, hole=0.5, color_discrete_sequence=[TEAL, NAVY])
        fig.update_traces(textinfo="label+percent", textfont=dict(color="white", size=13))
        fig.update_layout(height=300, margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Completitud de hitos y adopcion de modulos (cuentas activas) ---
    st.markdown("##### Completitud de hitos y adopcion de modulos")
    st.caption("Que porcentaje de las cuentas activas completo cada hito de arranque y activo cada modulo. "
               "En verde lo que se completo, en rojo lo que falta. Ordenado de mayor a menor adopcion.")
    def pct_si(col):
        return act[col].fillna("").str.lower().eq("si").mean() if col in act.columns else None
    hitos = []
    if "configuracion_empresa_completa" in act.columns: hitos.append(("Config. empresa", pct_si("configuracion_empresa_completa")))
    if "plan_cuentas_configurado" in act.columns: hitos.append(("Plan de cuentas", pct_si("plan_cuentas_configurado")))
    if "empleados_cargados" in act.columns: hitos.append(("Empleados cargados", pct_si("empleados_cargados")))
    if "facturas_emitidas_mes1" in act.columns:
        hitos.append(("Primera factura", (pd.to_numeric(act["facturas_emitidas_mes1"], errors="coerce").fillna(0) > 0).mean()))
    if "integracion_banco_conectada" in act.columns: hitos.append(("Banco conectado", pct_si("integracion_banco_conectada")))
    if "modulo_cxc_activo" in act.columns: hitos.append(("Modulo CxC", pct_si("modulo_cxc_activo")))
    if "modulo_nomina_activo" in act.columns: hitos.append(("Modulo Nomina", pct_si("modulo_nomina_activo")))
    if "modulo_inventario_activo" in act.columns: hitos.append(("Modulo Inventario", pct_si("modulo_inventario_activo")))
    hitos = [(h, p) for h, p in hitos if p is not None]
    if hitos:
        hitos.sort(key=lambda x: x[1])  # ascending para que el de mayor adopcion quede arriba en barra horizontal
        orden = [h for h, _ in hitos]
        filas = []
        for h, p in hitos:
            filas.append({"hito": h, "estado": "Completado", "pct": p})
            filas.append({"hito": h, "estado": "No completado", "pct": 1 - p})
        dfh = pd.DataFrame(filas)
        fig = px.bar(dfh, x="pct", y="hito", color="estado", orientation="h", template=TPL, barmode="stack",
                     text=dfh["pct"].map("{:.0%}".format),
                     color_discrete_map={"Completado": TEAL, "No completado": CORAL},
                     category_orders={"hito": orden, "estado": ["Completado", "No completado"]})
        fig.update_traces(textposition="inside", insidetextanchor="middle", textfont=dict(color="white", size=11))
        fig.update_layout(height=380, xaxis_tickformat=".0%", margin=dict(t=10, b=10),
                          yaxis_title="", xaxis_title="", legend_title="",
                          legend=dict(orientation="h", y=1.08, x=0), yaxis=dict(automargin=True))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Los hitos de arranque (configurar empresa, primera factura, plan de cuentas) tienen alta adopcion; "
                   "los modulos de nicho como nomina e inventario, mucho menor. Esa diferencia es la que separa una "
                   "cuenta bien activada de una que apenas usa el producto.")



# ---------------- Causa raiz ----------------
with T["Causa raiz"]:
    st.subheader("Cómo llegamos a la causa raíz")
    st.markdown("Antes de señalar un culpable medimos el efecto de **todas** las variables sobre el churn, no solo "
                "las de activación. Con ese panorama completo separamos lo que explica la baja de lo que solo la acompaña.")

    base = d[col_churn].mean()
    si = lambda c: d[c].fillna("").str.lower().eq("si")
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    sweep = []
    def add(lbl, cat, mask):
        m = mask.fillna(False)
        if m.any():
            sweep.append({"senal": lbl, "categoria": cat, "churn": d[m][col_churn].mean()})

    # --- 1. Barrido completo ---
    st.markdown("##### 1. Medimos el efecto de cada señal")
    if "sesiones_promedio_semana" in d.columns: add("Menos de 1 sesión por semana", "Síntoma", num("sesiones_promedio_semana").fillna(0) < 1)
    if "login_ultimos_30_dias" in d.columns: add("Sin login en 30 días", "Síntoma", ~si("login_ultimos_30_dias"))
    if "dias_primer_factura" in d.columns: add("Tardó más de 14 días en facturar", "Síntoma", num("dias_primer_factura") > 14)
    add("Configuración de empresa incompleta", "Activación", ~si("configuracion_empresa_completa"))
    add("Primera factura no emitida", "Activación", num("facturas_emitidas_mes1").fillna(0).eq(0))
    add("Plan de cuentas sin configurar", "Activación", ~si("plan_cuentas_configurado"))
    add("Empleados no cargados", "Activación", ~si("empleados_cargados"))
    add("Módulo CxC no activo", "Activación", ~si("modulo_cxc_activo"))
    add("Banco no conectado", "Activación", ~si("integracion_banco_conectada"))
    if "reportes_generados_mes3" in d.columns: add("Sin reportes en el mes 3", "Intensidad de uso", num("reportes_generados_mes3").eq(0))
    if "facturas_emitidas_mes3" in d.columns: add("Sin facturas en el mes 3", "Intensidad de uso", num("facturas_emitidas_mes3").eq(0))
    if "chat_reaperturas" in d.columns: add("Ticket de soporte reabierto", "Soporte", num("chat_reaperturas").fillna(0) > 0)
    if "chat_no_resueltos" in d.columns: add("Ticket sin resolver", "Soporte", num("chat_no_resueltos").fillna(0) > 0)
    if "tel_escalo" in d.columns: add("Llamada escalada", "Soporte", num("tel_escalo").fillna(0) > 0)
    add("Módulo de nómina no activo", "Nicho", ~si("modulo_nomina_activo"))
    add("Módulo de inventario no activo", "Nicho", ~si("modulo_inventario_activo"))
    sw = pd.DataFrame(sweep).dropna(subset=["churn"]).sort_values("churn")
    cmap_cat = {"Síntoma": GRIS, "Activación": TEAL, "Intensidad de uso": NAVY, "Soporte": AMBAR, "Nicho": "#C8D0D8"}
    figW = px.bar(sw, x="churn", y="senal", color="categoria", orientation="h", template=TPL,
                  text=sw["churn"].map("{:.0%}".format), color_discrete_map=cmap_cat)
    figW.update_traces(textposition="outside", cliponaxis=False)
    figW.add_vline(x=base, line_dash="dash", line_color=NAVY)
    figW.add_annotation(x=base, y=1.02, yref="paper", text=f"churn base {base:.0%}", showarrow=False, font=dict(color=NAVY, size=11))
    figW.update_layout(height=540, xaxis_tickformat=".0%", xaxis=dict(range=[0, 1.12]), margin=dict(t=24, b=10),
                       yaxis_title="", xaxis_title="churn cuando la señal está presente", legend_title="", yaxis=dict(automargin=True))
    st.plotly_chart(figW, use_container_width=True)
    st.caption("Cada barra es la tasa de churn de las cuentas que tienen esa señal, frente al churn base. Un **síntoma** "
               "ocurre al final, pegado a la baja; una **causa** ocurre temprano y se puede accionar.")

    # --- 2. Sintoma vs causa ---
    st.markdown("##### 2. ¿Síntoma o causa?")
    st.markdown("Arriba de todo aparecen el login y las sesiones: predicen el churn casi perfecto, pero son **síntomas**. "
                "Casi ninguna cuenta en churn tiene login reciente, y ninguna cuenta activa está así, de modo que \"no entrar "
                "en 30 días\" prácticamente describe el abandono en lugar de explicarlo, y llega tarde para actuar. Por eso "
                "miramos lo que pasa antes: la activación.")

    # --- 3. Activacion: las 5 senales ---
    st.markdown("##### 3. El bloque que sí explica: la activación")
    senales5 = [("configuracion_empresa_completa", "Configuración de empresa"),
                ("plan_cuentas_configurado", "Plan de cuentas"),
                ("empleados_cargados", "Empleados cargados"),
                ("modulo_cxc_activo", "Módulo CxC")]
    filas = []
    for col, lbl in senales5:
        if col in d.columns:
            falla = ~si(col)
            filas.append({"senal": lbl, "churn_falla": d[falla][col_churn].mean(), "churn_ok": d[~falla][col_churn].mean()})
    fmask = num("facturas_emitidas_mes1").fillna(0).eq(0)
    filas.append({"senal": "Primera factura", "churn_falla": d[fmask][col_churn].mean(), "churn_ok": d[~fmask][col_churn].mean()})
    sg = pd.DataFrame(filas).sort_values("churn_falla")
    sm = sg.melt(id_vars="senal", value_vars=["churn_falla", "churn_ok"], var_name="grupo", value_name="tasa")
    sm["grupo"] = sm["grupo"].map({"churn_falla": "Si falla", "churn_ok": "Si está ok"})
    figS = px.bar(sm, x="tasa", y="senal", color="grupo", barmode="group", orientation="h",
                  template=TPL, text=sm["tasa"].map("{:.0%}".format),
                  color_discrete_map={"Si falla": CORAL, "Si está ok": TEAL})
    figS.update_traces(textposition="outside", cliponaxis=False)
    figS.update_layout(height=340, xaxis_tickformat=".0%", margin=dict(t=10, b=10),
                       yaxis_title="", xaxis_title="tasa de churn", legend_title="", xaxis=dict(range=[0, 1]), yaxis=dict(automargin=True))
    st.plotly_chart(figS, use_container_width=True)
    st.caption("Las cinco señales del arranque, cada una por separado, multiplican el churn cuando fallan. Son tempranas "
               "y accionables, lo que las vuelve una causa y no un síntoma.")

    # --- 4. La escalera ---
    st.markdown("##### 4. La prueba: cuántas señales fallan")
    esc = d.groupby("fallas_activacion").agg(cuentas=("user_id", "count"), churn=(col_churn, "mean")).reset_index()
    esc = esc.sort_values("fallas_activacion")
    esc["etq"] = esc["fallas_activacion"].astype(int).astype(str)
    figE = px.bar(esc, x="etq", y="churn", text=esc["churn"].map("{:.0%}".format),
                  template=TPL, color="churn", color_continuous_scale=[TEAL, AMBAR, CORAL])
    figE.update_traces(textposition="outside", cliponaxis=False,
                       customdata=esc[["cuentas"]], hovertemplate="%{x} señales fallan<br>churn %{y:.0%}<br>%{customdata[0]} cuentas<extra></extra>")
    figE.update_layout(title="Churn según cuántas señales de activación fallan (de 5)",
                       height=400, yaxis_tickformat=".0%", coloraxis_showscale=False,
                       yaxis_title="tasa de churn", xaxis_title="señales que fallan", margin=dict(t=48, b=10))
    st.plotly_chart(figE, use_container_width=True)
    st.caption("Contamos, por cuenta, cuántas de las cinco señales fallan. El churn sube de forma escalonada, de 2% con el "
               "arranque completo a 93% sin ninguna señal. Esa progresión ordenada confirma que la activación es el motor del churn.")

    # --- 5. Lo que quedo afuera ---
    st.markdown("##### 5. Lo que quedó afuera")
    factores = []
    ea = d.groupby("fallas_activacion")[col_churn].mean()
    if len(ea): factores.append(("Activación (0 a 5 fallas)", ea.min(), ea.max()))
    for col, lbl in [("pais", "País"), ("plan", "Plan"), ("segmento", "Segmento")]:
        if col in d.columns:
            gg = d.groupby(col)[col_churn].mean()
            if len(gg): factores.append((lbl, gg.min(), gg.max()))
    fr = pd.DataFrame([{"factor": f, "spread": hi - lo, "lo": lo, "hi": hi} for f, lo, hi in factores]).sort_values("spread")
    tope = fr["spread"].max() * 1.35 if len(fr) else 1
    figR = px.bar(fr, x="spread", y="factor", orientation="h", template=TPL,
                  text=fr.apply(lambda r: f"{r.lo:.0%} a {r.hi:.0%}", axis=1),
                  color="spread", color_continuous_scale=[GRIS, AMBAR, CORAL])
    figR.update_traces(textposition="outside", cliponaxis=False)
    figR.update_layout(height=300, coloraxis_showscale=False, xaxis_tickformat=".0%",
                       yaxis_title="", xaxis_title="cuánto separa el churn (mayor menos menor)",
                       margin=dict(t=10, b=10), xaxis=dict(range=[0, tope]), yaxis=dict(automargin=True))
    st.plotly_chart(figR, use_container_width=True)
    st.caption("El perfil del cliente (segmento, plan, país) casi no separa el churn. También quedaron fuera del índice el "
               "banco conectado, por efecto débil, y los módulos de nicho como nómina e inventario, que usa poca gente.")

    st.info("El camino, en una línea: medimos todo, descartamos los síntomas que avisan tarde, y nos quedamos con las cinco "
            "señales de activación, que son tempranas, accionables y de efecto fuerte. La palanca contra el churn está en el arranque.")



# ---------------- Soporte ----------------
with T["Soporte"]:
    if not HAY_SOPORTE:
        st.warning("No se detectaron archivos de soporte en el repo. Para ver esta seccion, suba los CSV de chat, "
                   "whatsapp y telefono conservando sus columnas originales (csat_score, sentimiento_usuario, nps_post_llamada).")
    else:
        # --- Cobertura de soporte sobre la base (primero) ---
        st.subheader("Cobertura de soporte sobre la base")
        st.caption("Cuanto del churn pasa por soporte y, sobre todo, si el soporte funciono. La friccion (ticket reabierto, "
                   "sin resolver, escalado o con sentimiento negativo) es lo que se asocia al churn, no el volumen.")
        cov = st.columns(4)
        cov[0].metric("Cuentas con algun ticket", f"{int(d['usa_soporte'].sum()):,}", f"{d['usa_soporte'].mean():.0%} de la base", delta_color="off")
        cov[1].metric("Tickets totales", f"{int(d['tickets_total'].sum()):,}", delta_color="off")
        cov[2].metric("Cuentas con friccion", f"{int(d['friccion'].sum()):,}", f"{d['friccion'].mean():.0%} de la base", delta_color="off")
        churn_fric = d[d['friccion']][col_churn].mean() if d['friccion'].any() else 0
        churn_sin = d[~d['friccion']][col_churn].mean() if (~d['friccion']).any() else 0
        cov[3].metric("Churn con friccion vs sin", f"{churn_fric:.0%} vs {churn_sin:.0%}", delta_color="off",
                      help="Tasa de churn de las cuentas que tuvieron alguna mala experiencia de soporte frente a las que no.")

        st.markdown("---")
        st.subheader("Analisis por canal de soporte")
        canal = st.selectbox("Elegi el canal", ["Chat", "WhatsApp", "Telefono"])
        cmap = d.set_index("user_id")[col_churn]

        if canal == "Chat" and chat is not None:
            c = chat.copy(); c["_churn"] = c["user_id"].map(cmap)
            m = st.columns(4)
            m[0].metric("Tickets", f"{len(c):,}")
            m[1].metric("Tasa de resolucion", f"{(c.resuelto=='si').mean():.0%}")
            m[2].metric("Tasa de reapertura", f"{(c.reapertura=='si').mean():.0%}")
            m[3].metric("CSAT promedio", f"{c.csat_score.mean():.2f} / 5")
            m2 = st.columns(3)
            m2[0].metric("Cuentas atendidas", f"{c['user_id'].nunique():,}")
            m2[1].metric("Tickets por cuenta", f"{len(c)/c['user_id'].nunique():.1f}")
            m2[2].metric("CSAT activas vs churn", f"{c[c._churn==False].csat_score.mean():.2f} vs {c[c._churn==True].csat_score.mean():.2f}",
                         help="CSAT promedio de tickets de cuentas hoy activas frente a las que ya churnearon.")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Tickets por tema")
                g = c.tema_principal.value_counts().reset_index(); g.columns = ["tema", "tickets"]
                fig = px.bar(g, x="tickets", y="tema", orientation="h", template=TPL, text="tickets", color_discrete_sequence=[AZUL])
                estilo_barras(fig)
                fig.update_layout(height=340, yaxis_title="", yaxis={"categoryorder": "total ascending", "automargin": True})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.caption("CSAT promedio por tema (mas bajo = mas friccion)")
                g = c.groupby("tema_principal").csat_score.mean().reset_index().sort_values("csat_score")
                fig = px.bar(g, x="csat_score", y="tema_principal", orientation="h", template=TPL, color="csat_score",
                             text=g["csat_score"].map("{:.1f}".format),
                             color_continuous_scale=[CORAL, AMBAR, TEAL], range_x=[1, 5])
                fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color=NAVY))
                fig.update_layout(height=340, yaxis_title="", coloraxis_showscale=False, yaxis=dict(automargin=True))
                st.plotly_chart(fig, use_container_width=True)
            t1, t2 = st.columns(2)
            t1.metric("Primera respuesta (mediana)", f"{c.tiempo_primera_respuesta_min.median():.0f} min")
            t2.metric("Tiempo de resolucion (mediana)", f"{c.tiempo_resolucion_hs.median():.1f} hs")

        elif canal == "WhatsApp" and wa is not None:
            c = wa.copy(); c["_churn"] = c["user_id"].map(cmap)
            m = st.columns(4)
            m[0].metric("Tickets", f"{len(c):,}")
            m[1].metric("Resuelto en conversacion", f"{(c.resuelto_en_conversacion=='si').mean():.0%}")
            m[2].metric("Deriva a agente", f"{(c.deriva_a_agente=='si').mean():.0%}")
            m[3].metric("Sentimiento negativo", f"{(c.sentimiento_usuario=='negativo').mean():.0%}")
            m2 = st.columns(3)
            m2[0].metric("Cuentas atendidas", f"{c['user_id'].nunique():,}")
            m2[1].metric("Tickets por cuenta", f"{len(c)/c['user_id'].nunique():.1f}")
            neg_act = (c[c._churn==False].sentimiento_usuario=='negativo').mean()
            neg_chu = (c[c._churn==True].sentimiento_usuario=='negativo').mean()
            m2[2].metric("Negativo activas vs churn", f"{neg_act:.0%} vs {neg_chu:.0%}",
                         help="Porcentaje de tickets con sentimiento negativo, en cuentas activas frente a las que churnearon.")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Sentimiento del usuario")
                g = c.sentimiento_usuario.value_counts().reset_index(); g.columns = ["sentimiento", "tickets"]
                fig = px.pie(g, names="sentimiento", values="tickets", template=TPL, hole=0.45,
                             color="sentimiento", color_discrete_map={"positivo": VERDE, "neutro": GRIS, "negativo": ROJO})
                fig.update_traces(textinfo="label+percent", textfont=dict(color="white", size=13))
                fig.update_layout(height=330, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.caption("Sentimiento negativo por tema")
                g = c.groupby("tema_principal").apply(lambda x: (x.sentimiento_usuario == "negativo").mean()).reset_index()
                g.columns = ["tema", "neg"]; g = g.sort_values("neg", ascending=True)
                tope = g["neg"].max() * 1.18 if len(g) else 1
                fig = px.bar(g, x="neg", y="tema", orientation="h", template=TPL, text=g["neg"].map("{:.0%}".format), color_discrete_sequence=[ROJO])
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(height=330, margin=dict(t=10, b=10), yaxis_title="", xaxis_tickformat=".0%",
                                  xaxis=dict(range=[0, tope]), yaxis=dict(automargin=True))
                st.plotly_chart(fig, use_container_width=True)
            st.metric("Primera respuesta (mediana)", f"{c.tiempo_primera_respuesta_min.median():.0f} min")

        elif canal == "Telefono" and tel is not None:
            c = tel.copy(); c["_churn"] = c["user_id"].map(cmap)
            m = st.columns(4)
            m[0].metric("Llamadas", f"{len(c):,}")
            m[1].metric("Tasa de resolucion", f"{(c.resuelto=='si').mean():.0%}")
            m[2].metric("Tasa de escalamiento", f"{(c.escalo_a_especialista=='si').mean():.0%}")
            m[3].metric("NPS post llamada", f"{c.nps_post_llamada.mean():.2f} / 5")
            m2 = st.columns(3)
            m2[0].metric("Cuentas atendidas", f"{c['user_id'].nunique():,}")
            m2[1].metric("Llamadas por cuenta", f"{len(c)/c['user_id'].nunique():.1f}")
            m2[2].metric("NPS activas vs churn", f"{c[c._churn==False].nps_post_llamada.mean():.2f} vs {c[c._churn==True].nps_post_llamada.mean():.2f}",
                         help="NPS post llamada de cuentas activas frente a las que churnearon.")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Llamadas por tema")
                g = c.tema_principal.value_counts().reset_index(); g.columns = ["tema", "llamadas"]
                fig = px.bar(g, x="llamadas", y="tema", orientation="h", template=TPL, text="llamadas", color_discrete_sequence=[AZUL])
                estilo_barras(fig)
                fig.update_layout(height=330, yaxis_title="", yaxis={"categoryorder": "total ascending", "automargin": True})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.caption("Motivos de escalamiento")
                g = c.motivo_escala.dropna().value_counts().reset_index(); g.columns = ["motivo", "casos"]
                fig = px.bar(g, x="casos", y="motivo", orientation="h", template=TPL, text="casos", color_discrete_sequence=[NARANJA])
                estilo_barras(fig)
                fig.update_layout(height=330, yaxis_title="", yaxis={"categoryorder": "total ascending", "automargin": True})
                st.plotly_chart(fig, use_container_width=True)
            st.metric("Duracion de llamada (mediana)", f"{c.duracion_llamada_min.median():.0f} min")



# ---------------- Calidad de datos ----------------
with T["Calidad de datos"]:
    st.subheader("Panel de calidad de datos")
    st.caption("Inconsistencias detectadas. No se borran filas: se marcan y se decide su uso.")
    flags = [
        ("flag_estado_sin_retiro", "Cuenta no activa sin registro de retiro", "estado_cuenta como verdad de churn"),
        ("flag_activacion_contradice", "Marca no facturo pero tiene facturas mes 1", "Activacion desde facturas_mes1 > 0"),
        ("flag_fecha_invertida", "Ultimo pago anterior al registro", "Excluir de antiguedad y recencia"),
        ("flag_retiro_fecha_invertida", "Fecha de retiro anterior al ultimo pago", "No usar para recencia"),
        ("flag_login_no_activo", "Login reciente en cuenta no activa", "No usar login_30 como actividad"),
        ("flag_pago_geo", "Metodo de pago geograficamente imposible", "No usar metodo_pago por pais"),
    ]
    filas = []
    for col, desc, acc in flags:
        if col in d.columns:
            n = int(d[col].sum())
            filas.append({"Hallazgo": desc, "Cuentas afectadas": n, "% del filtro": f"{(n/len(d) if len(d) else 0):.1%}", "Decision": acc})
    reg = pd.DataFrame(filas).sort_values("Cuentas afectadas", ascending=False)
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(reg, x="Cuentas afectadas", y="Hallazgo", orientation="h", template=TPL, text="Cuentas afectadas", color_discrete_sequence=[NARANJA])
        fig.update_layout(height=340, margin=dict(t=10, b=10), yaxis_title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        afect = d[[c for c, _, _ in flags if c in d.columns]].any(axis=1).sum()
        st.metric("Cuentas con al menos una inconsistencia", f"{int(afect):,}", f"{(afect/len(d) if len(d) else 0):.0%} del filtro")
        st.metric("MRR no explicado por retiros", f"${(d.loc[d['flag_estado_sin_retiro'],'mrr'].sum()):,.0f}")
        if HAY_SOPORTE:
            st.caption("Soporte: telefono usa NPS en escala 1 a 5 (no 0 a 10) y motivo_escala esta vacio salvo en escalamientos. Documentado, no afecta las tasas.")
    st.dataframe(reg, hide_index=True, use_container_width=True)
