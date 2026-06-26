"""
Dashboard de Churn CloudMetrics
Fuentes: usuarios + retiros + uso de producto + soporte (chat, whatsapp, telefono).
Sistema de control continuo para el equipo de Customer Experience.
"""

import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Churn CloudMetrics", layout="wide", page_icon="📊")

# Paleta inspirada en la identidad de Alegra (turquesa de marca).
# Si conseguis el hex oficial, cambialo solo aca y se propaga a todo el dashboard.
TEAL = "#00B8A9"        # color de marca, positivo, cuentas activas
TEAL_OSCURO = "#0A8C82"
NAVY = "#16284B"        # texto y titulos
CORAL = "#FF6B6B"       # churn, negativo
AMBAR = "#F4A340"       # advertencia, suspendido
GRIS = "#9AA7B0"        # neutro, inactivo
# Alias para mantener compatibilidad con el resto del dashboard
VERDE = TEAL; ROJO = CORAL; AZUL = TEAL_OSCURO; NARANJA = AMBAR
TPL = "plotly_white"
ESCALA_CHURN = [TEAL, AMBAR, CORAL]

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
</style>""", unsafe_allow_html=True)


def logo_cloudmetrics():
    # Logo inventado: nube + barras de metrica, en turquesa de marca. Compacto (sin saltos de linea).
    return (
        f'<svg width="54" height="54" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">'
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
_motivo = ("Este tablero nace porque en CloudMetrics el churn existe pero estaba repartido entre soporte, "
           "producto, pagos y retiros, sin una medida unica ni confiable. Reune esas fuentes en un solo lugar "
           "para que el equipo de Customer Experience pueda medir el churn de forma estandar, entender sus "
           "causas raiz y actuar de forma continua, no como un analisis puntual.")
_header = (
    '<div style="text-align:center; margin-bottom:0.2rem;">'
    '<div style="display:flex; justify-content:center; align-items:center; gap:12px;">'
    f'{logo_cloudmetrics()}'
    f'<h1 style="margin:0; color:{NAVY};">CloudMetrics · Centro de Control de Churn</h1>'
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

r1 = st.columns(4)
r1[0].metric("Cuentas totales", f"{total:,}",
             help="Cantidad de cuentas que entran en el filtro actual.")
r1[1].metric("Churn real", f"{tasa:.1%}",
             help="Cuentas en estado distinto de activo (cancelado, inactivo o suspendido) sobre el total. Es la verdad operativa del churn.")
r1[2].metric("Churn teorico (retiros)", f"{churn_teorico:.1%}",
             help="Cuentas con ficha formal de retiro sobre el total. Es un piso: solo 1 de cada 3 bajas tiene registro, por eso es menor al churn real.")
r1[3].metric("MRR activo", f"${mrr_activo:,.0f}",
             help="Ingreso recurrente mensual que aportan hoy las cuentas activas, en USD.")

r2 = st.columns(3)
r2[0].metric("MRR en riesgo", f"${mrr_riesgo:,.0f}",
             help="Ingreso recurrente mensual de las cuentas que ya churnearon. Es la perdida que estamos midiendo.")
r2[1].metric("ARPU activos", f"${arpu:,.0f}",
             help="Ingreso promedio por cuenta activa. MRR activo dividido por cuentas activas.")
r2[2].metric("Llegaron a facturar", f"{tasa_activacion:.0%}",
             help="Porcentaje de cuentas que emitieron al menos una factura en su primer mes. Mide si el cliente realmente arranco a usar el producto.")
st.markdown("---")

nombres_tabs = ["Resumen", "Mapa", "Negocio", "Causa raiz", "Soporte", "Calidad de datos"]
tabs = st.tabs(nombres_tabs)
T = dict(zip(nombres_tabs, tabs))


def churn_por(data, col):
    g = data.groupby(col).agg(cuentas=("user_id", "count"), churn=(col_churn, "sum")).reset_index()
    g["tasa"] = g["churn"] / g["cuentas"]
    return g.sort_values("tasa", ascending=False)


# ---------------- Resumen ----------------
with T["Resumen"]:
    # ---- Cuentas activas hoy ----
    st.subheader("Cuentas activas hoy")
    st.caption("Como se compone la base que hoy sigue pagando. Util para entender quien es el cliente actual.")
    act = d[d.estado_cuenta == "activo"]
    cols = st.columns(4)
    cortes = [("segmento", "Por segmento", cols[0]), ("plan", "Por plan", cols[1]),
              ("pais", "Por pais", cols[2]), ("metodo_pago", "Por metodo de pago", cols[3])]
    for col, titulo, cont in cortes:
        with cont:
            g = act[col].value_counts().reset_index()
            g.columns = [col, "cuentas"]
            g = g.sort_values("cuentas", ascending=True)
            fig = px.bar(g, x="cuentas", y=col, orientation="h", template=TPL,
                         text="cuentas", color_discrete_sequence=[TEAL])
            fig.update_traces(textposition="outside")
            fig.update_layout(title=titulo, height=300, showlegend=False,
                              margin=dict(t=40, b=10), yaxis_title="", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    st.caption("Nota: el metodo de pago tiene inconsistencias geograficas documentadas (PSE y OXXO fuera de su pais). Tomar como referencia, no como dato confiable por pais.")

    st.markdown("---")

    # ---- Registros en el tiempo ----
    st.markdown("##### Cuentas registradas por mes")
    st.caption("Cuantas cuentas se sumaron cada mes, separando las que hoy siguen activas de las que ya churnearon. Muestra el crecimiento y como le fue a cada camada.")
    reg = d.dropna(subset=["fecha_registro"]).copy()
    reg["mes"] = reg["fecha_registro"].dt.to_period("M").dt.to_timestamp()
    reg["situacion"] = np.where(reg[col_churn], "Churneada", "Activa hoy")
    g = reg.groupby(["mes", "situacion"]).size().reset_index(name="cuentas")
    fig = px.bar(g, x="mes", y="cuentas", color="situacion", template=TPL, barmode="stack",
                 color_discrete_map={"Activa hoy": TEAL, "Churneada": CORAL})
    fig.update_layout(height=340, margin=dict(t=10, b=10), xaxis_title="", yaxis_title="cuentas",
                      legend_title="", legend=dict(orientation="h", y=1.1, x=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---- Donde se concentra el churn ----
    st.subheader("Donde se concentra el churn")
    st.caption("Tasa de churn por corte. Barra mas alta = grupo que mas se va.")
    cols = st.columns(3)
    for col, titulo, cont in [("segmento", "Por segmento", cols[0]), ("plan", "Por plan", cols[1]), ("pais", "Por pais", cols[2])]:
        with cont:
            g = churn_por(d, col)
            fig = px.bar(g, x=col, y="tasa", text=g["tasa"].map("{:.0%}".format), template=TPL, color_discrete_sequence=[CORAL])
            fig.update_traces(textposition="outside")
            fig.update_layout(title=titulo, yaxis_tickformat=".0%", height=320, margin=dict(t=40, b=10), showlegend=False, yaxis_title="", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("##### Composicion de la base por estado")
    g = d.estado_cuenta.value_counts().reset_index(); g.columns = ["estado", "cuentas"]
    fig = px.bar(g, x="cuentas", y="estado", orientation="h", template=TPL, color="estado",
                 color_discrete_map={"activo": TEAL, "suspendido": AMBAR, "inactivo": GRIS, "cancelado": CORAL}, text="cuentas")
    fig.update_layout(height=250, showlegend=False, margin=dict(t=10, b=10), yaxis_title="", xaxis_title="cuentas")
    st.plotly_chart(fig, use_container_width=True)


# ---------------- Mapa ----------------
with T["Mapa"]:
    st.subheader("Cuentas y churn por pais")
    st.caption("Tamano del circulo = cantidad de cuentas. Color = tasa de churn (mas rojo, mas churn).")
    g = d.groupby("pais").agg(cuentas=("user_id", "count"), churn=(col_churn, "sum"), mrr=("mrr", "sum")).reset_index()
    g["tasa"] = g["churn"] / g["cuentas"]
    g["lat"] = g["pais"].map(lambda p: CENTROIDES.get(p, (None, None))[0])
    g["lon"] = g["pais"].map(lambda p: CENTROIDES.get(p, (None, None))[1])
    g = g.dropna(subset=["lat", "lon"])
    fig = px.scatter_geo(g, lat="lat", lon="lon", size="cuentas", color="tasa", hover_name="pais",
                         hover_data={"cuentas": True, "tasa": ":.1%", "mrr": ":$,.0f", "lat": False, "lon": False},
                         color_continuous_scale=ESCALA_CHURN, size_max=55, template=TPL)
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="#cccccc", showland=True,
                    landcolor="#f7f7f7", showocean=True, oceancolor="#eaf2f8")
    fig.update_layout(height=500, margin=dict(t=10, b=10), coloraxis_colorbar_title="churn")
    st.plotly_chart(fig, use_container_width=True)
    g2 = g[["pais", "cuentas", "churn", "tasa", "mrr"]].sort_values("tasa", ascending=False).copy()
    g2["tasa"] = g2["tasa"].map("{:.1%}".format); g2["mrr"] = g2["mrr"].map("${:,.0f}".format)
    g2.columns = ["Pais", "Cuentas", "Churn", "Tasa churn", "MRR"]
    st.dataframe(g2, hide_index=True, use_container_width=True)


# ---------------- Negocio ----------------
with T["Negocio"]:
    st.subheader("Metricas de negocio")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("MRR por plan")
        g = d.groupby("plan")["mrr"].sum().reset_index().sort_values("mrr")
        fig = px.bar(g, x="mrr", y="plan", orientation="h", template=TPL, text=g["mrr"].map("${:,.0f}".format), color_discrete_sequence=[AZUL])
        fig.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10), xaxis_title="MRR USD", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.caption("MRR por segmento")
        g = d.groupby("segmento")["mrr"].sum().reset_index()
        fig = px.pie(g, names="segmento", values="mrr", template=TPL, hole=0.5, color_discrete_sequence=[AZUL, VERDE])
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("##### Embudo de activacion")
    st.caption("Cuantas cuentas completan cada hito del onboarding. Donde mas cae es donde se pierde valor.")
    hitos = [("configuracion_empresa_completa", "Config. empresa"), ("empleados_cargados", "Empleados cargados"),
             ("plan_cuentas_configurado", "Plan de cuentas"), ("primer_factura_emitida", "Primera factura"),
             ("integracion_banco_conectada", "Banco conectado")]
    filas = [{"hito": l, "cuentas": d[c].fillna("").str.lower().eq("si").sum()} for c, l in hitos if c in d.columns]
    if filas:
        fig = px.funnel(pd.DataFrame(filas), x="cuentas", y="hito", template=TPL, color_discrete_sequence=[VERDE])
        fig.update_layout(height=320, margin=dict(t=10, b=10), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)


# ---------------- Causa raiz ----------------
with T["Causa raiz"]:
    st.subheader("Que dispara el churn")
    st.caption("Tasa de churn segun comportamiento. La brecha entre barras es el tamano del problema.")

    def churn_binario(col, si_lbl, no_lbl):
        s = d[col].fillna("").str.lower().eq("si")
        out = [{"grupo": lbl, "tasa": d[s == v][col_churn].mean()} for v, lbl in [(True, si_lbl), (False, no_lbl)] if len(d[s == v])]
        return pd.DataFrame(out)

    cols = st.columns(3)
    pares = [("configuracion_empresa_completa", "Completo setup", "No completo", "Configuracion inicial"),
             ("integracion_banco_conectada", "Conecto banco", "No conecto", "Integracion bancaria")]
    for (col, si, no, tit), cont in zip(pares, cols):
        with cont:
            g = churn_binario(col, si, no)
            fig = px.bar(g, x="grupo", y="tasa", text=g["tasa"].map("{:.0%}".format), template=TPL, color="grupo", color_discrete_map={si: VERDE, no: ROJO})
            fig.update_traces(textposition="outside")
            fig.update_layout(title=tit, height=320, showlegend=False, yaxis_tickformat=".0%", margin=dict(t=40, b=10), yaxis_title="", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    with cols[2]:
        g = d.groupby("activado").agg(churn=(col_churn, "mean")).reset_index()
        g["grupo"] = g["activado"].map({True: "Facturo mes 1", False: "No facturo"})
        fig = px.bar(g, x="grupo", y="churn", text=g["churn"].map("{:.0%}".format), template=TPL, color="grupo", color_discrete_map={"Facturo mes 1": VERDE, "No facturo": ROJO})
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Activacion", height=320, showlegend=False, yaxis_tickformat=".0%", margin=dict(t=40, b=10), yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    if HAY_SOPORTE:
        st.markdown("##### Fricion de soporte y churn")
        st.caption("No es el volumen de soporte lo que predice el churn, sino que el soporte falle.")
        senales = []
        defs = [("Ticket reabierto", d.get("chat_reaperturas", 0) > 0),
                ("Ticket sin resolver", d.get("chat_no_resueltos", 0) > 0),
                ("Escalo a especialista", d.get("tel_escalo", 0) > 0),
                ("Sentimiento negativo", d.get("wa_negativos", 0) > 0)]
        for lbl, mask in defs:
            senales.append({"senal": lbl, "con": d[mask][col_churn].mean(), "sin": d[~mask][col_churn].mean()})
        s = pd.DataFrame(senales).melt(id_vars="senal", var_name="grupo", value_name="tasa")
        s["grupo"] = s["grupo"].map({"con": "Con la senal", "sin": "Sin la senal"})
        fig = px.bar(s, x="senal", y="tasa", color="grupo", barmode="group", template=TPL,
                     text=s["tasa"].map("{:.0%}".format), color_discrete_map={"Con la senal": ROJO, "Sin la senal": VERDE})
        fig.update_traces(textposition="outside")
        fig.update_layout(height=360, yaxis_tickformat=".0%", margin=dict(t=10, b=10), yaxis_title="tasa de churn", xaxis_title="", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    st.info("La activacion incompleta es la causa raiz dominante. La friccion de soporte (tickets reabiertos, sin resolver, escalados) es el segundo driver y funciona como alerta temprana: aparece antes de la baja.")


# ---------------- Soporte ----------------
with T["Soporte"]:
    with st.expander("Diagnostico de archivos detectados"):
        st.write("CSV encontrados en el repo y como los clasifico la app:")
        st.dataframe(pd.DataFrame(DETALLE, columns=["archivo", "clasificado como"]), hide_index=True, use_container_width=True)
    if not HAY_SOPORTE:
        st.warning("No se detectaron archivos de soporte. Revisa el diagnostico de arriba: cada archivo de soporte debe conservar sus columnas originales (csat_score para chat, sentimiento_usuario para whatsapp, nps_post_llamada para telefono).")
    else:
        st.subheader("Analisis por canal de soporte")
        canal = st.selectbox("Elegi el canal", ["Chat", "WhatsApp", "Telefono"])

        if canal == "Chat" and chat is not None:
            c = chat
            m = st.columns(4)
            m[0].metric("Tickets", f"{len(c):,}")
            m[1].metric("Tasa de resolucion", f"{(c.resuelto=='si').mean():.0%}")
            m[2].metric("Tasa de reapertura", f"{(c.reapertura=='si').mean():.0%}")
            m[3].metric("CSAT promedio", f"{c.csat_score.mean():.2f} / 5")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Tickets por tema")
                g = c.tema_principal.value_counts().reset_index(); g.columns = ["tema", "tickets"]
                fig = px.bar(g, x="tickets", y="tema", orientation="h", template=TPL, color_discrete_sequence=[AZUL])
                fig.update_layout(height=340, margin=dict(t=10, b=10), yaxis_title="", yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.caption("CSAT promedio por tema (mas bajo = mas fricion)")
                g = c.groupby("tema_principal").csat_score.mean().reset_index().sort_values("csat_score")
                fig = px.bar(g, x="csat_score", y="tema_principal", orientation="h", template=TPL, color="csat_score",
                             color_continuous_scale=[CORAL, AMBAR, TEAL], range_x=[1, 5])
                fig.update_layout(height=340, margin=dict(t=10, b=10), yaxis_title="", coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            t1, t2 = st.columns(2)
            t1.metric("Primera respuesta (mediana)", f"{c.tiempo_primera_respuesta_min.median():.0f} min")
            t2.metric("Tiempo de resolucion (mediana)", f"{c.tiempo_resolucion_hs.median():.1f} hs")

        elif canal == "WhatsApp" and wa is not None:
            c = wa
            m = st.columns(4)
            m[0].metric("Tickets", f"{len(c):,}")
            m[1].metric("Resuelto en conversacion", f"{(c.resuelto_en_conversacion=='si').mean():.0%}")
            m[2].metric("Deriva a agente", f"{(c.deriva_a_agente=='si').mean():.0%}")
            m[3].metric("Sentimiento negativo", f"{(c.sentimiento_usuario=='negativo').mean():.0%}")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Sentimiento del usuario")
                g = c.sentimiento_usuario.value_counts().reset_index(); g.columns = ["sentimiento", "tickets"]
                fig = px.pie(g, names="sentimiento", values="tickets", template=TPL, hole=0.45,
                             color="sentimiento", color_discrete_map={"positivo": VERDE, "neutro": GRIS, "negativo": ROJO})
                fig.update_layout(height=330, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.caption("Sentimiento negativo por tema")
                g = c.groupby("tema_principal").apply(lambda x: (x.sentimiento_usuario == "negativo").mean()).reset_index()
                g.columns = ["tema", "neg"]; g = g.sort_values("neg", ascending=True)
                fig = px.bar(g, x="neg", y="tema", orientation="h", template=TPL, text=g["neg"].map("{:.0%}".format), color_discrete_sequence=[ROJO])
                fig.update_layout(height=330, margin=dict(t=10, b=10), yaxis_title="", xaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
            st.metric("Primera respuesta (mediana)", f"{c.tiempo_primera_respuesta_min.median():.0f} min")

        elif canal == "Telefono" and tel is not None:
            c = tel
            m = st.columns(4)
            m[0].metric("Llamadas", f"{len(c):,}")
            m[1].metric("Tasa de resolucion", f"{(c.resuelto=='si').mean():.0%}")
            m[2].metric("Tasa de escalamiento", f"{(c.escalo_a_especialista=='si').mean():.0%}")
            m[3].metric("NPS post llamada", f"{c.nps_post_llamada.mean():.2f} / 5")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Llamadas por tema")
                g = c.tema_principal.value_counts().reset_index(); g.columns = ["tema", "llamadas"]
                fig = px.bar(g, x="llamadas", y="tema", orientation="h", template=TPL, color_discrete_sequence=[AZUL])
                fig.update_layout(height=330, margin=dict(t=10, b=10), yaxis_title="", yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.caption("Motivos de escalamiento")
                g = c.motivo_escala.dropna().value_counts().reset_index(); g.columns = ["motivo", "casos"]
                fig = px.bar(g, x="casos", y="motivo", orientation="h", template=TPL, color_discrete_sequence=[NARANJA])
                fig.update_layout(height=330, margin=dict(t=10, b=10), yaxis_title="", yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            st.metric("Duracion de llamada (mediana)", f"{c.duracion_llamada_min.median():.0f} min")

        st.markdown("---")
        st.markdown("##### Cobertura de soporte sobre la base")
        cov = st.columns(3)
        cov[0].metric("Cuentas con algun ticket", f"{int(d['usa_soporte'].sum()):,}", f"{d['usa_soporte'].mean():.0%}")
        cov[1].metric("Cuentas con fricion", f"{int(d['friccion'].sum()):,}", f"{d['friccion'].mean():.0%}")
        cov[2].metric("Churn con fricion vs sin", f"{d[d['friccion']][col_churn].mean():.0%} vs {d[~d['friccion']][col_churn].mean():.0%}")


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
