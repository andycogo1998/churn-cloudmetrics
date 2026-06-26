"""
Dashboard de Churn CloudMetrics
Fuentes integradas: usuarios + retiros + uso de producto.
Soporte (chat, whatsapp, telefono) se agrega en la proxima iteracion.
Pensado para uso continuo del equipo de Customer Experience.
"""

import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Churn CloudMetrics", layout="wide", page_icon="📊")

# Paleta y estilo simples para que sea facil de leer
VERDE = "#16a085"
ROJO = "#e74c3c"
GRIS = "#7f8c8d"
AZUL = "#2980b9"
TPL = "plotly_white"

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.6rem;}
    h1, h2, h3 {color: #1f2d3d;}
    </style>
    """,
    unsafe_allow_html=True,
)

CENTROIDES = {
    "Colombia": (4.57, -74.30),
    "México": (23.63, -102.55),
    "Costa Rica": (9.75, -83.75),
    "República Dominicana": (18.74, -70.16),
    "Panamá": (8.54, -80.78),
    "Ecuador": (-1.83, -78.18),
}


def encontrar_csv():
    for c in ["tabla_unificada_churn.csv", "data/tabla_unificada_churn.csv"]:
        if os.path.exists(c):
            return c
    m = glob.glob("**/*unificada*.csv", recursive=True) or glob.glob("**/*.csv", recursive=True)
    return m[0] if m else None


@st.cache_data
def cargar_datos(path):
    df = pd.read_csv(path, dtype=str)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    for c in ["user_id.1", "user_id.2"]:
        if c in df.columns:
            df = df.drop(columns=c)

    num = ["monto_mensual_usd", "nps_salida", "dias_primer_factura",
           "facturas_emitidas_mes1", "facturas_emitidas_mes3",
           "reportes_generados_mes3", "usuarios_adicionales",
           "sesiones_promedio_semana"]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["fecha_registro", "fecha_ultimo_pago", "fecha_retiro"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "fecha_primer_login" in df.columns:
        serial = pd.to_numeric(df["fecha_primer_login"], errors="coerce")
        df["fecha_primer_login"] = pd.to_datetime(serial, unit="D", origin="1899-12-30", errors="coerce")

    df["mrr"] = df["monto_mensual_usd"]
    df["churn_estado"] = df["estado_cuenta"].fillna("").str.lower().ne("activo")
    df["churn_retiro"] = df["tipo_retiro"].notna()
    df["activado"] = df["facturas_emitidas_mes1"].fillna(0) > 0

    # ---- Flags de calidad de datos ----
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


DATA_PATH = encontrar_csv()
if DATA_PATH is None:
    st.error("No se encontro ningun CSV. Subi tabla_unificada_churn.csv a la raiz del repo.")
    st.stop()

df = cargar_datos(DATA_PATH)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title("Filtros")

definicion = st.sidebar.radio(
    "Definicion de churn",
    ["estado_cuenta (recomendada)", "registro de retiro"],
    help="estado_cuenta marca 440 cuentas no activas. Retiros solo 140. La diferencia es ruido de datos documentado.",
)
col_churn = "churn_estado" if definicion.startswith("estado") else "churn_retiro"


def multi(label, col):
    opts = sorted(df[col].dropna().unique().tolist())
    return st.sidebar.multiselect(label, opts, default=opts)


f_estado = multi("Estado de cuenta", "estado_cuenta")
f_seg = multi("Segmento", "segmento")
f_pais = multi("Pais", "pais")
f_plan = multi("Plan", "plan")

d = df[
    df["estado_cuenta"].isin(f_estado)
    & df["segmento"].isin(f_seg)
    & df["pais"].isin(f_pais)
    & df["plan"].isin(f_plan)
]

st.sidebar.markdown("---")
st.sidebar.caption(f"Mostrando {len(d):,} de {len(df):,} cuentas")


# ------------------------------------------------------------------
# Cabecera y KPIs
# ------------------------------------------------------------------
st.title("Dashboard de Churn CloudMetrics")
st.caption("Sistema de control continuo del equipo de Customer Experience. Usuarios, retiros y uso de producto. Soporte en la proxima version.")

total = len(d)
churned = int(d[col_churn].sum())
tasa = churned / total if total else 0
activos = d[d["estado_cuenta"] == "activo"]
mrr_total = d["mrr"].sum()
mrr_activo = activos["mrr"].sum()
mrr_riesgo = d.loc[d[col_churn], "mrr"].sum()
arpu = mrr_activo / len(activos) if len(activos) else 0
tasa_activacion = d["activado"].mean() if total else 0

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Cuentas", f"{total:,}")
k2.metric("Tasa de churn", f"{tasa:.1%}", f"{churned:,} cuentas", delta_color="off")
k3.metric("MRR activo", f"${mrr_activo:,.0f}")
k4.metric("MRR en riesgo", f"${mrr_riesgo:,.0f}", f"{(mrr_riesgo/mrr_total if mrr_total else 0):.0%} del total", delta_color="off")
k5.metric("ARPU (activos)", f"${arpu:,.0f}")
k6.metric("Tasa de activacion", f"{tasa_activacion:.0%}", help="Cuentas que emitieron al menos una factura el mes 1")

st.markdown("---")

tab_res, tab_mapa, tab_neg, tab_causa, tab_cal = st.tabs(
    ["Resumen", "Mapa", "Negocio", "Causa raiz", "Calidad de datos"]
)


# ------------------------------------------------------------------
# TAB Resumen
# ------------------------------------------------------------------
def churn_por(data, col):
    g = data.groupby(col).agg(cuentas=("user_id", "count"), churn=(col_churn, "sum")).reset_index()
    g["tasa"] = g["churn"] / g["cuentas"]
    return g.sort_values("tasa", ascending=False)


with tab_res:
    st.subheader("Donde se concentra el churn")
    st.caption("Tasa de churn por corte. Las barras mas altas son los grupos que mas se van.")
    c1, c2, c3 = st.columns(3)
    for col, titulo, cont in [("segmento", "Por segmento", c1), ("plan", "Por plan", c2), ("pais", "Por pais", c3)]:
        with cont:
            g = churn_por(d, col)
            fig = px.bar(g, x=col, y="tasa", text=g["tasa"].map("{:.0%}".format),
                         template=TPL, color_discrete_sequence=[ROJO])
            fig.update_traces(textposition="outside")
            fig.update_layout(title=titulo, yaxis_tickformat=".0%", height=330,
                              margin=dict(t=40, b=10), showlegend=False, yaxis_title="", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Composicion de la base por estado")
    g = d["estado_cuenta"].value_counts().reset_index()
    g.columns = ["estado", "cuentas"]
    fig = px.bar(g, x="cuentas", y="estado", orientation="h", template=TPL,
                 color="estado",
                 color_discrete_map={"activo": VERDE, "suspendido": "#f39c12",
                                     "inactivo": GRIS, "cancelado": ROJO},
                 text="cuentas")
    fig.update_layout(height=260, showlegend=False, margin=dict(t=10, b=10),
                      yaxis_title="", xaxis_title="cuentas")
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# TAB Mapa
# ------------------------------------------------------------------
with tab_mapa:
    st.subheader("Cuentas y churn por pais")
    st.caption("El tamano del circulo es la cantidad de cuentas. El color es la tasa de churn: mas rojo, mas churn.")
    g = d.groupby("pais").agg(cuentas=("user_id", "count"), churn=(col_churn, "sum"),
                              mrr=("mrr", "sum")).reset_index()
    g["tasa"] = g["churn"] / g["cuentas"]
    g["lat"] = g["pais"].map(lambda p: CENTROIDES.get(p, (None, None))[0])
    g["lon"] = g["pais"].map(lambda p: CENTROIDES.get(p, (None, None))[1])
    g = g.dropna(subset=["lat", "lon"])

    fig = px.scatter_geo(
        g, lat="lat", lon="lon", size="cuentas", color="tasa",
        hover_name="pais",
        hover_data={"cuentas": True, "tasa": ":.1%", "mrr": ":$,.0f", "lat": False, "lon": False},
        color_continuous_scale=["#16a085", "#f1c40f", "#e74c3c"],
        size_max=55, template=TPL,
    )
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="#cccccc",
                    showland=True, landcolor="#f7f7f7", showocean=True, oceancolor="#eaf2f8")
    fig.update_layout(height=520, margin=dict(t=10, b=10), coloraxis_colorbar_title="churn")
    st.plotly_chart(fig, use_container_width=True)

    g2 = g[["pais", "cuentas", "churn", "tasa", "mrr"]].sort_values("tasa", ascending=False).copy()
    g2["tasa"] = g2["tasa"].map("{:.1%}".format)
    g2["mrr"] = g2["mrr"].map("${:,.0f}".format)
    g2.columns = ["Pais", "Cuentas", "Churn", "Tasa churn", "MRR"]
    st.dataframe(g2, hide_index=True, use_container_width=True)


# ------------------------------------------------------------------
# TAB Negocio
# ------------------------------------------------------------------
with tab_neg:
    st.subheader("Metricas de negocio")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("MRR por plan")
        g = d.groupby("plan")["mrr"].sum().reset_index().sort_values("mrr", ascending=True)
        fig = px.bar(g, x="mrr", y="plan", orientation="h", template=TPL,
                     text=g["mrr"].map("${:,.0f}".format), color_discrete_sequence=[AZUL])
        fig.update_layout(height=320, showlegend=False, margin=dict(t=10, b=10),
                          xaxis_title="MRR USD", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.caption("MRR por segmento")
        g = d.groupby("segmento")["mrr"].sum().reset_index()
        fig = px.pie(g, names="segmento", values="mrr", template=TPL, hole=0.5,
                     color_discrete_sequence=[AZUL, VERDE])
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Embudo de activacion")
    st.caption("Cuantas cuentas completan cada hito del onboarding. Donde mas cae el embudo es donde se pierde valor.")
    hitos = [("configuracion_empresa_completa", "Config. empresa"),
             ("empleados_cargados", "Empleados cargados"),
             ("plan_cuentas_configurado", "Plan de cuentas"),
             ("primer_factura_emitida", "Primera factura"),
             ("integracion_banco_conectada", "Banco conectado")]
    filas = []
    for col, lbl in hitos:
        if col in d.columns:
            filas.append({"hito": lbl, "cuentas": d[col].fillna("").str.lower().eq("si").sum()})
    if filas:
        emb = pd.DataFrame(filas)
        fig = px.funnel(emb, x="cuentas", y="hito", template=TPL, color_discrete_sequence=[VERDE])
        fig.update_layout(height=330, margin=dict(t=10, b=10), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    a1, a2, a3 = st.columns(3)
    a1.metric("ARPU activos", f"${arpu:,.0f}")
    a2.metric("Cuentas activadas", f"{int(d['activado'].sum()):,}", f"{tasa_activacion:.0%}")
    banco = d["integracion_banco_conectada"].fillna("").str.lower().eq("si").mean() if total else 0
    a3.metric("Conexion bancaria", f"{banco:.0%}")


# ------------------------------------------------------------------
# TAB Causa raiz
# ------------------------------------------------------------------
with tab_causa:
    st.subheader("Que dispara el churn")
    st.caption("Tasa de churn segun comportamiento en el producto. La brecha entre las barras es el tamano del problema.")

    def churn_binario(col, etiqueta_si, etiqueta_no):
        s = d[col].fillna("").str.lower().eq("si")
        out = []
        for val, lbl in [(True, etiqueta_si), (False, etiqueta_no)]:
            sub = d[s == val]
            if len(sub):
                out.append({"grupo": lbl, "tasa": sub[col_churn].mean(), "cuentas": len(sub)})
        return pd.DataFrame(out)

    pares = [
        ("configuracion_empresa_completa", "Completo el setup", "No completo el setup", "Configuracion inicial"),
        ("integracion_banco_conectada", "Conecto el banco", "No conecto el banco", "Integracion bancaria"),
    ]
    cols = st.columns(2)
    for (col, si, no, titulo), cont in zip(pares, cols):
        with cont:
            g = churn_binario(col, si, no)
            fig = px.bar(g, x="grupo", y="tasa", text=g["tasa"].map("{:.0%}".format),
                         template=TPL, color="grupo",
                         color_discrete_map={si: VERDE, no: ROJO})
            fig.update_traces(textposition="outside")
            fig.update_layout(title=titulo, height=330, showlegend=False,
                              yaxis_tickformat=".0%", margin=dict(t=40, b=10),
                              yaxis_title="", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Activacion vs churn")
    g = d.groupby("activado").agg(cuentas=("user_id", "count"), churn=(col_churn, "sum")).reset_index()
    g["tasa"] = g["churn"] / g["cuentas"]
    g["grupo"] = g["activado"].map({True: "Facturo el mes 1", False: "No facturo el mes 1"})
    fig = px.bar(g, x="grupo", y="tasa", text=g["tasa"].map("{:.0%}".format), template=TPL,
                 color="grupo", color_discrete_map={"Facturo el mes 1": VERDE, "No facturo el mes 1": ROJO})
    fig.update_traces(textposition="outside")
    fig.update_layout(height=330, showlegend=False, yaxis_tickformat=".0%",
                      margin=dict(t=10, b=10), yaxis_title="", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Las cuentas que no completan la configuracion inicial ni emiten su primera factura churnean varias veces mas. La activacion es la causa raiz dominante, por encima de segmento, pais o plan.")


# ------------------------------------------------------------------
# TAB Calidad de datos
# ------------------------------------------------------------------
with tab_cal:
    st.subheader("Panel de calidad de datos")
    st.caption("Inconsistencias detectadas en las fuentes. No se borran filas: se marcan y se decide su uso. Esto sostiene la confianza en las metricas de arriba.")

    flags = [
        ("flag_estado_sin_retiro", "Cuenta no activa sin registro de retiro",
         "estado_cuenta como verdad de churn, retiros como causa"),
        ("flag_activacion_contradice", "Marca no facturo pero tiene facturas mes 1",
         "Activacion derivada de facturas_mes1 > 0, no del flag"),
        ("flag_fecha_invertida", "Ultimo pago anterior al registro",
         "Excluir de antiguedad y recencia"),
        ("flag_retiro_fecha_invertida", "Fecha de retiro anterior al ultimo pago",
         "No usar estas fechas para recencia"),
        ("flag_login_no_activo", "Login reciente en cuenta no activa",
         "No usar login_30 como actividad en churneados"),
        ("flag_pago_geo", "Metodo de pago geograficamente imposible",
         "No usar metodo_pago como atributo por pais"),
    ]
    filas = []
    for col, desc, accion in flags:
        if col in d.columns:
            n = int(d[col].sum())
            filas.append({"Hallazgo": desc, "Cuentas afectadas": n,
                          "% del filtro": f"{(n/len(d) if len(d) else 0):.1%}", "Decision": accion})
    reg = pd.DataFrame(filas).sort_values("Cuentas afectadas", ascending=False)

    cc1, cc2 = st.columns([1, 1])
    with cc1:
        fig = px.bar(reg, x="Cuentas afectadas", y="Hallazgo", orientation="h",
                     template=TPL, text="Cuentas afectadas", color_discrete_sequence=["#f39c12"])
        fig.update_layout(height=360, margin=dict(t=10, b=10), yaxis_title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        afectadas = d[[c for c, _, _ in flags if c in d.columns]].any(axis=1).sum()
        st.metric("Cuentas con al menos una inconsistencia", f"{int(afectadas):,}",
                  f"{(afectadas/len(d) if len(d) else 0):.0%} del filtro actual")
        st.metric("MRR no explicado por retiros",
                  f"${(d.loc[d['flag_estado_sin_retiro'],'mrr'].sum()):,.0f}",
                  help="Churn que existe por estado pero no tiene ficha de salida")

    st.dataframe(reg, hide_index=True, use_container_width=True)
