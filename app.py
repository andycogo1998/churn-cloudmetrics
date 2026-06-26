"""
Dashboard de Churn CloudMetrics (v1)
Fuentes integradas hasta ahora: usuarios + retiros + uso de producto.
Soporte (chat, whatsapp, telefono) se agrega en la proxima iteracion.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Churn CloudMetrics", layout="wide")

import glob
import os


def encontrar_csv():
    """Busca el CSV unificado sin depender del nombre exacto."""
    candidatos = [
        "tabla_unificada_churn.csv",
        "data/tabla_unificada_churn.csv",
    ]
    for c in candidatos:
        if os.path.exists(c):
            return c
    # buscar cualquier csv con 'unificada' en el nombre, en raiz o subcarpetas
    matches = glob.glob("**/*unificada*.csv", recursive=True)
    if matches:
        return matches[0]
    # ultimo recurso: el primer csv que aparezca en el repo
    matches = glob.glob("**/*.csv", recursive=True)
    if matches:
        return matches[0]
    return None


# ------------------------------------------------------------------
# Carga y limpieza
# ------------------------------------------------------------------
@st.cache_data
def cargar_datos(path):
    df = pd.read_csv(path, dtype=str)

    # quitar columnas user_id duplicadas que deja el join horizontal
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    for c in ["user_id.1", "user_id.2"]:
        if c in df.columns:
            df = df.drop(columns=c)

    # numericos
    num_cols = [
        "monto_mensual_usd", "nps_salida", "dias_primer_factura",
        "facturas_emitidas_mes1", "facturas_emitidas_mes3",
        "reportes_generados_mes3", "usuarios_adicionales",
        "sesiones_promedio_semana",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # fechas ISO
    for c in ["fecha_registro", "fecha_ultimo_pago", "fecha_retiro"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # fecha_primer_login viene como serial de Excel, lo convertimos
    if "fecha_primer_login" in df.columns:
        serial = pd.to_numeric(df["fecha_primer_login"], errors="coerce")
        df["fecha_primer_login"] = pd.to_datetime(
            serial, unit="D", origin="1899-12-30", errors="coerce"
        )

    # banderas de churn segun las dos definiciones
    df["churn_estado"] = df["estado_cuenta"].fillna("").str.lower().ne("activo")
    df["churn_retiro"] = df["tipo_retiro"].notna()

    return df


DATA_PATH = encontrar_csv()
if DATA_PATH is None:
    st.error(
        "No se encontro ningun CSV en el repositorio. "
        "Subi el archivo a la raiz del repo (idealmente como "
        "tabla_unificada_churn.csv) y volve a desplegar."
    )
    st.stop()

df = cargar_datos(DATA_PATH)


# ------------------------------------------------------------------
# Sidebar: filtros y definicion de churn
# ------------------------------------------------------------------
st.sidebar.title("Controles")

definicion = st.sidebar.radio(
    "Definicion de churn",
    ["estado_cuenta (operativa)", "registro de retiro"],
    help="estado_cuenta marca 440 cuentas. retiros solo 140. No coinciden.",
)
col_churn = "churn_estado" if definicion.startswith("estado") else "churn_retiro"

st.sidebar.markdown("---")

def multi(label, col):
    opts = sorted(df[col].dropna().unique().tolist())
    sel = st.sidebar.multiselect(label, opts, default=opts)
    return sel

f_seg = multi("Segmento", "segmento")
f_pais = multi("Pais", "pais")
f_plan = multi("Plan", "plan")

d = df[
    df["segmento"].isin(f_seg)
    & df["pais"].isin(f_pais)
    & df["plan"].isin(f_plan)
]


# ------------------------------------------------------------------
# KPIs
# ------------------------------------------------------------------
st.title("Dashboard de Churn CloudMetrics")
st.caption("v1 con usuarios, retiros y uso de producto. Soporte pendiente de integrar.")

total = len(d)
churned = int(d[col_churn].sum())
tasa = churned / total if total else 0
mrr_total = d["monto_mensual_usd"].sum()
mrr_riesgo = d.loc[d[col_churn], "monto_mensual_usd"].sum()
nps_salida = d["nps_salida"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Cuentas", f"{total:,}")
k2.metric("Churn", f"{churned:,}", f"{tasa:.1%}")
k3.metric("MRR total", f"${mrr_total:,.0f}")
k4.metric("MRR en riesgo", f"${mrr_riesgo:,.0f}", f"{(mrr_riesgo/mrr_total if mrr_total else 0):.1%}")
k5.metric("NPS de salida", f"{nps_salida:.1f}" if pd.notna(nps_salida) else "s/d")

st.markdown("---")


# ------------------------------------------------------------------
# Churn por corte
# ------------------------------------------------------------------
c1, c2, c3 = st.columns(3)

def churn_por(col):
    g = d.groupby(col).agg(
        cuentas=("user_id", "count"),
        churn=(col_churn, "sum"),
    ).reset_index()
    g["tasa"] = g["churn"] / g["cuentas"]
    return g

with c1:
    st.subheader("Por segmento")
    g = churn_por("segmento")
    fig = px.bar(g, x="segmento", y="tasa", text=g["tasa"].map("{:.1%}".format))
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False, height=320)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Por plan")
    g = churn_por("plan")
    fig = px.bar(g, x="plan", y="tasa", text=g["tasa"].map("{:.1%}".format))
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False, height=320)
    st.plotly_chart(fig, use_container_width=True)

with c3:
    st.subheader("Por pais")
    g = churn_por("pais").sort_values("tasa", ascending=False)
    fig = px.bar(g, x="pais", y="tasa", text=g["tasa"].map("{:.1%}".format))
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False, height=320)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ------------------------------------------------------------------
# Motivos de retiro y NPS
# ------------------------------------------------------------------
c4, c5 = st.columns(2)

with c4:
    st.subheader("Motivos principales de retiro")
    ret = d[d["churn_retiro"]]
    if len(ret):
        g = ret["motivo_principal"].value_counts().reset_index()
        g.columns = ["motivo", "cuentas"]
        fig = px.bar(g, x="cuentas", y="motivo", orientation="h")
        fig.update_layout(height=360, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay registros de retiro con los filtros actuales.")

with c5:
    st.subheader("Activacion vs churn")
    st.caption("Hitos de configuracion inicial completados segun estado.")
    hitos = [
        "configuracion_empresa_completa", "empleados_cargados",
        "plan_cuentas_configurado", "primer_factura_emitida",
        "integracion_banco_conectada",
    ]
    filas = []
    for h in hitos:
        if h in d.columns:
            for estado, sub in d.groupby(col_churn):
                tasa_h = sub[h].fillna("").str.lower().eq("si").mean()
                filas.append({
                    "hito": h.replace("_", " "),
                    "grupo": "churn" if estado else "activo",
                    "tasa": tasa_h,
                })
    if filas:
        gh = pd.DataFrame(filas)
        fig = px.bar(gh, x="tasa", y="hito", color="grupo", orientation="h", barmode="group")
        fig.update_layout(height=360, xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


st.markdown("---")


# ------------------------------------------------------------------
# Panel de calidad de datos
# ------------------------------------------------------------------
with st.expander("Calidad de datos y supuestos"):
    st.markdown(
        """
        **Inconsistencia clave.** estado_cuenta marca 440 cuentas no activas,
        pero retiros solo tiene 140 registros. De las 129 canceladas, solo 38
        tienen registro de retiro. Por eso el dashboard permite elegir la definicion.

        **Formato.** fecha_primer_login venia como serial de Excel y se normalizo a fecha.

        **Pendiente.** Integrar soporte (chat, whatsapp, telefono) para correlacionar
        CSAT, reaperturas y sentimiento con el churn.
        """
    )
    nulos = d.isna().mean().sort_values(ascending=False)
    nulos = (nulos[nulos > 0] * 100).round(1)
    st.write("Porcentaje de nulos por columna:")
    st.dataframe(nulos.rename("% nulos"))
