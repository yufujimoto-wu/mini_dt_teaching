import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title = "Mini Energy Digital Twin",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Mini Energy Digital Twin")

# -----------------------------
# Parameters
# -----------------------------

# -----------------------------
# Parameters
# -----------------------------

DEFAULTS = {
    "pv_capacity": 3.0,
    "load_scale": 1.0,
    "ci": 0.45,
    "battery_capacity": 0.0
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
    
def reset_parameters():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value

with st.sidebar:
    st.header("条件設定")

    st.button(
        "初期状態に戻す",
        on_click=reset_parameters,
        use_container_width=True
    )
    
    pv_capacity = st.slider(
        "PV容量 [kW]",
        min_value=0.0,
        max_value=10.0,
        step=0.5,
        key="pv_capacity"
    )

    load_scale = st.slider(
        "需要倍率",
        min_value=0.5,
        max_value=2.0,
        step=0.1,
        key="load_scale"
    )

    ci = st.slider(
        "系統CO₂排出原単位 [kg/kWh]",
        min_value=0.1,
        max_value=0.8,
        step=0.05,
        key="ci"
    )

    battery_capacity = st.slider(
        "蓄電池容量 [kWh]",
        min_value=0.0,
        max_value=20.0,
        step=1.0,
        key="battery_capacity"
    )
    
# -----------------------------
# Dummy data
# -----------------------------

time = np.arange(24)

# 家庭需要（適当）
load = (
    2
    + 1.2*np.sin((time-7)/24*2*np.pi)**2
    + 0.8*np.sin((time-18)/24*2*np.pi)**2
)

load *= load_scale

# PV
pv_shape = np.maximum(
    0,
    np.sin((time-6)/12*np.pi)
)

pv = pv_capacity * pv_shape

soc = np.zeros(24)

battery = 0   # 初期SOC=0%

grid = np.zeros(24)
charge = np.zeros(24)
discharge = np.zeros(24)

for t in range(24):

    surplus = pv[t] - load[t]

    # ----------------------
    # 余剰PV → 充電
    # ----------------------
    if surplus > 0:

        charge_amount = min(
            surplus,
            battery_capacity - battery
        )

        battery += charge_amount

        charge[t] = charge_amount

        grid[t] = 0

    # ----------------------
    # PV不足 → 放電
    # ----------------------
    else:

        deficit = -surplus

        discharge_amount = min(
            deficit,
            battery
        )

        battery -= discharge_amount

        discharge[t] = discharge_amount

        grid[t] = deficit - discharge_amount

    soc[t] = battery

co2 = np.sum(grid * ci)

self_consumption = np.minimum(load, pv).sum()

self_ratio = 100 * self_consumption / max(np.sum(pv), 1e-6)

pv_supply_ratio = (
    100 * self_consumption /
    max(np.sum(load), 1e-6)
)

demand_energy = np.sum(load)
pv_generation = np.sum(pv)

# 蓄電池を含めて最終的にPV由来で利用できた量
pv_used = demand_energy - np.sum(grid)

pv_surplus = max(pv_generation - pv_used, 0)

pv_supply_ratio = (
    100 * pv_used / max(demand_energy, 1e-6)
)

pv_self_consumption_ratio = (
    100 * pv_used / max(pv_generation, 1e-6)
)

battery_use = discharge.sum()

# 設備情報
pv_cost = pv_capacity * 20      # 万円
battery_cost = battery_capacity * 8

total_cost = pv_cost + battery_cost

# -----------------------------
# Plotly
# -----------------------------
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.7, 0.3],
    subplot_titles=(
        "電力フロー",
        "蓄電池SOC"
    )
)

# -----------------------------
# 上段：電力
# -----------------------------
fig.add_trace(
    go.Scatter(
        x=time,
        y=load,
        mode="lines",
        name="需要",
        line=dict(width=3)
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=time,
        y=pv,
        mode="lines",
        name="PV",
        line=dict(width=3)
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=time,
        y=grid,
        mode="lines",
        name="系統購入",
        line=dict(width=3)
    ),
    row=1,
    col=1
)

# -----------------------------
# 下段：SOC
# -----------------------------
fig.add_trace(
    go.Scatter(
        x=time,
        y=soc,
        mode="lines+markers",
        name="SOC",
        line=dict(width=3)
    ),
    row=2,
    col=1
)

fig.update_yaxes(
    title_text="電力 [kW]",
    row=1,
    col=1
)

fig.update_yaxes(
    range=[0, battery_capacity],
    title_text="SOC [kWh]",
    row=2,
    col=1
)

fig.update_xaxes(
    title_text="時刻",
    row=2,
    col=1
)

fig.update_layout(
    template="plotly_white",
    height=450,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.08,
        xanchor="center",
        x=0.5
    ),
    margin=dict(t=80)
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# Indicators
# -----------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("購入電力量", f"{grid.sum():.1f} kWh")
c2.metric("CO₂排出量", f"{co2:.1f} kg")
c3.metric("PV自給率", f"{pv_supply_ratio:.1f} %")
c4.metric("PV自家消費率", f"{self_ratio:.1f} %")

with st.expander("電力量の内訳詳細を見る"):
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("需要電力量", f"{demand_energy:.1f} kWh")
    d2.metric("PV発電量", f"{pv_generation:.1f} kWh")
    d3.metric("PV利用量", f"{pv_used:.1f} kWh")
    d4.metric("PV余剰量", f"{pv_surplus:.1f} kWh")
    d5.metric("蓄電池放電量", f"{discharge.sum():.1f} kWh")

with st.expander("設備情報を見る"):
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("PV容量", f"{pv_capacity:.1f} kW")
    e2.metric("蓄電池容量", f"{battery_capacity:.1f} kWh")
    e3.metric("PV設備費", f"{pv_cost:.1f} 万円")
    e4.metric("蓄電池設備費", f"{battery_cost:.1f} 万円")
    e5.metric("合計設備費", f"{total_cost:.1f} 万円")


    
