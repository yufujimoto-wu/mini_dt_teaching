import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

st.set_page_config(
    page_title="Mini Mobility Digital Twin",
    layout="wide"
)

st.title("Mini Mobility Digital Twin")
#st.caption(
#    "限られた台数しか電化できないとしたら、"
#    "どのバスから電化すべきでしょうか？"
#)

# ============================================================
# 1. Dummy data
# ============================================================

time = np.arange(24)

# 時間別系統CO2排出原単位 [kg-CO2/kWh]
# 昼間は再エネ等で低く、夜間・夕方は高いという仮想例
ci = np.array([
    0.55, 0.56, 0.56, 0.55, 0.53, 0.50,
    0.46, 0.42, 0.38, 0.33, 0.28, 0.24,
    0.22, 0.23, 0.26, 0.31, 0.38, 0.46,
    0.54, 0.58, 0.60, 0.59, 0.57, 0.56
])

# ------------------------------------------------------------
# 5台の仮想バス
#
# distance       : 1日走行距離 [km/day]
# diesel_eff     : ディーゼル燃費 [km/L]
# diesel_ci      : 軽油排出係数 [kg-CO2/L]
# ev_eff         : EV電費 [kWh/km]
# charge_hours   : 充電可能時刻
#
# 充電可能時間の違いが電化効果に影響するように設定
# ------------------------------------------------------------

buses = {
    "Bus 1": {
        "distance": 80,
        "diesel_eff": 4.0,
        "diesel_ci": 2.62,
        "ev_eff": 1.10,
        "charge_hours": [10, 11, 12, 13, 14, 22, 23]
    },

    "Bus 2": {
        "distance": 150,
        "diesel_eff": 3.8,
        "diesel_ci": 2.62,
        "ev_eff": 1.20,
        "charge_hours": [0, 1, 2, 3, 4, 22, 23]
    },

    "Bus 3": {
        "distance": 60,
        "diesel_eff": 4.2,
        "diesel_ci": 2.62,
        "ev_eff": 1.00,
        "charge_hours": [9, 10, 11, 12, 13, 14, 15]
    },

    "Bus 4": {
        "distance": 120,
        "diesel_eff": 3.6,
        "diesel_ci": 2.62,
        "ev_eff": 1.25,
        "charge_hours": [5, 6, 7, 20, 21, 22, 23]
    },

    "Bus 5": {
        "distance": 200,
        "diesel_eff": 3.5,
        "diesel_ci": 2.62,
        "ev_eff": 1.30,
        "charge_hours": [0, 1, 2, 3, 4]
    }
}

# ============================================================
# 2. Sidebar
# ============================================================

with st.sidebar:

    st.header("条件設定")

    st.markdown("### 電化するバス")

    electrified = {}

    for bus_name in buses.keys():

        electrified[bus_name] = st.checkbox(
            bus_name,
            value=False
        )

    st.divider()

    charger_power = st.slider(
        "充電器出力 [kW]",
        min_value=10,
        max_value=100,
        value=50,
        step=10
    )

    st.caption(
        "※ 各バスが独立した充電器を"
        "利用できるものとする"
    )

# ============================================================
# 3. Simulation
# ============================================================

def hours_to_intervals(hours):
    """
    [10,11,12,13,14,22,23]
    ->
    [(10,15), (22,24)]
    """
    if len(hours) == 0:
        return []

    hours = sorted(hours)

    intervals = []

    start = hours[0]
    prev = hours[0]

    for h in hours[1:]:

        if h == prev + 1:
            prev = h

        else:
            intervals.append(
                (start, prev + 1)
            )

            start = h
            prev = h

    intervals.append(
        (start, prev + 1)
    )

    return intervals

results = []

total_charging = np.zeros(24)
charging_profiles = {}

for bus_name, p in buses.items():

    distance = p["distance"]

    # --------------------------------------------------------
    # Diesel
    # --------------------------------------------------------

    diesel_liter = distance / p["diesel_eff"]

    diesel_co2 = (
        diesel_liter
        * p["diesel_ci"]
    )

    # --------------------------------------------------------
    # Electric bus
    # --------------------------------------------------------

    energy_required = (
        distance
        * p["ev_eff"]
    )

    charging = np.zeros(24)

    remaining = energy_required

    # CO2の低い充電可能時間から優先的に充電
    #
    # 初版では非常に単純なルールベース
    # --------------------------------------------------------

    available_hours = sorted(
        p["charge_hours"],
        key=lambda h: ci[h]
    )

    for h in available_hours:

        if remaining <= 0:
            break

        charged = min(
            charger_power,
            remaining
        )

        charging[h] += charged

        remaining -= charged

    charging_complete = (
        remaining <= 1e-6
    )

    electric_co2 = np.sum(
        charging * ci
    )

    # --------------------------------------------------------
    # Selected technology
    # --------------------------------------------------------

    if electrified[bus_name]:

        operation_co2 = electric_co2

        total_charging += charging

        technology = "Electric"

    else:

        operation_co2 = diesel_co2

        technology = "Diesel"

    results.append({
        "Bus": bus_name,
        "Technology": technology,
        "Distance_km": distance,
        "Diesel_CO2": diesel_co2,
        "Electric_CO2": electric_co2,
        "Operation_CO2": operation_co2,
        "Energy_required": energy_required,
        "Charging_complete": charging_complete
    })

    charging_profiles[bus_name] = charging.copy()

results_df = pd.DataFrame(results)

# ============================================================
# 4. Baseline
# ============================================================

baseline_co2 = results_df[
    "Diesel_CO2"
].sum()

current_co2 = results_df[
    "Operation_CO2"
].sum()

co2_reduction = (
    baseline_co2
    - current_co2
)

reduction_ratio = (
    100
    * co2_reduction
    / baseline_co2
)

n_electric = sum(
    electrified.values()
)

total_ev_energy = results_df.loc[
    results_df["Technology"] == "Electric",
    "Energy_required"
].sum()

# ============================================================
# 5. Plot
# ============================================================

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,
    row_heights=[0.45, 0.55],
    subplot_titles=(
        "時間別 系統CO₂排出原単位",
        "EVバス充電電力"
    )
)

# ------------------------------------------------------------
# CI
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=time,
        y=ci,
        mode="lines+markers",
        name="系統CO₂排出原単位"
    ),
    row=1,
    col=1
)

# ------------------------------------------------------------
# Charging demand
# ------------------------------------------------------------

fig.add_trace(
    go.Bar(
        x=time,
        y=total_charging,
        name="EV充電"
    ),
    row=2,
    col=1
)

fig.update_yaxes(
    title_text="kg-CO₂/kWh",
    row=1,
    col=1
)

fig.update_yaxes(
    title_text="充電電力 [kWh]",
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
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    margin=dict(t=80)
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# 6. Main indicators
# ============================================================

selected_buses = [
    bus_name
    for bus_name in buses.keys()
    if electrified[bus_name]
]

infeasible_buses = [
    r["Bus"]
    for r in results
    if (
        electrified[r["Bus"]]
        and not r["Charging_complete"]
    )
]

if len(infeasible_buses) == 0:
    st.success(
        "✅ 選択したすべてのEVバスで走行に必要な充電量を確保できています"
    )
else:
    st.error(
        "⚠️ 充電不足："
        + ", ".join(infeasible_buses)
    )

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "電動化台数",
    f"{n_electric} / 5 台"
)

c2.metric(
    "1日CO₂排出量",
    f"{current_co2:.1f} kg"
)

c3.metric(
    "CO₂削減率",
    f"{reduction_ratio:.1f} %"
)

c4.metric(
    "EV充電電力量",
    f"{total_ev_energy:.1f} kWh"
)

# ============================================================
# 7. Details
# ============================================================

with st.expander(
    "バスごとの詳細を見る"
):

    display_df = results_df.copy()

    display_df["Diesel_CO2"] = (
        display_df["Diesel_CO2"]
        .round(1)
    )

    display_df["Electric_CO2"] = (
        display_df["Electric_CO2"]
        .round(1)
    )

    display_df["Operation_CO2"] = (
        display_df["Operation_CO2"]
        .round(1)
    )

    display_df["Energy_required"] = (
        display_df["Energy_required"]
        .round(1)
    )

    display_df = display_df.rename(
        columns={
            "Bus": "バス",
            "Technology": "方式",
            "Distance_km": "走行距離 [km]",
            "Diesel_CO2": "Diesel CO₂ [kg]",
            "Electric_CO2": "EV CO₂ [kg]",
            "Operation_CO2": "現在のCO₂ [kg]",
            "Energy_required": "必要充電量 [kWh]",
            "Charging_complete": "充電可能"
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# 8. Charging opportunities
# ============================================================

with st.expander(
    "充電可能時間と実際の充電を見る"
):

    fig_timeline = go.Figure()
    
    # -----------------------------
    # 充電可能時間
    # -----------------------------
    available_color = "rgba(100,180,255,0.35)"
    charging_color = "rgba(255,170,0,1.0)"
    
    first_available = True
    
    for bus_name, p in buses.items():

        intervals = hours_to_intervals(
            p["charge_hours"]
        )

        for start, end in intervals:

            fig_timeline.add_trace(
                go.Bar(
                    x=[end - start],
                    y=[bus_name],
                    base=[start],
                    orientation="h",
                    name="充電可能",
                    legendgroup="available",
                    marker_color=available_color,
                    showlegend=first_available,
                    opacity=0.45
                )
            )

            first_available = False

    # -----------------------------
    # 実際の充電時間
    # -----------------------------

    first_actual = True

    for bus_name in buses.keys():

        # 電化していないバスは「実際に充電」を描かない
        if not electrified[bus_name]:
            continue

        actual_hours = [
            h
            for h in range(24)
            if charging_profiles[bus_name][h] > 1e-6
        ]

        intervals = hours_to_intervals(
            actual_hours
        )

        for start, end in intervals:

            fig_timeline.add_trace(
                go.Bar(
                    x=[end - start],
                    y=[bus_name],
                    base=[start],
                    orientation="h",
                    name="実際に充電",
                    legendgroup="actual",
                    showlegend=first_actual,
                    opacity=0.95,
                    marker_color="orange"
                )
            )
            
            first_actual = False

        
    fig_timeline.update_layout(
        barmode="overlay",
        template="plotly_white",
        height=220,
        xaxis=dict(
            title="時刻",
            range=[0, 24],
            tickvals=list(range(0, 25, 2)),
            ticktext=[
                f"{h}:00"
                for h in range(0, 25, 2)
            ]
        ),
        yaxis=dict(
            title="",
            autorange="reversed"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )

    st.plotly_chart(
        fig_timeline,
        use_container_width=True
    )


    
# ============================================================
# 9. Mission
# ============================================================

#st.divider()
#st.subheader("考えてみよう")
#st.write(
#    """
#    **5台のうち2台だけを電化できるとします。  
#    どの2台を選べばCO₂排出量を最も削減できるでしょうか？**
#
#    走行距離だけで決めてよいでしょうか。  
#    充電可能時間と系統CO₂排出原単位にも注目してみてください。
#    """
#)
