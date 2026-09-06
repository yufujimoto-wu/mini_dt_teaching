import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Mini Mobility Digital Twin",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0.6rem;
        max-width: 1500px;
    }
    h1, h2, h3 {
        margin-top: 0.15rem;
        margin-bottom: 0.25rem;
    }
    div[data-testid="stMetric"] {
        padding-top: 0.05rem;
        padding-bottom: 0.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Mini Mobility Digital Twin")

# ============================================================
# 1. Base data
# ============================================================

time = np.arange(24)

CI_PATTERNS = {
    "昼間低CI": np.array([
        0.55, 0.56, 0.56, 0.55, 0.53, 0.50,
        0.46, 0.42, 0.38, 0.33, 0.28, 0.24,
        0.22, 0.23, 0.26, 0.31, 0.38, 0.46,
        0.54, 0.58, 0.60, 0.59, 0.57, 0.56
    ]),
    "夜間低CI": np.array([
        0.28, 0.27, 0.26, 0.26, 0.27, 0.30,
        0.34, 0.39, 0.44, 0.48, 0.50, 0.51,
        0.50, 0.49, 0.48, 0.47, 0.46, 0.44,
        0.41, 0.37, 0.33, 0.30, 0.28, 0.27
    ]),
    "ほぼ一定": np.array([
        0.42, 0.42, 0.42, 0.41, 0.41, 0.42,
        0.42, 0.43, 0.42, 0.41, 0.42, 0.42,
        0.41, 0.42, 0.42, 0.43, 0.42, 0.42,
        0.41, 0.42, 0.42, 0.41, 0.42, 0.42
    ]),
}

# charge_hours = normal-timetable charging opportunities
buses_base = {
    "Bus 1": {
        "distance": 80,
        "diesel_eff": 4.0,
        "diesel_ci": 2.62,
        "ev_eff": 1.10,
        "charge_hours": [0,1,2,3,4,10, 11, 12, 13, 14, 21, 22, 23],
    },
    "Bus 2": {
        "distance": 150,
        "diesel_eff": 3.8,
        "diesel_ci": 2.62,
        "ev_eff": 1.20,
        "charge_hours": [0, 1, 2, 3, 4, 22, 23],
    },
    "Bus 3": {
        "distance": 60,
        "diesel_eff": 4.2,
        "diesel_ci": 2.62,
        "ev_eff": 1.00,
        "charge_hours": [9, 10, 12, 13, 15, 17, 18],
    },
    "Bus 4": {
        "distance": 120,
        "diesel_eff": 3.6,
        "diesel_ci": 2.62,
        "ev_eff": 1.25,
        "charge_hours": [5, 6, 7, 20, 21],
    },
    "Bus 5": {
        "distance": 200,
        "diesel_eff": 3.5,
        "diesel_ci": 2.62,
        "ev_eff": 1.30,
        "charge_hours": [0, 1, 2, 3, 22, 23],
    },
}


def hours_to_intervals(hours):
    if not hours:
        return []

    hours = sorted(set(hours))
    intervals = []
    start = hours[0]
    prev = hours[0]

    for h in hours[1:]:
        if h == prev + 1:
            prev = h
        else:
            intervals.append((start, prev + 1))
            start = h
            prev = h

    intervals.append((start, prev + 1))
    return intervals


def apply_timetable_condition(base_hours, condition):
    """
    Teaching abstraction:
      通常     : original charging opportunities
      タイト   : each opportunity block is shortened
      余裕あり : each opportunity block is extended by one hour where possible
    """
    # 「通常」は指定された1時間コマを一切変形せず、そのまま使用する。
    if condition == "通常":
        return sorted(set(int(h) for h in base_hours if 0 <= int(h) <= 23))

    intervals = hours_to_intervals(base_hours)
    adjusted = []

    for start, end in intervals:
        if condition == "タイト":

            length = end - start

            # 基本は連続区間の両端を1コマずつ削る。
            # ただし、もともと0:00から充電可能な区間では0:00-1:00を残し、
            # もともと24:00まで充電可能な区間では23:00-24:00を残す。
            preserve_start = (start == 0)
            preserve_end = (end == 24)

            if length <= 2:
                if preserve_start and preserve_end:
                    new_start, new_end = start, end
                elif preserve_start:
                    new_start, new_end = start, start + 1
                elif preserve_end:
                    new_start, new_end = end - 1, end
                else:
                    new_start, new_end = start, start + 1
            else:
                new_start = start if preserve_start else start + 1
                new_end = end if preserve_end else end - 1

                if new_end <= new_start:
                    if preserve_start:
                        new_start, new_end = start, min(start + 1, end)
                    elif preserve_end:
                        new_start, new_end = max(start, end - 1), end
                    else:
                        new_start, new_end = start, start + 1

        else:  # 余裕あり
            new_start = max(0, start - 1)
            new_end = min(24, end + 1)

        adjusted.extend(range(new_start, new_end))

    return sorted(set(adjusted))


# ============================================================
# 2. Sidebar
# ============================================================

with st.sidebar:
    st.header("条件設定")

    selected_buses = st.multiselect(
        "電化するバス（最大2台）",
        options=list(buses_base.keys()),
        default=[],
        max_selections=2,
        placeholder="最大2台を選択",
    )

    st.caption("限られた2台だけ電化可能")

    st.divider()

    charger_power = st.slider(
        "共用充電器の出力 [kW]",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
    )

    timetable_condition = st.radio(
        "運行ダイヤの余裕",
        ["通常", "タイト", "余裕あり"],
        horizontal=False,
    )

    ci_pattern_name = st.radio(
        "系統CIパターン",
        ["昼間低CI", "夜間低CI", "ほぼ一定"],
        horizontal=False,
    )

    st.caption("※ EVバスは車庫の共用充電器1台を利用する")

ci = CI_PATTERNS[ci_pattern_name]

buses = {}
for bus_name, p in buses_base.items():
    q = dict(p)
    q["charge_hours"] = apply_timetable_condition(
        p["charge_hours"],
        timetable_condition,
    )
    buses[bus_name] = q

electrified = {
    bus_name: bus_name in selected_buses
    for bus_name in buses.keys()
}

# ============================================================
# 3. Shared-charger operation simulation
# ============================================================

def choose_bus_for_charger_late_first(
    hour,
    candidates,
    remaining_energy,
    buses,
    charger_power,
):
    """
    Priority rule when multiple buses compete for the one shared charger.

    We schedule backward from 23:00 toward 0:00.

    For each candidate bus, calculate how much charging capacity remains
    in *earlier* chargeable slots if the current slot is skipped.

        deficit_if_skip
          = remaining_required_energy
            - charging_capacity_in_earlier_slots

    A positive value means:
      "If this bus is not charged now, it can no longer receive enough
       energy in the remaining earlier slots."

    The bus with the largest deficit_if_skip is therefore prioritized.

    If all buses can still meet their energy requirement after skipping
    the current slot, the bus with the smaller remaining slack is chosen.
    Bus name is used only as a deterministic final tie-breaker.
    """
    scored = []

    for bus_name in candidates:
        earlier_slots = sum(
            1
            for h in buses[bus_name]["charge_hours"]
            if h < hour
        )

        earlier_capacity = earlier_slots * charger_power

        deficit_if_skip = (
            remaining_energy[bus_name]
            - earlier_capacity
        )

        slack_if_skip = (
            earlier_capacity
            - remaining_energy[bus_name]
        )

        scored.append({
            "bus": bus_name,
            "deficit_if_skip": deficit_if_skip,
            "slack_if_skip": slack_if_skip,
        })

    # First: a bus that becomes infeasible if the current slot is skipped.
    urgent = [
        x for x in scored
        if x["deficit_if_skip"] > 1e-9
    ]

    if urgent:
        urgent.sort(
            key=lambda x: (
                -x["deficit_if_skip"],
                x["bus"],
            )
        )
        return urgent[0]["bus"]

    # Otherwise choose the bus with the least slack in the remaining
    # earlier charging opportunities.
    scored.sort(
        key=lambda x: (
            x["slack_if_skip"],
            x["bus"],
        )
    )
    return scored[0]["bus"]


def simulate_shared_charger(buses, selected_buses, charger_power, ci):
    """
    Single authoritative shared-charger simulation.

    Charging policy:
      1. Schedule from the latest hour backward (23 -> 0).
      2. Charge only in each bus's charge_hours.
      3. Only one bus can use the charger in each one-hour slot.
      4. When buses compete, prioritize the bus that would otherwise
         become unable to satisfy its required charging energy.

    The resulting charging_profiles are the single source of truth for
    plots, CO2, charged energy, and feasibility.
    """
    charging_profiles = {
        bus_name: np.zeros(24)
        for bus_name in buses.keys()
    }

    energy_required = {
        bus_name: (
            buses[bus_name]["distance"]
            * buses[bus_name]["ev_eff"]
        )
        for bus_name in selected_buses
    }

    remaining_energy = dict(energy_required)

    conflict_hours = []
    dispatch_log = []

    # Latest possible charging first.
    for h in range(23, -1, -1):

        candidates = [
            bus_name
            for bus_name in selected_buses
            if (
                remaining_energy[bus_name] > 1e-9
                and h in buses[bus_name]["charge_hours"]
            )
        ]

        if len(candidates) > 1:
            conflict_hours.append(h)

        if not candidates:
            continue

        if len(candidates) == 1:
            chosen_bus = candidates[0]
        else:
            chosen_bus = choose_bus_for_charger_late_first(
                hour=h,
                candidates=candidates,
                remaining_energy=remaining_energy,
                buses=buses,
                charger_power=charger_power,
            )

        charged_energy = min(
            charger_power,
            remaining_energy[chosen_bus],
        )

        charging_profiles[chosen_bus][h] = charged_energy
        remaining_energy[chosen_bus] -= charged_energy

        # Record why this vehicle was selected at a conflict hour.
        earlier_slots_after_current = {
            bus_name: sum(
                1
                for hh in buses[bus_name]["charge_hours"]
                if hh < h
            )
            for bus_name in candidates
        }

        deficit_if_skip = {
            bus_name: (
                remaining_energy[bus_name]
                + (
                    charged_energy
                    if bus_name == chosen_bus
                    else 0.0
                )
                - earlier_slots_after_current[bus_name] * charger_power
            )
            for bus_name in candidates
        }

        dispatch_log.append({
            "hour": h,
            "candidates": list(candidates),
            "chosen_bus": chosen_bus,
            "charged_kwh": charged_energy,
            "reason": (
                "この時間を逃すと必要量確保が難しい車両を優先"
                if len(candidates) > 1
                else "候補は1台のみ"
            ),
        })

    total_charging = np.sum(
        np.vstack([
            charging_profiles[name]
            for name in buses.keys()
        ]),
        axis=0,
    )

    electric_co2 = {
        bus_name: float(
            np.sum(charging_profiles[bus_name] * ci)
        )
        for bus_name in selected_buses
    }

    charging_complete = {
        bus_name: (
            remaining_energy[bus_name] <= 1e-6
        )
        for bus_name in selected_buses
    }

    return {
        "charging_profiles": charging_profiles,
        "energy_required": energy_required,
        "remaining_energy": remaining_energy,
        "total_charging": total_charging,
        "electric_co2": electric_co2,
        "charging_complete": charging_complete,
        "conflict_hours": sorted(set(conflict_hours)),
        "dispatch_log": sorted(
            dispatch_log,
            key=lambda x: x["hour"],
        ),
    }


operation = simulate_shared_charger(
    buses=buses,
    selected_buses=selected_buses,
    charger_power=charger_power,
    ci=ci,
)

charging_profiles = operation["charging_profiles"]
total_charging = operation["total_charging"]

# ============================================================
# 4. Results derived from the same operation
# ============================================================

results = []

for bus_name, p in buses.items():
    distance = p["distance"]

    diesel_liter = distance / p["diesel_eff"]
    diesel_co2 = diesel_liter * p["diesel_ci"]

    energy_required = distance * p["ev_eff"]

    if electrified[bus_name]:
        electric_co2 = operation["electric_co2"][bus_name]
        operation_co2 = electric_co2
        technology = "Electric"
        charging_complete = operation["charging_complete"][bus_name]
        charged_energy = float(charging_profiles[bus_name].sum())
        remaining = float(operation["remaining_energy"][bus_name])
    else:
        electric_co2 = np.nan
        operation_co2 = diesel_co2
        technology = "Diesel"
        charging_complete = True
        charged_energy = 0.0
        remaining = 0.0

    results.append({
        "Bus": bus_name,
        "Technology": technology,
        "Distance_km": distance,
        "Diesel_CO2": diesel_co2,
        "Electric_CO2": electric_co2,
        "Operation_CO2": operation_co2,
        "Energy_required": energy_required,
        "Charged_energy": charged_energy,
        "Remaining_energy": remaining,
        "Charging_complete": charging_complete,
        "Chargeable_hours": len(p["charge_hours"]),
    })

results_df = pd.DataFrame(results)

baseline_co2 = float(results_df["Diesel_CO2"].sum())
current_co2 = float(results_df["Operation_CO2"].sum())
co2_reduction = baseline_co2 - current_co2
reduction_ratio = 100 * co2_reduction / baseline_co2
n_electric = len(selected_buses)
total_ev_energy = float(total_charging.sum())

infeasible_buses = [
    bus_name
    for bus_name in selected_buses
    if not operation["charging_complete"][bus_name]
]

# ============================================================
# 5. Fixed two-column dashboard
# ============================================================

col_visual, col_kpi = st.columns([3.5, 1.0], gap="large")

with col_visual:

    # --------------------------------------------------------
    # 5a. Always-visible operation timeline
    # --------------------------------------------------------

    st.markdown("### 各バスの1日の運行・充電可能時間")

    fig_timeline = go.Figure()

    first_unavailable = True
    first_available = True
    first_actual = True

    all_hours = set(range(24))

    for bus_name, p in buses.items():

        # IMPORTANT:
        # Draw one rectangle per one-hour slot.
        # The exact same charge_hours list is used by the operation simulation.
        available_hours = set(int(h) for h in p["charge_hours"])
        unavailable_hours = sorted(all_hours - available_hours)

        # 1-hour unavailable cells
        for h in unavailable_hours:
            fig_timeline.add_trace(
                go.Bar(
                    x=[1],
                    y=[bus_name],
                    base=[h],
                    orientation="h",
                    name="運行・充電不可",
                    legendgroup="unavailable",
                    showlegend=first_unavailable,
                    marker_color="rgba(150,150,150,0.42)",
                    opacity=0.65,
                    hovertemplate=(
                        f"{bus_name}<br>"
                        f"{h}:00–{h+1}:00 運行・充電不可"
                        "<extra></extra>"
                    ),
                )
            )
            first_unavailable = False

        # 1-hour chargeable cells: exact representation of charge_hours
        for h in sorted(available_hours):
            fig_timeline.add_trace(
                go.Bar(
                    x=[1],
                    y=[bus_name],
                    base=[h],
                    orientation="h",
                    name="充電可能",
                    legendgroup="available",
                    showlegend=first_available,
                    marker_color="rgba(90,170,245,0.62)",
                    opacity=0.85,
                    hovertemplate=(
                        f"{bus_name}<br>"
                        f"{h}:00–{h+1}:00 充電可能"
                        "<extra></extra>"
                    ),
                )
            )
            first_available = False

        # Actual charging: also draw the exact 1-hour slot from the authoritative
        # shared-charger simulation result.
        for h in range(24):
            charged = charging_profiles[bus_name][h]
            if charged <= 1e-9:
                continue

            fig_timeline.add_trace(
                go.Bar(
                    x=[1],
                    y=[bus_name],
                    base=[h],
                    orientation="h",
                    name="実際に充電",
                    legendgroup="actual",
                    showlegend=first_actual,
                    marker_color="rgba(245,155,35,0.98)",
                    opacity=0.98,
                    hovertemplate=(
                        f"{bus_name}<br>"
                        f"{h}:00–{h+1}:00 実際に充電 "
                        f"{charged:.1f} kWh"
                        "<extra></extra>"
                    ),
                )
            )
            first_actual = False

    timeline_labels = {
        name: f"{name} | {p['distance']} km/day"
        for name, p in buses.items()
    }

    fig_timeline.update_layout(
        barmode="overlay",
        template="plotly_white",
        height=255,
        xaxis=dict(
            title="時刻",
            range=[0, 24],
            tickvals=list(range(0, 25, 2)),
            ticktext=[f"{h}:00" for h in range(0, 25, 2)],
            fixedrange=True,
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            tickmode="array",
            tickvals=list(buses.keys()),
            ticktext=[timeline_labels[name] for name in buses.keys()],
            fixedrange=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=38, b=30),
    )

    st.plotly_chart(
        fig_timeline,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # --------------------------------------------------------
    # 5b. CI and charging — same width as the timeline
    # --------------------------------------------------------

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.11,
        row_heights=[0.52, 0.48],
        subplot_titles=(
            f"時間別 系統CO₂排出原単位（{ci_pattern_name}）",
            "共用充電器の実際の充電電力"
        )
    )

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

    # Per-bus traces are taken from the exact same charging profiles as the timeline/KPIs.
    for bus_name in selected_buses:
        fig.add_trace(
            go.Bar(
                x=time,
                y=charging_profiles[bus_name],
                name=bus_name,
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
        title_text="kWh / h",
        range=[0, max(10, charger_power * 1.15)],
        row=2,
        col=1
    )

    fig.update_xaxes(
        title_text="時刻",
        tickvals=list(range(0, 24, 2)),
        row=2,
        col=1
    )

    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        height=365,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=58, l=45, r=20, b=28)
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    if infeasible_buses:
        details = []
        for bus_name in infeasible_buses:
            details.append(
                f"{bus_name}（不足 {operation['remaining_energy'][bus_name]:.1f} kWh）"
            )
        st.error(
            "⚠️ 共用充電器1台では充電不足："
            + "、".join(details)
        )
    elif selected_buses:
        st.success("選択したEVバスはすべて走行に必要な充電量を確保可能")
    else:
        st.caption("左の条件設定から電化するバスを最大2台選択")

    if operation["conflict_hours"]:
        conflict_text = "、".join(
            f"{h}:00" for h in operation["conflict_hours"]
        )
        st.caption(
            "共用充電器の競合が発生した時間："
            + conflict_text
            + "。必要量を確保できなくなる可能性が高い車両を優先し、できるだけ遅い時間帯から充電しています。"
        )


with col_kpi:

    st.markdown("### 結果")

    st.metric(
        "電動化台数",
        f"{n_electric} / 2 台"
    )

    st.metric(
        "1日CO₂排出量",
        f"{current_co2:.1f} kg"
    )

    st.metric(
        "CO₂削減量",
        f"{co2_reduction:.1f} kg"
    )

    st.metric(
        "CO₂削減率",
        f"{reduction_ratio:.1f} %"
    )

    st.metric(
        "実充電電力量",
        f"{total_ev_energy:.1f} kWh"
    )

    st.markdown("#### 現在の条件")
    st.caption(f"運行ダイヤ：{timetable_condition}")
    st.caption(f"CI：{ci_pattern_name}")
    st.caption(f"共用充電器：1台・{charger_power} kW")

    if selected_buses:
        st.markdown("#### 電化車両")
        for name in selected_buses:
            p = buses[name]
            charged = charging_profiles[name].sum()
            required = operation["energy_required"][name]
            st.caption(
                f"{name}：{p['distance']} km/day"
                f"・充電 {charged:.0f}/{required:.0f} kWh"
            )
    else:
        st.caption("電化車両：なし")

# ============================================================
# 6. Details
# ============================================================

with st.expander("バスごとの比較値を見る"):

    display_df = results_df.copy()

    for col in [
        "Diesel_CO2",
        "Electric_CO2",
        "Operation_CO2",
        "Energy_required",
        "Charged_energy",
        "Remaining_energy",
    ]:
        display_df[col] = display_df[col].round(1)

    display_df = display_df.rename(
        columns={
            "Bus": "バス",
            "Technology": "方式",
            "Distance_km": "走行距離 [km/day]",
            "Diesel_CO2": "Diesel CO₂ [kg/day]",
            "Electric_CO2": "EV実運用CO₂ [kg/day]",
            "Operation_CO2": "現在のCO₂ [kg/day]",
            "Energy_required": "必要充電量 [kWh/day]",
            "Charged_energy": "実充電量 [kWh/day]",
            "Remaining_energy": "充電不足量 [kWh/day]",
            "Charging_complete": "必要量を充電可能",
            "Chargeable_hours": "充電可能時間 [h/day]",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

with st.expander("共用充電器の配分ルールを見る"):
    st.write(
        "翌日の運行に必要な充電量を、充電可能な時間のうち"
        "できるだけ遅い時間帯から確保します。"
        "同じ時間に複数のEVバスが競合した場合は、"
        "その時間を逃すと残りの充電可能時間だけでは"
        "必要充電量を満たしにくい車両を優先します。"
    )

    if operation["dispatch_log"]:
        log_df = pd.DataFrame(operation["dispatch_log"])
        log_df["時刻"] = log_df["hour"].map(lambda h: f"{h}:00")
        log_df["候補車両"] = log_df["candidates"].map(lambda x: ", ".join(x))
        log_df = log_df.rename(
            columns={
                "chosen_bus": "充電車両",
                "charged_kwh": "充電量 [kWh]",
                "reason": "選択理由",
            }
        )
        st.dataframe(
            log_df[
                ["時刻", "候補車両", "充電車両", "充電量 [kWh]", "選択理由"]
            ],
            use_container_width=True,
            hide_index=True,
        )
