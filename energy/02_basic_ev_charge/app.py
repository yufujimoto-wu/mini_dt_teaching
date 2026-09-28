import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Mini EV Energy DT", layout="wide")

st.markdown(
    """
    <style>
    /* Keep the teaching dashboard visible without unnecessary scrolling. */
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 0.7rem;
        max-width: 1450px;
    }
    div[data-testid="stRadio"] {
        margin-top: -0.35rem;
        margin-bottom: -0.55rem;
    }
    div[data-testid="stRadio"] > label {
        display: none;
    }
    div[data-testid="stMetric"] {
        padding-top: 0.15rem;
        padding-bottom: 0.15rem;
    }
    h1, h2, h3 {
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
.small-note {color:#666; font-size:0.9rem;}
.kpi-note {color:#777; font-size:0.82rem; margin-top:-0.6rem; margin-bottom:0.8rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Mini EV Energy Digital Twin")
#st.caption("いつ充電する？ → 予測で比べる → 実際はどうなった？ → 必要なら計画を更新する")

# ============================================================
# Teaching assumptions
# ============================================================
EV_CAPACITY = 40.0          # kWh
CHARGER_POWER = 3.0         # kW (1-hour time step)
CHARGE_EFF = 0.95
TARGET_SOC = 80.0           # %
INITIAL_SOC = 30.0          # %

DAY_TYPES = {
    "☀️ 休日": {
        "description": "当日0:00時点でSOC 30%。8–10時は買い物、17–20時は外食。それ以外は自宅にEVがあります。",
        "start_hour": 0,
        "planned_departure_abs": 32,  # 翌8:00
        "availability_windows": [(0, 8), (10, 17), (20, 32)],
        "summary": ["08:00–10:00 買い物", "17:00–20:00 外食", "翌08:00 次の出発"],
        # SOC consumption while away (%-points per hour)
        "away_soc_per_hour": 4.0,
        "required_soc": 80.0,
    },
    "🏢 平日": {
        "description": "当日0:00時点でSOC 30%。朝8時に出発し、18時に帰宅。翌朝8時まで充電できます。",
        "start_hour": 0,
        "planned_departure_abs": 32,  # 翌8:00
        "availability_windows": [(0, 8), (18, 32)],
        "summary": ["08:00-18:00 仕事", "翌08:00 出発"],
        # SOC consumption while away (%-points per hour)
        "away_soc_per_hour": 3,
        "required_soc": 80.0,
    },
}

STRATEGIES = {
    "⚡ 利用するたびにすぐ充電": "必要SOC未満なら、利用可能な最初の1時間枠から定格で充電します。",
    "🌙 夜間充電": "23:00–翌7:00の1時間枠を優先し、選んだ枠では定格で充電します。",
    "🕒 自分で設定": "充電したい1時間枠を自由に選びます。",
}


def hour_label(h):
    if h < 24:
        return f"当日 {h:02d}:00"
    return f"翌日 {h - 24:02d}:00"


def availability_mask(hours, windows):
    mask = np.zeros(len(hours), dtype=bool)
    for start, end in windows:
        mask |= (hours >= start) & (hours < end)
    return mask

def total_driving_soc_loss(day_type):
    """Total SOC percentage-point loss while the EV is away from home."""
    cfg = DAY_TYPES[day_type]
    available = set()
    for start, end in cfg["availability_windows"]:
        available.update(range(start, end))

    away_hours = [
        h for h in range(0, cfg["planned_departure_abs"])
        if h not in available
    ]
    return len(away_hours) * cfg["away_soc_per_hour"]

def required_plug_energy_kwh(soc0=INITIAL_SOC, target_soc=TARGET_SOC, extra_soc_loss=0.0):
    """Grid/PV energy needed to end at target SOC after fixed driving consumption."""
    delta_soc = max(0.0, target_soc + extra_soc_loss - soc0)
    battery_energy = EV_CAPACITY * delta_soc / 100.0
    return battery_energy / CHARGE_EFF

def generate_profiles(day_type):
    """Create one deterministic teaching day (forecast + actual)."""
    cfg = DAY_TYPES[day_type]
    hours = np.arange(0, 33)
    hod = hours % 24

    # Household demand
    morning = 1.1 * np.exp(-0.5 * ((hod - 7.0) / 1.7) ** 2)
    evening = 2.1 * np.exp(-0.5 * ((hod - 20.0) / 2.0) ** 2)
    load_fc = 0.75 + morning + evening

    # PV: clear-day profile, peaking around noon.
    daylight = np.maximum(0.0, np.sin((hod - 6.0) / 12.0 * np.pi))
    pv_fc = 5.2 * daylight

    # Grid carbon intensity (deterministic in this teaching example).
    # Uncertainty is intentionally limited to PV generation only.
    if day_type == "🏢 平日":
        # Weekday example:
        # - relatively low CI from 00:00 to around 05:00
        # - higher through daytime/evening
        # - even lower after the next midnight, reflecting stronger wind output
        ci_fc = (
            0.42
            - 0.12 * np.exp(-0.5 * ((hours - 3.0) / 2.3) ** 2)
            + 0.055 * np.exp(-0.5 * ((hours - 19.0) / 2.5) ** 2)
            - 0.18 * np.exp(-0.5 * ((hours - 27.0) / 2.2) ** 2)
        )
    else:
        # Holiday example: a smoother daily CI profile.
        ci_fc = (
            0.42
            - 0.10 * np.exp(-0.5 * ((hod - 13.5) / 3.2) ** 2)
            + 0.06 * np.exp(-0.5 * ((hod - 19.5) / 2.2) ** 2)
        )

    # Electricity price: TEPCO Energy Partner "Yoru-Toku 8"-type time-of-use tariff.
    # Teaching simplification: energy charge only; fuel-cost adjustment etc. are omitted.
    # 07:00–23:00 = 42.60 JPY/kWh, 23:00–07:00 = 31.64 JPY/kWh.
    price_fc = np.where((hod >= 7) & (hod < 23), 42.60, 31.64)

    # "Actual" conditions used in the second half of the exercise.
    # Forecast: mostly sunny all day.
    # Reality: it was cloudy from 09:00 to 12:00, then cleared up in the afternoon.
    # Keep demand / CI close to forecast so students can focus on the PV forecast error.
    rng = np.random.default_rng(2026)
    load_act = load_fc * (1 + rng.normal(0, 0.04, len(hours)))  # small demand forecast error
    ci_act = ci_fc.copy()  # CI is treated as known/deterministic in this demo
    price_act = price_fc.copy()  # electricity price is known and fixed in this demo

    # 07:00 updated PV forecast:
    # the morning cloud becomes visible in the revised weather forecast,
    # but the forecast is still not identical to the eventual actual output.
    pv_upd = pv_fc.copy()
    cloudy_morning = (hod >= 9) & (hod < 12)
    pv_upd[cloudy_morning] = pv_fc[cloudy_morning] * 0.45

    # Actual PV: morning clouds are somewhat stronger than the 07:00 forecast;
    # after noon it clears and output returns close to the original forecast.
    pv_act = pv_fc.copy()
    pv_act[cloudy_morning] = pv_fc[cloudy_morning] * 0.25

    afternoon = (hod >= 12) & (hod < 18)
    pv_act[afternoon] = pv_fc[afternoon] * (
        1 + rng.normal(0, 0.015, afternoon.sum())
    )

    actual_windows = list(cfg["availability_windows"])
    actual_departure_abs = cfg["planned_departure_abs"]

    df = pd.DataFrame(
        {
            "hour": hours,
            "load_fc": np.clip(load_fc, 0.2, None),
            "load_act": np.clip(load_act, 0.2, None),
            "pv_fc": np.clip(pv_fc, 0.0, None),
            "pv_upd": np.clip(pv_upd, 0.0, None),
            "pv_act": np.clip(pv_act, 0.0, None),
            "ci_fc": np.clip(ci_fc, 0.15, 0.8),
            "ci_act": np.clip(ci_act, 0.15, 0.8),
            "price_fc": np.clip(price_fc, 10.0, None),
            "price_act": np.clip(price_act, 10.0, None),
        }
    )
    df["load_upd"] = df["load_fc"]  # demand forecast is not updated in this simple example
    df["ci_upd"] = df["ci_fc"]  # CI forecast does not change
    df["price_upd"] = df["price_fc"]

    df["available_plan"] = availability_mask(hours, cfg["availability_windows"])
    df["available_actual"] = availability_mask(hours, actual_windows)
    return df, actual_departure_abs, actual_windows

def allocate_energy_by_order(df, ordered_indices, energy_kwh):
    """Allocate required plug energy over ordered one-hour slots."""
    schedule = np.zeros(len(df), dtype=float)
    remaining = energy_kwh
    for idx in ordered_indices:
        if remaining <= 1e-9:
            break
        p = min(CHARGER_POWER, remaining)  # one-hour slot => kW numerically equals kWh
        schedule[idx] = p
        remaining -= p
    return schedule


def strategy_schedule(
    df,
    strategy,
    day_type,
    soc0=INITIAL_SOC,
    availability_col="available_plan",
    energy_kwh=None,
    selected_hours=None,
):
    """
    Build the requested charging profile in 1-hour discrete slots.

    Common rule for all strategies:
      - If a slot is selected for charging, request CHARGER_POWER for the full hour.
      - No within-slot modulation is used to hit the target SOC exactly.
      - Physical feasibility (EV availability, SOC <= 100%, etc.) is enforced
        later by execute_operation().
    """
    cfg = DAY_TYPES[day_type]
    available = df[availability_col].to_numpy(dtype=bool)
    requested = np.zeros(len(df), dtype=float)

    if strategy == "⚡ 利用するたびにすぐ充電":
        # Sequential baseline:
        # whenever the EV is home and SOC is below the required level at the
        # beginning of the slot, select that whole one-hour slot for charging.
        soc = float(soc0)
        required_soc = cfg["required_soc"]

        for i, row in df.iterrows():
            hour = int(row["hour"])
            if hour >= cfg["planned_departure_abs"]:
                break

            if available[i]:
                if soc < required_soc - 1e-9:
                    requested[i] = CHARGER_POWER
                    soc += CHARGER_POWER * CHARGE_EFF / EV_CAPACITY * 100.0
                    soc = min(100.0, soc)
            else:
                soc -= cfg["away_soc_per_hour"]
                soc = max(0.0, soc)

        return requested

    if strategy == "🌙 夜間充電":
        # Select whole hourly slots, prioritizing the cheaper 23:00–07:00 window.
        # Keep selecting full-power slots until the simulated next-morning SOC
        # reaches or exceeds the required SOC.
        soc = float(soc0)
        required_soc = cfg["required_soc"]

        # Candidate slots: night first, then other available slots.
        available_idx = df.index[available].tolist()
        night = [
            i for i in available_idx
            if (df.loc[i, "hour"] % 24 >= 23) or (df.loc[i, "hour"] % 24 < 7)
        ]
        day = [i for i in available_idx if i not in night]
        ordered_candidates = (
            sorted(night, key=lambda i: df.loc[i, "hour"])
            + sorted(day, key=lambda i: df.loc[i, "hour"])
        )

        # Greedily add full-hour slots and evaluate the resulting SOC trajectory
        # using the same physical model as the display.
        for i in ordered_candidates:
            requested[i] = CHARGER_POWER
            _, soc_series_tmp, op_tmp = execute_operation(
                df,
                requested,
                day_type,
                availability_col=availability_col,
                soc0=soc0,
            )
            if op_tmp["driving_feasible"] and soc_series_tmp[-1] >= required_soc - 1e-9:
                break

        return requested

    # Student-selected hourly slots.
    if selected_hours:
        selected_set = set(int(h) for h in selected_hours)
        for i, row in df.iterrows():
            hour = int(row["hour"])
            if hour in selected_set and hour < cfg["planned_departure_abs"]:
                requested[i] = CHARGER_POWER

    return requested

def apply_actual_availability(df, schedule):
    applied = schedule.copy()
    applied[~df["available_actual"].to_numpy()] = 0.0
    return applied


def execute_operation(
    df,
    requested_schedule,
    day_type,
    availability_col="available_plan",
    soc0=INITIAL_SOC,
):
    """
    Execute the requested schedule hour by hour under physical constraints.

    Single source of truth for:
      - executed charging power
      - SOC trajectory
      - driving feasibility

    Rules:
      1. While the EV is away, SOC decreases by away_soc_per_hour.
      2. Charging is possible only while the EV is available.
      3. Charging power cannot exceed CHARGER_POWER.
      4. SOC cannot exceed 100%.
      5. If SOC reaches 0 during an away period, the operation is infeasible.
    """
    cfg = DAY_TYPES[day_type]
    available = df[availability_col].to_numpy(dtype=bool)
    requested = np.asarray(requested_schedule, dtype=float)

    executed = np.zeros(len(df), dtype=float)
    soc_series = np.zeros(len(df), dtype=float)
    soc = float(soc0)

    driving_feasible = True
    depletion_hour = None

    for i, row in df.iterrows():
        hour = int(row["hour"])

        if hour < cfg["planned_departure_abs"]:
            if not available[i]:
                soc -= cfg["away_soc_per_hour"]

                if soc <= 0.0:
                    soc = 0.0
                    if driving_feasible:
                        driving_feasible = False
                        depletion_hour = hour

            else:
                requested_kw = max(0.0, min(float(requested[i]), CHARGER_POWER))

                stored_headroom_kwh = EV_CAPACITY * (100.0 - soc) / 100.0
                plug_headroom_kwh = stored_headroom_kwh / CHARGE_EFF
                feasible_kw = min(requested_kw, plug_headroom_kwh)

                executed[i] = feasible_kw
                soc += feasible_kw * CHARGE_EFF / EV_CAPACITY * 100.0
                soc = min(100.0, soc)

        soc_series[i] = soc

    operation_info = {
        "driving_feasible": driving_feasible,
        "depletion_hour": depletion_hour,
    }
    return executed, soc_series, operation_info

def evaluate_executed_operation(df, executed_schedule, soc_series, mode="forecast"):
    """
    Evaluate CO2, charging cost, PV use, and final SOC from the same executed
    charging profile shown in the graph.
    """
    if mode == "forecast":
        pv = df["pv_fc"].to_numpy()
        load = df["load_fc"].to_numpy()
        ci = df["ci_fc"].to_numpy()
        price = df["price_fc"].to_numpy()
    elif mode == "updated":
        pv = df["pv_upd"].to_numpy()
        load = df["load_upd"].to_numpy()
        ci = df["ci_upd"].to_numpy()
        price = df["price_upd"].to_numpy()
    else:
        pv = df["pv_act"].to_numpy()
        load = df["load_act"].to_numpy()
        ci = df["ci_act"].to_numpy()
        price = df["price_act"].to_numpy()

    charge = np.asarray(executed_schedule, dtype=float)
    local_surplus = np.maximum(pv - load, 0.0)
    pv_to_ev = np.minimum(charge, local_surplus)
    grid_to_ev = np.maximum(charge - pv_to_ev, 0.0)

    return {
        "charge_kwh": float(charge.sum()),
        "pv_to_ev_kwh": float(pv_to_ev.sum()),
        "grid_to_ev_kwh": float(grid_to_ev.sum()),
        "co2_kg": float(np.sum(grid_to_ev * ci)),
        "cost_yen": float(np.sum(grid_to_ev * price)),
        "final_soc": float(soc_series[-1]) if len(soc_series) else INITIAL_SOC,
    }


def infeasible_selected_hours(df, selected_hours, availability_col="available_plan"):
    """Return selected hourly slots that violate EV availability."""
    if not selected_hours:
        return []
    unavailable = []
    for h in selected_hours:
        rows = df.loc[df["hour"] == h]
        if rows.empty or not bool(rows.iloc[0][availability_col]):
            unavailable.append(h)
    return unavailable


def make_main_figure(df, metric, schedule, soc_series, day_type, show_actual=False, updated_df=None, update_hour=None, plan_label="EV充電計画"):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.70, 0.30],
        subplot_titles=(metric, "EV充電計画・SOC"),
        specs=[[{}], [{"secondary_y": True}]],
    )

    x = df["hour"]

    if metric == "エネルギー":
        fig.add_trace(go.Scatter(x=x, y=df["pv_fc"], name="PV予測", mode="lines", line=dict(width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df["load_fc"], name="需要予測", mode="lines", line=dict(width=3)), row=1, col=1)
        if stage == "実際には…":
            fig.add_trace(go.Scatter(x=x, y=df["pv_act"], name="PV実績（午前は曇り）", mode="lines", line=dict(width=2, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=df["load_act"], name="需要実績", mode="lines", line=dict(width=2, dash="dot")), row=1, col=1)
        if updated_df is not None:
            fig.add_trace(go.Scatter(x=x, y=updated_df["pv_upd"], name="朝7時更新PV予測", mode="lines", line=dict(width=3, dash="dash")), row=1, col=1)
        ylabel = "kW"

    elif metric == "CO₂原単位":
        fig.add_trace(go.Scatter(x=x, y=df["ci_fc"], name="CI予測", mode="lines", line=dict(width=3)), row=1, col=1)
        ylabel = "kg-CO₂/kWh"

    else:
        fig.add_trace(go.Scatter(x=x, y=df["price_act"], name="電気料金（固定）", mode="lines", line=dict(width=2, dash="dot")), row=1, col=1)
#        fig.add_trace(go.Scatter(x=x, y=df["price_fc"], name="電気料金（固定）", mode="lines", line=dict(width=3)), row=1, col=1)
#        if stage == "実際には…":
#        if updated_df is not None:
#            fig.add_trace(go.Scatter(x=x, y=updated_df["price_upd"], name="電気料金（固定）", mode="lines", line=dict(width=3, dash="dash")), row=1, col=1)
        ylabel = "円/kWh"

    fig.add_trace(go.Bar(x=x, y=schedule, name=plan_label, opacity=0.72), row=2, col=1)

    available = df["available_plan"].to_numpy()
    in_block = False
    block_start = None
    for i, is_available in enumerate(available):
        if is_available and not in_block:
            block_start = df.loc[i, "hour"]
            in_block = True
        if in_block and (not is_available or i == len(available) - 1):
            block_end = df.loc[i, "hour"] if not is_available else df.loc[i, "hour"] + 1
            fig.add_vrect(
                x0=block_start - 0.5,
                x1=block_end - 0.5,
                fillcolor="lightgray",
                opacity=0.12,
                line_width=0,
                row=2,
                col=1,
            )
            in_block = False

    if update_hour is not None:
        fig.add_vline(
            x=update_hour,
            line_width=2,
            line_dash="dash",
            annotation_text="情報更新",
            annotation_position="top",
        )

    ticks = list(range(0, 32, 2))
    fig.update_xaxes(
        tickmode="array",
        tickvals=ticks,
        ticktext=[f"{h % 24:02d}:00" for h in ticks],
        title_text="時刻",
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text=ylabel, row=1, col=1)
    fig.update_yaxes(
        title_text="充電電力 (kW)",
        range=[0, CHARGER_POWER * 1.25],
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="SOC (%)",
        range=[0, 100],
        row=2,
        col=1,
        secondary_y=True,
        showgrid=False,
    )
    # EV SOC on the right-hand axis of the lower panel.
    fig.add_trace(
        go.Scatter(
            x=df["hour"],
            y=soc_series,
            mode="lines+markers",
            name="EV SOC",
            line=dict(width=2),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_width=1,
        annotation_text="SOC 0%（電欠）",
        annotation_position="bottom right",
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.add_hline(
        y=DAY_TYPES[day_type]["required_soc"],
        line_dash="dash",
        line_width=1,
        annotation_text="翌朝必要SOC",
        annotation_position="top right",
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_white",
        height=570,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        margin=dict(t=65, b=25, l=50, r=55),
        barmode="overlay",
    )
    return fig


def metric_delta(actual, forecast, digits=2):
    delta = actual - forecast
    if digits == 0:
        return f"{delta:+.0f}"
    return f"{delta:+.{digits}f}"


def feasibility_label(result, operation, required_soc):
    if not operation["driving_feasible"]:
        return "未充足：走行中に電欠"
    if result["final_soc"] + 1e-9 < required_soc:
        return "未充足：翌朝SOC不足"
    return "充足"


def comparison_table(left, right, left_label, right_label,
                     left_operation, right_operation, required_soc, difference_label):
    specs = [
        ("CO₂排出量［kg］", "co2_kg", 2),
        ("EV充電コスト［円］", "cost_yen", 0),
        ("PVからの充電量［kWh］", "pv_to_ev_kwh", 1),
        ("総充電量［kWh］", "charge_kwh", 1),
        ("翌朝の出発時SOC［%］", "final_soc", 1),
    ]
    rows = []
    for label, key, digits in specs:
        delta = right[key] - left[key]
        if abs(delta) < 0.5 * 10 ** (-digits):
            delta = 0.0
        difference = f"{delta:+.{digits}f}"
        if key == "final_soc":
            difference += " pt"
        rows.append({
            "評価指標": label,
            left_label: f"{left[key]:.{digits}f}",
            right_label: f"{right[key]:.{digits}f}",
            difference_label: difference,
        })
    rows.append({
        "評価指標": "制約の充足",
        left_label: feasibility_label(left, left_operation, required_soc),
        right_label: feasibility_label(right, right_operation, required_soc),
        difference_label: "—",
    })
    return pd.DataFrame(rows)


def make_actual_comparison_figure(df, metric, original_schedule, original_soc,
                                  revised_schedule, revised_soc, day_type):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.09, row_heights=[0.36, 0.32, 0.32],
        subplot_titles=("共通の実績条件", "元の計画を継続した場合", "朝7時に計画を更新した場合"),
        specs=[[{}], [{"secondary_y": True}], [{"secondary_y": True}]],
    )
    if metric == "エネルギー":
        for column, label, color in [("pv_act", "PV実績", "#e7a21b"),
                                     ("load_act", "家庭需要実績", "#546e7a")]:
            fig.add_trace(go.Scatter(x=df["hour"], y=df[column], name=label,
                                    line=dict(color=color, width=2.5)), row=1, col=1)
        unit = "kW"
    else:
        column, unit = ("ci_act", "kg-CO₂/kWh") if metric == "CO₂原単位" else ("price_act", "円/kWh")
        fig.add_trace(go.Scatter(x=df["hour"], y=df[column], name=metric,
                                line=dict(color="#546e7a", width=2.5)), row=1, col=1)
    fig.update_yaxes(title_text=unit, row=1, col=1)
    for row, schedule, soc, color, label in [
        (2, original_schedule, original_soc, "#3277b3", "元の計画"),
        (3, revised_schedule, revised_soc, "#269460", "更新後の計画"),
    ]:
        # Bars represent [h, h+1); SOC values are the end of each interval.
        active = df["hour"].to_numpy() < DAY_TYPES[day_type]["planned_departure_abs"]
        hours = df.loc[active, "hour"].to_numpy()
        fig.add_trace(go.Bar(x=hours + 0.5, y=np.asarray(schedule)[active], width=0.85,
                             marker_color=color, name=f"{label}：充電電力"), row=row, col=1)
        fig.add_trace(go.Scatter(x=np.r_[0, hours + 1], y=np.r_[INITIAL_SOC, np.asarray(soc)[active]],
                                 name=f"{label}：SOC", line=dict(color=color, width=2)),
                      row=row, col=1, secondary_y=True)
        fig.update_yaxes(title_text="充電電力［kW］", range=[0, CHARGER_POWER * 1.25],
                         row=row, col=1, secondary_y=False)
        fig.update_yaxes(title_text="SOC［%］", range=[0, 100], showgrid=False,
                         row=row, col=1, secondary_y=True)
        fig.add_hline(y=DAY_TYPES[day_type]["required_soc"], line_dash="dot", line_width=1,
                      row=row, col=1, secondary_y=True)
        fig.add_vrect(x0=0, x1=7, fillcolor="#adb5bd", opacity=0.12,
                      line_width=0, row=row, col=1)
    fig.add_vline(x=7, line_dash="dash", line_color="#777", line_width=1)
    ticks = list(range(0, 33, 4))
    labels = [f"{'翌' if h >= 24 else ''}{h % 24:02d}:00" for h in ticks]
    fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=labels, range=[0, 32])
    fig.update_xaxes(title_text="時刻（灰色部分：朝7時以前の固定済みの実行結果）", row=3, col=1)
    fig.update_layout(template="plotly_white", height=680,
                      legend=dict(orientation="h", yanchor="bottom", y=1.08),
                      margin=dict(t=95, b=40, l=50, r=50))
    return fig


# ============================================================
# 1. Conditions + charging plan
# ============================================================
with st.sidebar:
    st.header("条件設定")

    day_type = st.radio(
        "生活パターン",
        list(DAY_TYPES.keys()),
        horizontal=True,
    )
    cfg = DAY_TYPES[day_type]

    st.markdown("#### 今日の予定")
    for item in cfg["summary"]:
        st.write(f"・{item}")

    st.markdown("#### EV")
    st.write(f"当日0:00時点のSOC：**{INITIAL_SOC:.0f}%**")
    st.write(f"翌朝の走行に必要なSOC：**{cfg['required_soc']:.0f}%**")
#    if day_type == "☀️ 休日":
#        st.caption(f"外出中のSOC消費：1時間あたり {cfg['away_soc_per_hour']:.0f}ポイント")
#    else:
#        st.caption(f"外出中のSOC消費：1時間あたり {cfg['away_soc_per_hour']:.0f}ポイント")
#    st.caption(cfg["description"])

    st.markdown("#### 予測情報について")
    st.caption("CO₂原単位と電気料金は既知とする")

    st.markdown("#### 電気料金")
#    st.caption("夜トク8相当：7:00–23:00 42.60円/kWh、23:00–翌7:00 31.64円/kWh（燃料費調整等は省略）")
    st.caption("東電EP 夜トク8相当")

# Compact control band
control_title, control_strategy = st.columns([1.0, 5.0], vertical_alignment="center")
with control_title:
    st.markdown("**充電方法**")
with control_strategy:
    strategy = st.radio(
        "充電方法",
        list(STRATEGIES.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )

selected_hours = []
if strategy == "🕒 自分で設定":
    with st.sidebar:
        st.markdown("#### 充電する時間を自分で選択")
        #st.caption(
        #    "充電したい1時間枠を選んでください。"
        #    "EVが外出している時間も選択できますが、その時間の充電は実行できません。"
        #)
        selectable_hours = list(range(0, cfg["planned_departure_abs"]))
        selected_hours = st.pills(
            "充電する時間",
            options=selectable_hours,
            selection_mode="multi",
            format_func=lambda h: hour_label(h),
            key="selected_charge_hours",
            label_visibility="collapsed",
        )

metric = st.radio(
    "グラフで見る情報",
    ["エネルギー", "CO₂原単位", "電気料金"],
    horizontal=True,
    label_visibility="collapsed",
)

# ============================================================
# 2. Forecast-based plan -> what actually happened
# ============================================================
stage = st.radio(
    "表示",
    ["前日の予測で計画", "朝7時に予測更新", "実際には…"],
    horizontal=True,
    label_visibility="collapsed",
)

# Forecasts and actual values for the same teaching day.
df, actual_departure_abs, actual_windows = generate_profiles(day_type)

# --- Initial plan made on the previous day ---
plan_request = strategy_schedule(
    df,
    strategy,
    day_type,
    selected_hours=selected_hours,
)
plan_schedule, plan_soc, plan_operation = execute_operation(
    df,
    plan_request,
    day_type,
    availability_col="available_plan",
)
forecast_result = evaluate_executed_operation(
    df,
    plan_schedule,
    plan_soc,
    mode="forecast",
)

# --- 07:00 update and possible replanning ---
UPDATE_HOUR = 7
# Keep the saved plan separate from the conditional widget: Streamlit may
# remove widget state when the update screen is no longer displayed.
replan_context = (day_type, strategy, tuple(sorted(selected_hours)))
if st.session_state.get("replan_context") != replan_context:
    st.session_state["replan_context"] = replan_context
    st.session_state["saved_updated_charge_hours"] = [
        h for h in selected_hours if h >= UPDATE_HOUR
    ]
    st.session_state.pop("updated_selected_charge_hours", None)


def save_updated_charge_hours():
    st.session_state["saved_updated_charge_hours"] = list(
        st.session_state["updated_selected_charge_hours"]
    )


if stage == "朝7時に予測更新":
    st.caption("朝7時の更新予報：**9:00–12:00は曇りそう**")

    if strategy == "🕒 自分で設定":
        with st.sidebar:
            st.markdown("#### 朝7時：計画を見直す")
            st.caption(
                "0:00–7:00の実行結果は固定し、7:00以降の充電時間を見直します。"
                "元の計画や生活パターン・充電方法を変えた場合は、見直し案を初期化します。"
                "外出中の時間を選んでも、その時間の充電は実行されません。"
            )
            # Restore the editor after its widget state was cleaned up while
            # viewing another stage. The saved plan survives that cleanup.
            if "updated_selected_charge_hours" not in st.session_state:
                st.session_state["updated_selected_charge_hours"] = list(
                    st.session_state["saved_updated_charge_hours"]
                )
            st.pills(
                "更新後に充電する時間",
                options=list(range(UPDATE_HOUR, cfg["planned_departure_abs"])),
                selection_mode="multi",
                format_func=lambda h: hour_label(h),
                key="updated_selected_charge_hours",
                on_change=save_updated_charge_hours,
                label_visibility="collapsed",
            )

updated_selected_hours = [h for h in selected_hours if h < UPDATE_HOUR] + [
    h for h in st.session_state["saved_updated_charge_hours"]
    if UPDATE_HOUR <= h < cfg["planned_departure_abs"]
]
updated_request = strategy_schedule(
    df,
    strategy,
    day_type,
    selected_hours=updated_selected_hours,
)
# Replay the identical executed prefix before simulating the revised future.
# The deterministic physical model therefore carries the exact 07:00 SOC
# (and any prior driving infeasibility) into the revised trajectory.
past_mask = df["hour"].to_numpy() < UPDATE_HOUR
updated_request[past_mask] = plan_schedule[past_mask]
updated_schedule, updated_soc, updated_operation = execute_operation(
    df,
    updated_request,
    day_type,
    availability_col="available_plan",
)
updated_result = evaluate_executed_operation(
    df,
    updated_schedule,
    updated_soc,
    mode="updated",
)

# --- Actual outcome ---
# The morning-updated plan is executed under actual EV availability.
actual_schedule, actual_soc, actual_operation = execute_operation(
    df,
    updated_request,
    day_type,
    availability_col="available_actual",
)
actual_result = evaluate_executed_operation(
    df,
    actual_schedule,
    actual_soc,
    mode="actual",
)

# Counterfactual baseline: the original plan under exactly the same actual conditions.
original_actual_schedule, original_actual_soc, original_actual_operation = execute_operation(
    df, plan_request, day_type, availability_col="available_actual",
)
original_actual_result = evaluate_executed_operation(
    df, original_actual_schedule, original_actual_soc, mode="actual",
)

if stage == "実際には…":
    st.caption(
        "実際の天気：**9:00–12:00は曇り，12:00以降は晴れ**．"
        "同じ実績条件で、元の計画の継続と朝7時の更新計画を比較．"
    )

if stage == "前日の予測で計画":
    display_schedule = plan_schedule
    display_result = forecast_result
    display_soc = plan_soc
    display_operation = plan_operation
    displayed_selected_hours = selected_hours
    plan_label = "前日の予測に基づく充電計画"
elif stage == "朝7時に予測更新":
    display_schedule = updated_schedule
    display_result = updated_result
    display_soc = updated_soc
    display_operation = updated_operation
    displayed_selected_hours = updated_selected_hours
    plan_label = "朝7時の更新予測に基づく充電計画"
else:
    display_schedule = actual_schedule
    display_result = actual_result
    display_soc = actual_soc
    display_operation = actual_operation
    displayed_selected_hours = updated_selected_hours
    plan_label = "朝7時に更新した充電計画"

display_infeasible_hours = (
    infeasible_selected_hours(df, displayed_selected_hours)
    if strategy == "🕒 自分で設定"
    else []
)

# ============================================================
# Forecast planning view / same-condition actual comparison
# ============================================================
final_soc = float(display_soc[-1]) if len(display_soc) > 0 else INITIAL_SOC
required_soc = float(DAY_TYPES[day_type]["required_soc"])

if stage == "実際には…":
    st.markdown("#### 同じ実績条件での計画比較")
    st.caption("天候・家庭需要・電力料金・CO₂排出原単位・EV利用予定は両案で共通。変更するのは朝7時以降の充電計画。")
    st.plotly_chart(
        make_actual_comparison_figure(df, metric, original_actual_schedule, original_actual_soc,
                                      actual_schedule, actual_soc, day_type),
        use_container_width=True,
    )
    if np.allclose(original_actual_schedule, actual_schedule):
        st.info("両案の実行される充電時間・電力は同じです。計画変更による評価の差はありません。")
    st.dataframe(
        comparison_table(original_actual_result, actual_result,
                         "元の計画を継続", "更新後の計画",
                         original_actual_operation, actual_operation, required_soc,
                         "差（更新後−元）"),
        hide_index=True, use_container_width=True,
    )
    st.caption(f"CO₂・料金はEV充電に伴う系統購入分を集計。制約：走行中の電欠なし・翌朝SOC {required_soc:.0f}%以上。総充電量・翌朝SOCも合わせて比較。")
    for label, result, operation in [
        ("元の計画", original_actual_result, original_actual_operation),
        ("更新後の計画", actual_result, actual_operation),
    ]:
        status = feasibility_label(result, operation, required_soc)
        if status != "充足":
            st.warning(f"{label}：{status}。数値が小さくても実行可能な案としては採用できません。")

    with st.expander("予測時の評価と実績条件での評価の違い", expanded=False):
        st.caption("ここでは各計画を固定し、評価に用いる予測と実績の違いを比較。上の表の『計画変更の効果』とは別の比較。")
        st.markdown("##### 元の計画：前日の予測と実績")
        st.dataframe(
            comparison_table(forecast_result, original_actual_result,
                             "前日の予測での評価", "実績条件での評価",
                             plan_operation, original_actual_operation, required_soc,
                             "差（実績−予測）"),
            hide_index=True, use_container_width=True,
        )
        st.markdown("##### 更新後の計画：朝7時の予測と実績")
        st.dataframe(
            comparison_table(updated_result, actual_result,
                             "朝7時の予測での評価", "実績条件での評価",
                             updated_operation, actual_operation, required_soc,
                             "差（実績−予測）"),
            hide_index=True, use_container_width=True,
        )
else:
    col_graph, col_kpi = st.columns([3.2, 1.0], gap="large")
    with col_graph:
        st.plotly_chart(
            make_main_figure(
                df=df, metric=metric, schedule=display_schedule, soc_series=display_soc,
                day_type=day_type,
                updated_df=df if stage == "朝7時に予測更新" else None,
                update_hour=7 if stage == "朝7時に予測更新" else None,
                plan_label=plan_label,
            ),
            use_container_width=True,
        )
    with col_kpi:
        st.markdown("#### 期待される結果")
        st.metric("CO₂排出量", f"{display_result['co2_kg']:.2f} kg")
        st.metric("EV充電コスト", f"{display_result['cost_yen']:.0f} 円")
        st.metric("PVからの充電", f"{display_result['pv_to_ev_kwh']:.1f} kWh")
        st.metric("翌朝の出発時SOC", f"{final_soc:.0f}%", delta=f"必要SOC {required_soc:.0f}%")
        st.caption("いずれも、現時点の予測に基づく評価です。")
    if not display_operation["driving_feasible"]:
        st.error("走行中に電欠が発生するため、この計画は実行不可能です。")
    elif final_soc + 1e-9 < required_soc:
        st.warning("走行は可能ですが、翌朝の出発時に必要SOCへ届きません。")
    else:
        st.success("走行中のSOCと、翌朝の出発時に必要なSOCの両方を満たしています。")

if display_infeasible_hours:
    st.warning(
        "選択した時間のうち実行できない時間："
        + "、".join(hour_label(h) for h in display_infeasible_hours)
        + "（EVが外出中）"
    )

# ============================================================
# Teaching prompts
# ============================================================
#st.divider()
#st.subheader("考えてみよう")

#if stage != "実際には…":
#    st.markdown(
#        """
#- 3つの充電方法を切り替えると、**CO₂・料金・PV利用量**はどう変わりますか？
#- 充電する時間を変えると、CO₂と電気料金はそれぞれどう変わるでしょうか？
#- 「一番よい充電方法」を一つだけ決めることはできるでしょうか？
#"""
#    )
#else:
#    st.markdown(
#        """
#- 天気予報・予測結果を見て決めた計画は、実際の結果ではどう評価が変わりましたか？
#- PV発電量の予測が外れると、**PV利用量・CO₂排出量・電気料金**のどれが変わるでしょうか？
#- 未来が完全には分からないなら、計画を立てた後に何を観測し、どう更新するとよいでしょうか？
#"""
#    )
#with st.expander("補足：休日と平日でPV利用が変わるのはなぜ？"):
#    st.write(
#        "休日は10時以降にEVが自宅にあるため、PVがよく発電する昼間に充電できます。"
#        "一方、平日は昼間にEVが外出しているため、PVが発電していてもEV充電には使えません。"
#        "同じ設備・同じPVでも、**人の行動やEVの在宅時間によって利用できるエネルギーが変わる**ことがポイントです。"
#    )

