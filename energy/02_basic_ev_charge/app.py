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
        padding-top: 3.5rem;
        padding-bottom: 0.7rem;
        max-width: 1450px;
    }
    [data-testid="stVerticalBlock"] { gap: 0.2rem; }
    h1 { font-size: 1.35rem !important; padding: 0 !important; line-height: 1.5 !important; }
    .st-key-titlebar { min-height: 38px; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .st-key-slotbar [data-testid="stHorizontalBlock"] { gap: 2px !important; flex-wrap: nowrap !important; }
    .st-key-slotbar [data-testid="stColumn"] { min-width: 0 !important; flex: 1 1 0 !important; }
    .st-key-slotbar button { min-width: 0 !important; border-radius: 3px; }
    .st-key-slotbar [data-testid="stColumn"]:nth-child(25) { border-left: 2px solid #555; padding-left: 3px; }
    [data-testid="stButton"] button { padding: 0.35rem 0.25rem; }
    div[class*="st-key-slot_"] button { min-height: 1.65rem; height: 1.65rem; padding: 0 0.1rem; }
    div[class*="st-key-slot_"] button p { font-size: 0.78rem; }
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
    padding-top: 3.5rem;
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

with st.container(key="titlebar"):
    st.title("Mini EV Energy Digital Twin")
#st.caption("いつ充電する？ → 予測で比べる → 実際はどうなった？ → 必要なら計画を更新する")

# ============================================================
# Teaching assumptions
# ============================================================
EV_CAPACITY = 40.0          # kWh
CHARGER_POWER = 3.0         # kW (1-hour time step)
CHARGE_EFF = 0.95
TARGET_SOC = 60.0           # %
INITIAL_SOC = 60.0          # %

DAY_TYPES = {
    "休日1": {
        "description": "当日0:00時点でSOC 60%。8–10時は買い物、17–20時は外食。それ以外は自宅にEVがあります。",
        "start_hour": 0,
        "planned_departure_abs": 32,  # 翌8:00
        "availability_windows": [(0, 8), (10, 17), (20, 32)],
        "summary": ["08:00–10:00 買い物", "17:00–20:00 外食", "翌08:00 次の出発"],
        # SOC consumption while away (%-points per hour)
        "away_soc_per_hour": 4.0,
        "required_soc": 60.0,
    },
    "平日": {
        "description": "当日0:00時点でSOC 60%。朝8時に出発し、18時に帰宅。翌朝8時まで充電できます。",
        "start_hour": 0,
        "planned_departure_abs": 32,  # 翌8:00
        "availability_windows": [(0, 8), (18, 32)],
        "summary": ["08:00-18:00 仕事", "翌08:00 出発"],
        # SOC consumption while away (%-points per hour)
        "away_soc_per_hour": 3,
        "required_soc": 60.0,
    },
}

DAY_TYPES["休日2"] = {
    "description": "当日0:00時点でSOC 60%。16–18時は買い物。それ以外は自宅。前日は終日曇りの予測、朝7時に昼から晴れる予測へ更新。",
    "start_hour": 0,
    "planned_departure_abs": 32,
    "availability_windows": [(0, 16), (18, 32)],
    "summary": ["16:00–18:00 買い物", "翌08:00 次の出発"],
    "away_soc_per_hour": 4.0,
    "required_soc": 60.0,
}
DAY_TYPES = {key: DAY_TYPES[key] for key in ("平日", "休日1", "休日2")}

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

    # Original grid carbon-intensity forecast.
    if day_type == "平日":
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

    if day_type == "休日2":
        # Higher daytime grid intensity, smoothly raised between 08:00 and 18:00.
        daytime_bump = np.where(
            (hod >= 8) & (hod <= 18),
            0.18 * np.sin(np.pi * (hod - 8) / 10.0) ** 2,
            0.0,
        )
        ci_fc = ci_fc + daytime_bump

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
    # At 07:00, revise evening and next-morning grid mix expectations.
    ci_shift = np.where(hours >= 7,
        0.025 * np.exp(-0.5 * ((hours - 19) / 2.5) ** 2)
        - 0.035 * np.exp(-0.5 * ((hours - 27) / 2.5) ** 2), 0.0)
    ci_upd = ci_fc + ci_shift
    ci_act = ci_fc + 1.15 * ci_shift
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

    if day_type == "休日2":
        # Cloudy original forecast; at 07:00, clearing from noon is expected.
        clear_pv = pv_fc.copy()
        pv_fc = clear_pv * 0.25
        pv_upd = pv_fc.copy()
        clearing = (hours >= 12) & (hours < 24)
        pv_upd[clearing] = clear_pv[clearing] * 0.95
        pv_act = clear_pv * 0.25
        pv_act[clearing] = clear_pv[clearing] * 0.90

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
    df["ci_upd"] = np.clip(ci_upd, 0.15, 0.8)
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



# Three-stage student workflow. Durable plans are independent of widget state.
def reset_exercise():
    st.session_state["step"] = 1
    st.session_state["draft_hours"] = []
    st.session_state["original_hours"] = []
    st.session_state["revised_hours"] = []


def toggle_hour(hour, field):
    values = set(st.session_state[field])
    if hour in values:
        values.remove(hour)
    else:
        values.add(hour)
    st.session_state[field] = sorted(values)


def confirm_original():
    st.session_state["original_hours"] = list(st.session_state["draft_hours"])
    st.session_state["revised_hours"] = list(st.session_state["draft_hours"])
    st.session_state["step"] = 2


def go_to(step):
    st.session_state["step"] = step


def keep_original():
    st.session_state["revised_hours"] = list(st.session_state["original_hours"])
    st.session_state["step"] = 3


def calculate(hours, mode):
    request = np.where(df["hour"].isin(hours), CHARGER_POWER, 0.0)
    schedule, soc, operation = execute_operation(
        df, request, day_type,
        availability_col="available_actual" if mode == "actual" else "available_plan",
    )
    return schedule, soc, operation, evaluate_executed_operation(df, schedule, soc, mode)


def profile_figure(mode, metric):
    fig = go.Figure()
    if metric == "電力需要・PV":
        cols = [("load_fc", "家庭需要予測", "#546e7a", "solid"),
                ("pv_fc", "前日のPV予測", "#3277b3", "solid")]
        if mode == "updated":
            cols.append(("pv_upd", "朝7時のPV予測", "#269460", "dash"))
        unit = "kW"
    else:
        cols = [("ci_fc" if metric == "CO₂原単位" else "price_fc", metric, "#3277b3", "solid")]
        if metric == "CO₂原単位":
            cols[0] = ("ci_fc", "前日のCO₂原単位予測", "#3277b3", "solid")
            if mode == "updated":
                cols.append(("ci_upd", "朝7時のCO₂原単位予測", "#269460", "dash"))
        unit = "kg-CO₂/kWh" if metric == "CO₂原単位" else "円/kWh"
    for column, label, color, dash in cols:
        fig.add_trace(go.Scatter(x=df["hour"], y=df[column], name=label,
                                line=dict(color=color, dash=dash, width=2)))
    if mode == "updated":
        fig.add_vrect(x0=0, x1=7, fillcolor="#adb5bd", opacity=0.15, line_width=0)
        fig.add_vline(x=7, line_dash="dash", line_width=1)
    fig.update_layout(height=150, template="plotly_white", margin=dict(t=25,b=20,l=35,r=15),
                      legend=dict(orientation="h", y=1.18), yaxis_title=unit)
    ticks=list(range(0,33,4))
    fig.update_xaxes(range=[0,32],tickvals=ticks,
                     ticktext=[f"{'翌' if h>=24 else ''}{h%24:02d}:00" for h in ticks])
    return fig


def render_editor(field, fixed_before=0):
    selected=set(st.session_state[field])
    st.markdown('<div style="display:flex;font-size:12px;color:#666;line-height:20px;min-height:32px"><span style="width:75%">当日 0〜24時</span><span>翌日 0〜8時</span></div>', unsafe_allow_html=True)
    with st.container(key="slotbar"):
        columns=st.columns(32, gap="small")
        for h,column in enumerate(columns):
            available=bool(df.loc[df["hour"]==h,"available_plan"].iloc[0])
            fixed=h<fixed_before
            status=("充電済" if h in selected else "固定") if fixed else ("外出" if not available else ("充電" if h in selected else "未選択"))
            with column:
                st.button(f"{h%24:02d}", help=f"{'翌日' if h>=24 else '当日'} {h%24:02d}:00〜{h%24+1:02d}:00 · {status}",
                          key=f"slot_{field}_{h}", disabled=fixed or not available,
                          type="primary" if h in selected else "secondary",
                          on_click=toggle_hour,args=(h,field),use_container_width=True)


def render_soc(schedule, soc):
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=df["hour"][:32]+0.5,y=schedule[:32],name="充電電力"),secondary_y=False)
    fig.add_trace(go.Scatter(x=np.arange(33),y=np.r_[INITIAL_SOC,soc[:32]],name="SOC"),secondary_y=True)
    fig.add_hline(y=required_soc,line_dash="dot",secondary_y=True)
    fig.update_yaxes(range=[0,CHARGER_POWER*1.25],title_text="kW",secondary_y=False)
    fig.update_yaxes(range=[0,100],title_text="SOC [%]",secondary_y=True)
    fig.update_layout(height=175,template="plotly_white",margin=dict(t=30,b=30,l=35,r=35),legend=dict(orientation="h",y=1.2))
    ticks=list(range(0,33,4));fig.update_xaxes(tickvals=ticks,ticktext=[f"{'翌' if h>=24 else ''}{h%24:02d}:00" for h in ticks])
    st.plotly_chart(fig,use_container_width=True)


if "step" not in st.session_state:
    reset_exercise()
step=st.session_state["step"]
with st.sidebar:
    st.markdown("### 生活パターン")
    day_type=st.radio("生活パターン",list(DAY_TYPES),key="exercise_day",disabled=step!=1,
                      on_change=reset_exercise,label_visibility="collapsed")
    cfg=DAY_TYPES[day_type]
    st.markdown("### EVの利用予定")
    for item in cfg["summary"]:
        st.write(item)
    st.markdown("### 充電条件")
    st.write(f"初期SOC：{INITIAL_SOC:g}% ／ 翌朝必要SOC：{cfg['required_soc']:g}%以上")
    st.write("容量：40 kWh ／ 充電電力：3 kW")
    st.caption("1時間枠ごとに充電。効率95%。満充電時には停止。")
    st.caption(f"外出中のSOC消費：1時間あたり{cfg['away_soc_per_hour']:g}ポイント")
    st.caption("CO₂原単位は予測・更新の対象。料金は固定。CO₂・料金の評価対象はEV充電に伴う系統購入分。")
    st.button("最初から計画を作り直す",on_click=reset_exercise,use_container_width=True)

df,_,_=generate_profiles(day_type)
required_soc=cfg["required_soc"]
for col,i,title in zip(st.columns(3),[1,2,3],["充電計画の作成","予測更新と見直し","実績条件での比較"]):
    with col:
        color = "#17365d" if step == i else "#b3bac3"
        weight = "700" if step == i else "400"
        st.markdown(
            f'<div style="color:{color};font-weight:{weight};'
            f'font-size:14px;line-height:24px;padding-bottom:8px;">'
            f'{i}．{title}</div>',
            unsafe_allow_html=True,
        )

if step in (1,2):
    mode="forecast" if step==1 else "updated"
    field="draft_hours" if step==1 else "revised_hours"
    if step==2:
        # Always preserve the original executed past, even across navigation.
        st.session_state[field]=sorted([h for h in st.session_state['original_hours'] if h<7]+[h for h in st.session_state[field] if h>=7])
        weather_update = "昼から晴れる予測へ更新" if day_type == "休日2" else "午前のPV予測を下方修正"
        st.caption(f"朝7時：{weather_update}。CO₂原単位も更新。0〜7時は固定。")
    else:
        weather = "終日曇り" if day_type == "休日2" else "晴れ"
        st.caption(f"前日の予測：{weather}。当日0時〜翌朝8時の充電時間を選択。")
    metric=st.radio("予測情報",["電力需要・PV","CO₂原単位","電気料金"],horizontal=True)
    st.plotly_chart(profile_figure(mode,metric),use_container_width=True)
    st.caption("充電する開始時刻を選択（1枠＝1時間）　色付き：充電 ／ グレー：外出中・経過済み")
    render_editor(field,7 if step==2 else 0)
    schedule,soc,operation,result=calculate(st.session_state[field],mode)
    render_soc(schedule,soc)
    status=feasibility_label(result,operation,required_soc)
    st.caption(f"選択：{len(st.session_state[field])}時間分 ／ 総充電量：{result['charge_kwh']:.1f} kWh ／ 制約：{status} ／ 点線：必要SOC")
    cols=st.columns(4)
    for col,label,value in zip(cols,["翌朝の出発時SOC","CO₂排出量（予測）","充電コスト（予測）","PVからの充電（予測）"],
                               [f"{result['final_soc']:.1f}%",f"{result['co2_kg']:.2f} kg",f"{result['cost_yen']:.0f} 円",f"{result['pv_to_ev_kwh']:.1f} kWh"]):
        col.metric(label,value)
    if step==2:
        st.button("この計画で結果へ進む",on_click=go_to,args=(3,),disabled=status!="充足",type="primary",use_container_width=True)
    else:
        st.button("この計画を確定して朝7時へ進む",on_click=confirm_original,disabled=status!="充足",type="primary",use_container_width=True)
else:
    original_schedule,original_soc,original_op,original_result=calculate(st.session_state['original_hours'],'actual')
    revised_schedule,revised_soc,revised_op,revised_result=calculate(st.session_state['revised_hours'],'actual')
    _,_,_,original_estimate=calculate(st.session_state['original_hours'],'forecast')
    _,_,_,updated_estimate=calculate(st.session_state['revised_hours'],'updated')
    st.caption("両計画を同じ実績条件で比較。灰色：朝7時以前の固定済みの実行結果。")
    metric=st.radio("実績情報",["エネルギー","CO₂原単位","電気料金"],horizontal=True)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.15,
        specs=[[{}],[{"secondary_y":True}]],subplot_titles=("実線：実績 ／ 薄い点線：前日予測 ／ 薄い破線：朝7時予測", "充電電力とSOC：元の計画・更新後の計画"))
    series = [("pv_act","PV実績","#e7a21b"),("load_act","家庭需要実績","#546e7a")] if metric=="エネルギー" else [("ci_act" if metric=="CO₂原単位" else "price_act",metric,"#546e7a")]
    for key,label,color in series:
        base = key.removesuffix("_act")
        quantity = label.removesuffix("実績")
        for suffix, forecast_label, dash, opacity in [
            ("fc", "前日予測", "dot", 0.45),
            ("upd", "朝7時予測", "dash", 0.65),
        ]:
            visible_hours = df.hour >= 7 if suffix == "upd" else df.hour >= 0
            fig.add_trace(go.Scatter(
                x=df.loc[visible_hours, "hour"], y=df.loc[visible_hours, f"{base}_{suffix}"],
                name=f"{quantity}：{forecast_label}", showlegend=False,
                line=dict(color=color, width=1.5, dash=dash), opacity=opacity,
                hovertemplate=f"{quantity}：{forecast_label}<br>時刻 %{{x}}<br>%{{y:.3f}}<extra></extra>",
            ), row=1, col=1)
        # Draw actual values last so coincident forecasts do not obscure them.
        fig.add_trace(go.Scatter(x=df.hour,y=df[key],name=f"{quantity}実績",line=dict(color=color,width=2.2)),row=1,col=1)
    for schedule,soc,label,color in [(original_schedule,original_soc,"元の計画","#3277b3"),(revised_schedule,revised_soc,"更新後","#269460")]:
        fig.add_trace(go.Bar(x=df.hour[:32]+0.5,y=schedule[:32],name=label+" 充電",marker_color=color),row=2,col=1)
        fig.add_trace(go.Scatter(x=np.arange(33),y=np.r_[INITIAL_SOC,soc[:32]],name=label+" SOC",line=dict(color=color,dash="dot" if label=="元の計画" else "solid")),row=2,col=1,secondary_y=True)
    fig.update_yaxes(title_text="kW" if metric=="エネルギー" else ("kg-CO₂/kWh" if metric=="CO₂原単位" else "円/kWh"),row=1,col=1)
    fig.update_yaxes(title_text="kW",range=[0,3.8],row=2,col=1,secondary_y=False)
    fig.update_yaxes(title_text="SOC [%]",range=[0,100],showgrid=False,row=2,col=1,secondary_y=True)
    fig.add_hline(y=required_soc,line_dash="dot",line_width=1,row=2,col=1,secondary_y=True)
    fig.add_vrect(x0=0,x1=7,fillcolor="#adb5bd",opacity=0.12,line_width=0)
    ticks=list(range(0,33,4))
    fig.update_xaxes(range=[0,32],tickvals=ticks,ticktext=[f"{'翌' if h>=24 else ''}{h%24:02d}:00" for h in ticks])
    fig.update_layout(height=320,template="plotly_white",barmode="group",margin=dict(t=50,b=25,l=40,r=40),legend=dict(orientation="h",y=1.2,font=dict(size=11)))
    fig.update_annotations(font_size=12)
    st.plotly_chart(fig,use_container_width=True)
    st.caption("更新後の計画の実績 ／ 各予測値は、その時点で選択した計画の評価")
    for col,label,key,fmt,unit in zip(st.columns(4),
        ["翌朝の出発時SOC","CO₂排出量","充電コスト","PVからの充電"],
        ["final_soc","co2_kg","cost_yen","pv_to_ev_kwh"],[".1f",".2f",".0f",".1f"],["%"," kg"," 円"," kWh"]):
        col.metric(label,format(revised_result[key],fmt)+unit)
        col.caption("前日の予測："+format(original_estimate[key],fmt)+unit+"  \n朝7時の予測："+format(updated_estimate[key],fmt)+unit)
    st.caption(f"総充電量：{revised_result['charge_kwh']:.1f} kWh ／ 制約：{feasibility_label(revised_result,revised_op,required_soc)}")
