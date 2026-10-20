import math
import time

import plotly.graph_objects as go
import streamlit as st

from oscillator import state_at

st.set_page_config(page_title="線形自由振動のデジタルツイン", layout="wide")
st.markdown("<style>.block-container{padding-top:1.5rem}div[data-testid='stMetricValue']{font-size:1.7rem}</style>", unsafe_allow_html=True)

DEFAULTS = dict(m=1.0, k=4.0, c=0.3, x0=0.5, v0=0.0, duration=20.0, speed=1.0)
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
for key, value in dict(t=0.0, playing=False, anchor=time.monotonic(), rate=1.0, seek=0.0).items():
    st.session_state.setdefault(key, value)
ss = st.session_state


def advance():
    now = time.monotonic()
    if ss.playing:
        ss.t = min(ss.duration, ss.t + (now - ss.anchor) * ss.rate)
        if ss.t >= ss.duration:
            ss.playing = False
    ss.anchor = now


def restart():
    ss.t, ss.seek, ss.playing = 0.0, 0.0, False
    ss.anchor = time.monotonic()


def reset_defaults():
    ss.update(DEFAULTS)
    ss.rate = ss.speed
    restart()


def play():
    if ss.t >= ss.duration:
        ss.t = 0.0
    ss.anchor, ss.playing = time.monotonic(), True


def pause():
    advance()
    ss.playing = False


def seek():
    ss.t, ss.playing, ss.anchor = ss.seek, False, time.monotonic()


def change_speed():
    advance()
    ss.rate = ss.speed


with st.sidebar:
    st.header("条件設定")
    st.button("初期設定に戻す", on_click=reset_defaults, use_container_width=True)
    st.subheader("物理パラメータ")
    st.slider("質量 m [kg]", 0.2, 10.0, step=0.1, key="m", on_change=restart)
    st.slider("ばね定数 k [N/m]", 0.2, 40.0, step=0.2, key="k", on_change=restart)
    st.slider("減衰係数 c [N·s/m]", 0.0, 20.0, step=0.1, key="c", on_change=restart)
    st.subheader("初期条件")
    st.slider("初期変位 x₀ [m]", -1.0, 1.0, step=0.05, key="x0", on_change=restart)
    st.slider("初期速度 v₀ [m/s]", -2.0, 2.0, step=0.1, key="v0", on_change=restart)
    st.subheader("再生設定")
    st.slider("計算時間 [s]", 5.0, 60.0, step=5.0, key="duration", on_change=restart)
    st.select_slider("再生速度", options=[0.25, 0.5, 1.0, 2.0], key="speed", on_change=change_speed, format_func=lambda v: f"{v:g} 倍")
    st.caption("物理パラメータ・初期条件・計算時間を変えると、時刻0に戻って停止します。")

st.title("線形自由振動のデジタルツイン")
st.latex(r"m\ddot{x}(t)+c\dot{x}(t)+kx(t)=0")
st.caption("外力のない、ばね・ダンパにつながれた台車の運動。右向きを変位の正方向とします。")
buttons = st.columns([1, 1, 1, 3])
buttons[0].button("▶ 再生", on_click=play, disabled=ss.playing, use_container_width=True)
buttons[1].button("⏸ 一時停止", on_click=pause, disabled=not ss.playing, use_container_width=True)
buttons[2].button("↶ 先頭に戻る", on_click=restart, use_container_width=True)
st.slider("時刻を指定 [s]（操作すると一時停止）", 0.0, ss.duration, step=0.05, key="seek", on_change=seek)
st.caption("上のスライダーは時刻指定用です。現在の再生時刻は下に表示します。")

params = (ss.m, ss.k, ss.c, ss.x0, ss.v0)
zeta = ss.c / (2 * math.sqrt(ss.m * ss.k))
regime = "減衰なし" if ss.c == 0 else "不足減衰" if zeta < 1 - 1e-8 else "臨界減衰" if abs(zeta - 1) <= 1e-8 else "過減衰"
st.write(f"**{regime}**　減衰比 ζ = {zeta:.3f}　固有周期 Tₙ = {2 * math.pi * math.sqrt(ss.m / ss.k):.2f} s")


@st.cache_data(show_spinner=False)
def trajectory(duration, params):
    # Dense enough for the highest natural frequency allowed by the controls.
    n = max(1000, math.ceil(duration / 0.01))
    times = [duration * i / n for i in range(n + 1)]
    values = [state_at(t, *params)[0] for t in times]
    # Mechanical energy bounds displacement, including initial velocity.
    amplitude = math.sqrt(params[3] ** 2 + params[0] / params[1] * params[4] ** 2)
    return times, values, max(0.1, amplitude * 1.15)


times, values, limit = trajectory(ss.duration, params)


def cart_svg(x):
    # Fixed scale throughout a run: wall, equilibrium and viewing window stay still.
    cx = 540 + 170 * x / limit
    left = cx - 55
    spring_start, spring_end = 95, left - 18
    points = [(spring_start, 128), (spring_start + 20, 128)]
    for i in range(17):
        px = spring_start + 25 + (spring_end - spring_start - 45) * i / 16
        points.append((px, 128 + (13 if i % 2 else -13)))
    points.extend([(spring_end, 128), (left, 128)])
    poly = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
    return f'''<svg viewBox="0 0 800 300" width="100%" role="img" aria-label="ばねとダンパにつながれた台車。変位 {x:.3f} メートル">
    <rect width="800" height="300" rx="12" fill="#f7f9fc"/>
    <line x1="45" y1="237" x2="765" y2="237" stroke="#64748b" stroke-width="5"/>
    <rect x="70" y="60" width="25" height="177" fill="#64748b"/>
    <line x1="540" y1="58" x2="540" y2="237" stroke="#94a3b8" stroke-dasharray="6 6"/>
    <text x="540" y="43" text-anchor="middle" fill="#475569" font-size="17">平衡位置 x = 0</text>
    <polyline points="{poly}" fill="none" stroke="#2563eb" stroke-width="4"/>
    <text x="235" y="91" fill="#2563eb" font-size="19">ばね k</text>
    <line x1="95" y1="185" x2="240" y2="185" stroke="#7c3aed" stroke-width="4"/>
    <path d="M 305 169 H 240 V 201 H 305" fill="none" stroke="#7c3aed" stroke-width="4"/>
    <line x1="275" y1="173" x2="275" y2="197" stroke="#7c3aed" stroke-width="4"/>
    <line x1="275" y1="185" x2="{left:.2f}" y2="185" stroke="#7c3aed" stroke-width="4"/>
    <text x="225" y="226" fill="#7c3aed" font-size="17">ダンパ c</text>
    <rect x="{left:.2f}" y="107" width="110" height="98" rx="7" fill="#bfdbfe" stroke="#1e40af" stroke-width="3"/>
    <text x="{cx:.2f}" y="163" text-anchor="middle" fill="#1e3a8a" font-size="26">m</text>
    <circle cx="{cx-32:.2f}" cy="220" r="15" fill="#94a3b8" stroke="#334155" stroke-width="3"/>
    <circle cx="{cx+32:.2f}" cy="220" r="15" fill="#94a3b8" stroke="#334155" stroke-width="3"/>
    <text x="400" y="278" text-anchor="middle" fill="#475569" font-size="17">台車の変位 x(t)　右向きが正 →</text>
    </svg>'''


@st.fragment(run_every=0.1 if ss.playing else None)
def animation():
    was_playing = ss.playing
    advance()
    t = ss.t
    x, v = state_at(t, *params)
    metrics = st.columns(3)
    metrics[0].metric("現在の時刻", f"{t:.2f} s")
    metrics[1].metric("変位 x(t)", f"{x:+.3f} m")
    metrics[2].metric("速度 v(t)", f"{v:+.3f} m/s")
    left, right = st.columns([1, 1.15])
    with left:
        st.subheader("台車の運動")
        st.markdown(cart_svg(x), unsafe_allow_html=True)
        st.caption(f"台車の表示倍率は初期条件に合わせて調整します。同じ再生中は固定です。変位の基準範囲：±{limit:.2f} m")
    with right:
        stop = min(len(times), int(t / ss.duration * (len(times) - 1)) + 1)
        tx, xx = times[:stop], values[:stop]
        if tx[-1] < t:
            tx, xx = tx + [t], xx + [x]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tx, y=xx, mode="lines", line=dict(color="#2563eb", width=3), name="変位"))
        fig.add_trace(go.Scatter(x=[t], y=[x], mode="markers", marker=dict(color="#dc2626", size=10), name="現在位置"))
        fig.add_vline(x=t, line_color="#94a3b8", line_dash="dot")
        fig.update_layout(title="変位の時間変化", height=340, margin=dict(l=45, r=15, t=50, b=45), showlegend=False, uirevision="fixed", xaxis=dict(title="時間 t [s]", range=[0, ss.duration], fixedrange=True), yaxis=dict(title="変位 x [m]", range=[-limit, limit], zeroline=True, fixedrange=True))
        st.plotly_chart(fig, use_container_width=True, key="displacement", config={"displayModeBar": False})
    st.progress(min(1.0, t / ss.duration))
    if was_playing and not ss.playing:
        st.rerun()  # Refresh buttons and disable periodic reruns after completion.


animation()
st.caption("計算は線形運動方程式の解析解を使用します。描画は約0.1秒間隔を目標に更新し、再生時刻は実経過時間から求めます。")
