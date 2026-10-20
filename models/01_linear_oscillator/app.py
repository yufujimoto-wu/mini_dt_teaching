import math
import time

import numpy as np
import pandas as pd
from ar_model import observe, fit_orders, rmse

import plotly.graph_objects as go
import streamlit as st

from oscillator import state_at

st.set_page_config(page_title="自由振動・センサ観測・ARモデル", layout="wide")
st.markdown("<style>.block-container{padding-top:1.5rem}div[data-testid='stMetricValue']{font-size:1.7rem}</style>", unsafe_allow_html=True)

DEFAULTS = dict(m=1.0, k=4.0, c=0.3, x0=0.5, v0=0.0, duration=20.0, speed=1.0, sensor_dt=0.2, noise=0.02, seed=42, train_fraction=0.65,
                max_order=12, order_method="BICで自動選択", manual_order=2, intercept=False)
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


def clear_model():
    ss.pop("ar_report", None)


def restart():
    clear_model()
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


def record_all():
    ss.t, ss.seek, ss.playing = ss.duration, ss.duration, False
    ss.anchor = time.monotonic()


def preset(name):
    ss.update(DEFAULTS)
    if name == "良好な観測":
        ss.update(c=0.15, sensor_dt=0.15, noise=0.005)
    elif name == "過学習を比較":
        ss.update(c=0.15, duration=10.0, sensor_dt=0.2, noise=0.08,
                  max_order=16, order_method="学習誤差最小")
    elif name == "信号がノイズに埋もれる":
        ss.update(x0=0.05, sensor_dt=0.8, noise=0.3, max_order=8)
    ss.rate = ss.speed
    restart()


def change_speed():
    advance()
    ss.rate = ss.speed


with st.sidebar:
    st.header("条件設定")
    st.button("初期設定に戻す", on_click=reset_defaults, width="stretch")
    with st.expander("授業用の設定例"):
        for name in ["良好な観測", "過学習を比較", "信号がノイズに埋もれる"]:
            st.button(name, on_click=preset, args=(name,), width="stretch")
        #st.caption("設定例を選んだ後，再生か「全時間を記録」を選択．結果は観測条件と乱数に依存．")
    st.subheader("物理パラメータ")
    st.slider("質量 m [kg]", 0.2, 10.0, step=0.1, key="m", on_change=restart)
    st.slider("ばね定数 k [N/m]", 0.2, 40.0, step=0.2, key="k", on_change=restart)
    st.slider("減衰係数 c [N·s/m]", 0.0, 20.0, step=0.1, key="c", on_change=restart)
    st.subheader("初期条件")
    st.slider("初期変位 x₀ [m]", -1.0, 1.0, step=0.05, key="x0", on_change=restart)
    st.slider("初期速度 v₀ [m/s]", -2.0, 2.0, step=0.1, key="v0", on_change=restart)
    st.subheader("センサの観測条件")
    st.select_slider("記録周期 Δt [s]", options=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0], key="sensor_dt", on_change=restart)
    st.slider("観測誤差の標準偏差 σ [m]", 0.0, 0.5, step=0.005, key="noise", on_change=restart)
    st.number_input("乱数シード", min_value=0, max_value=999999, step=1, key="seed", on_change=restart)
 #   st.caption("観測値 = 真の変位 + 独立な正規乱数．")
    st.subheader("ARモデルの学習")
    st.slider("学習に使う記録の割合", 0.4, 0.8, step=0.05, key="train_fraction", on_change=clear_model)
    st.slider("候補の最大次数", 2, 20, step=1, key="max_order", on_change=clear_model)
    st.selectbox("次数の選び方", ["BICで自動選択", "AICで自動選択", "学習誤差最小", "手動指定"], key="order_method", on_change=clear_model)
    st.slider("手動指定する次数 p", 0, 20, key="manual_order", on_change=clear_model, disabled=ss.order_method != "手動指定")
    st.checkbox("切片（定数項）も推定する", key="intercept", on_change=clear_model)
    st.subheader("再生設定")
    st.slider("計算時間 [s]", 5.0, 60.0, step=5.0, key="duration", on_change=restart)
    st.select_slider("再生速度", options=[0.25, 0.5, 1.0, 2.0], key="speed", on_change=change_speed, format_func=lambda v: f"{v:g} 倍")
    st.caption("物理パラメータ・初期条件・計算時間を変えると時刻0に戻って停止")

st.title("自由振動・センサ観測・ARモデル")
st.latex(r"m\ddot{x}(t)+c\dot{x}(t)+kx(t)=0")
#st.caption("外力のない，ばね・ダンパにつながれた台車の運動．右向きを変位の正方向")
buttons = st.columns(4)
buttons[0].button("▶ 再生", on_click=play, disabled=ss.playing, width="stretch")
buttons[1].button("⏸ 一時停止", on_click=pause, disabled=not ss.playing, width="stretch")
buttons[2].button("↶ 先頭に戻る", on_click=restart, width="stretch")
buttons[3].button("全時間を記録して停止", on_click=record_all, width="stretch")
st.slider("時刻を指定 [s]（操作すると一時停止）", 0.0, ss.duration, step=0.05, key="seek", on_change=seek)
#st.caption("上のスライダーは時刻指定用．現在の再生時刻は下に表示．")

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


@st.cache_data(show_spinner=False)
def sensor_records(duration, interval, noise, seed, params):
    return observe(duration, interval, noise, seed, params)


sensor_t, sensor_truth, sensor_y = sensor_records(ss.duration, ss.sensor_dt, ss.noise, ss.seed, params)
observation_limit = max(limit, float(np.max(np.abs(sensor_y))) * 1.12)
if zeta < 1:
    damped_period = 2 * math.pi / (math.sqrt(ss.k / ss.m) * math.sqrt(1 - zeta ** 2))
    st.caption(f"減衰振動の1周期あたりの記録数：約 {damped_period / ss.sensor_dt:.1f} 点")
    if ss.sensor_dt >= damped_period / 2:
        st.info("記録周期が振動周期の半分以上．標本化によって元の振動と異なる動きに見える可能性．")


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
    return f'''<svg viewBox="0 0 800 300" width="100%" role="img" aria-label="ばねとダンパにつながれた台車．変位 {x:.3f} メートル">
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


def show_ar_report(report):
    result, chosen = report['result'], report['chosen']
    ts, y, truth = report['times'], report['y'], report['truth']
    split, start, p = result['split'], result['start'], chosen['p']
    st.write(f"**AR({p})** を採用：{report['method']}　／　記録 {len(y)} 点，学習 {split} 点，検証 {len(y)-split} 点")
    st.caption(f"Δt = {report['dt']:g} s．候補次数 0〜{result['pmax']}（データ数に応じて上限を調整）．全候補を同じ {split-start} 個の目的値で学習・比較します．")
    st.latex(r"y_n=x(n\Delta t)+\varepsilon_n,\qquad \varepsilon_n\sim\mathcal{N}(0,\sigma^2)")
    st.latex(r"\widehat{y}_n=b+\sum_{j=1}^{p}a_j y_{n-j}")
    bias = float(chosen['coef'][0]) if report['intercept'] else 0.0
    weights = chosen['coef'][1:] if report['intercept'] else chosen['coef']
    terms = ([f"({bias:.4g})"] if report['intercept'] else [])
    terms += [rf"({float(a):.4g})y_{{n-{j}}}" for j, a in enumerate(weights, 1)]
    if not terms:
        terms = ['0']
    chunks = [' + '.join(terms[i:i+2]) for i in range(0, len(terms), 2)]
    lines = [r"\widehat{y}_n &= " + chunks[0]] + [r'&\quad + ' + x for x in chunks[1:]]
    st.latex(r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}")
#    if p == 0:
        #st.info("AR(0)は過去の変位を未使用．切片なしなら常に0，切片ありなら学習した定数を予測．単純なモデルが選ばれても、元の運動が単純だと確定したわけではありません。")
#    if not report['intercept']:
#        st.caption("平衡位置が既知なので切片 b = 0 に固定しています。表示式の係数は丸めており、計算には丸める前の値を使います。")
    metrics = st.columns(4)
    metrics[0].metric("学習 RMSE", f"{chosen['train_rmse']:.4g} m")
    metrics[1].metric("検証：1ステップ RMSE", f"{chosen['validation_rmse']:.4g} m")
    metrics[2].metric("検証：連続予測 RMSE", f"{chosen['recursive_rmse']:.4g} m")
    metrics[3].metric("検証：ゼロ予測 RMSE", f"{rmse(y[split:], np.zeros(len(y)-split)):.4g} m")
#    st.caption("上の誤差はセンサ記録に対する値です。連続予測は検証中の観測値を入力しません。")
    fig = go.Figure()
    boundary = (float(ts[split-1]) + float(ts[split])) / 2
    fig.add_vrect(x0=0, x1=boundary, fillcolor='#dbeafe', opacity=0.4, line_width=0, annotation_text='学習区間', annotation_position='top left')
    fig.add_vrect(x0=boundary, x1=float(ts[-1]), fillcolor='#fef3c7', opacity=0.35, line_width=0, annotation_text='検証区間', annotation_position='top right')
    dense_t = np.linspace(float(ts[0]), float(ts[-1]), 1500)
    dense_x = [state_at(float(ti), *params)[0] for ti in dense_t]
    fig.add_trace(go.Scatter(x=dense_t, y=dense_x, mode='lines', name='真の変位（比較用）', line=dict(color='#94a3b8', width=2)))
    fig.add_trace(go.Scatter(x=ts, y=y, mode='markers', name='センサ記録', marker=dict(symbol='x', color='#dc2626', size=8)))
    fig.add_trace(go.Scatter(x=ts[start:split], y=chosen['fitted'], mode='lines+markers', name='学習区間の1ステップ当てはまり', line=dict(color='#7c3aed', width=2)))
    fig.add_trace(go.Scatter(x=ts[split:], y=chosen['one_step'], mode='lines+markers', name='検証：1ステップ予測', line=dict(color='#16a34a', dash='dot')))
    fig.add_trace(go.Scatter(x=ts[split:], y=chosen['recursive'], mode='lines+markers', name='検証：連続予測', line=dict(color='#ea580c', width=3)))
    scale = max(0.1, float(np.max(np.abs(y))), float(np.max(np.abs(truth)))) * 1.3
    fig.update_layout(height=440, margin=dict(t=120, b=45), legend=dict(orientation='h', y=1.38), xaxis_title='時間 t [s]', yaxis_title='変位 [m]', yaxis_range=[-scale, scale])
    st.plotly_chart(fig, width="stretch", key='ar_prediction')
 #   if not np.all(np.isfinite(chosen['recursive'])) or np.any(np.abs(chosen['recursive']) > scale):
 #       st.warning("連続予測が表示範囲を超えています。観測値が見えるよう縦軸を固定しています。予測の発散・振幅増大は誤差指標と下の表でも確認できます。")
    truth_error = rmse(truth[split:], chosen['recursive'])
    st.caption(f"参考：仮想実験でのみ分かる『真の変位』に対する連続予測 RMSE = {truth_error:.4g} m。真の変位は学習・次数選択に使っていません。")
    rows = result['rows']
    scores = go.Figure()
    for field, name, color in [('train_rmse', '学習', '#7c3aed'), ('validation_rmse', '検証：1ステップ', '#16a34a'), ('recursive_rmse', '検証：連続予測', '#ea580c')]:
        scores.add_trace(go.Scatter(x=[r['p'] for r in rows], y=[max(r[field], 1e-12) if math.isfinite(r[field]) else None for r in rows], mode='lines+markers', name=name, line=dict(color=color)))
    scores.add_vline(x=p, line_dash='dot', line_color='#64748b')
    scores.update_layout(title='次数と誤差の比較', height=320, yaxis=dict(type='log', title='RMSE [m]（対数目盛）'), xaxis=dict(title='次数 p', dtick=1), legend=dict(orientation='h', y=1.12), margin=dict(t=80))
    st.plotly_chart(scores, width="stretch", key='ar_order_errors')
    st.caption('次数ごとの検証誤差は比較用．次数の自動選択には未使用．')
    with st.expander('次数ごとの誤差・AIC・BIC・係数・記録データ'):
        st.dataframe(pd.DataFrame([dict(次数=r['p'], 学習RMSE=r['train_rmse'], 検証1ステップRMSE=r['validation_rmse'], 検証連続予測RMSE=r['recursive_rmse'], AIC=r['aic'], BIC=r['bic']) for r in rows]), hide_index=True)
        st.caption('AIC・BICは小さい値を採用．分散を1パラメータとして数え全候補共通の定数項は省略．')
        if p:
            st.dataframe(pd.DataFrame({'係数': [f'a{j}' for j in range(1, p+1)], '値': weights}), hide_index=True)
            radius = max(abs(np.roots(np.r_[1.0, -weights])))
            st.write(f'漸化式の特性根の最大絶対値：{radius:.4g}')
            if radius > 1 + 1e-8:
                st.caption('1を超える特性根があり長時間の連続予測が増大する可能性．')
        st.dataframe(pd.DataFrame({'時間 [s]': ts, '観測値 [m]': y, '区間': ['学習']*split + ['検証']*(len(y)-split)}), hide_index=True)
        st.download_button('記録データをCSVで保存', pd.DataFrame({'time_s': ts, 'observed_m': y, 'true_m': truth, 'split': ['train']*split + ['validation']*(len(y)-split)}).to_csv(index=False).encode('utf-8-sig'), file_name='sensor_records.csv', mime='text/csv', key='download_records')


@st.fragment(run_every=0.1 if ss.playing else None)
def animation():
    was_playing = ss.playing
    advance()
    t = ss.t
    x, v = state_at(t, *params)
    metrics = st.columns(4)
    observed_n = int(np.searchsorted(sensor_t, t + 1e-9, side="right"))
    metrics[0].metric("現在の時刻", f"{t:.2f} s")
    metrics[1].metric("変位 x(t)", f"{x:+.3f} m")
    metrics[2].metric("速度 v(t)", f"{v:+.3f} m/s")
    metrics[3].metric("記録済みの観測点", f"{observed_n} 点")
    left, right = st.columns([1, 1.15])
    with left:
        st.subheader("台車の運動")
        st.markdown(cart_svg(x), unsafe_allow_html=True)
        # st.caption(f"台車の表示倍率は初期条件に合わせて調整．同じ再生中は固定．変位の基準範囲：±{limit:.2f} m")
    with right:
        st.subheader("変位の時間変化")
        stop = min(len(times), int(t / ss.duration * (len(times) - 1)) + 1)
        tx, xx = times[:stop], values[:stop]
        if tx[-1] < t:
            tx, xx = tx + [t], xx + [x]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tx, y=xx, mode="lines", line=dict(color="#2563eb", width=3), name="真の変位"))
        fig.add_trace(go.Scatter(x=[t], y=[x], mode="markers", marker=dict(color="#2563eb", size=8), name="現在位置", showlegend=False))
        fig.add_trace(go.Scatter(x=sensor_t[:observed_n], y=sensor_y[:observed_n], mode="markers", marker=dict(color="#dc2626", symbol="x", size=10, line=dict(width=2)), name="センサ記録"))
        fig.add_vline(x=t, line_color="#94a3b8", line_dash="dot")
        fig.update_layout(height=380, margin=dict(l=45, r=15, t=50, b=45), showlegend=True, legend=dict(orientation="h", y=1.15), uirevision="fixed", xaxis=dict(title="時間 t [s]", range=[0, ss.duration], fixedrange=True), yaxis=dict(title="変位 x [m]", range=[-observation_limit, observation_limit], zeroline=True, fixedrange=True))
        st.plotly_chart(fig, width="stretch", key="displacement", config={"displayModeBar": False})
    st.progress(min(1.0, t / ss.duration))
    st.caption("赤い×は記録周期ごとの観測値．線は連続時間の真の変位でARモデルの学習には赤い×だけを使用．")
    st.subheader("記録したデータからARモデルを学習")
    st.caption("現在時刻までの記録を固定し前半を学習，後半を検証に利用．")
    if st.button("現在までの記録で学習する", disabled=observed_n < 12, key="fit_ar"):
        try:
            result = fit_orders(sensor_y[:observed_n], ss.train_fraction, ss.max_order, ss.intercept)
            method = {"BICで自動選択": "bic", "AICで自動選択": "aic", "学習誤差最小": "train_rmse"}.get(ss.order_method)
            if method:
                chosen = min(result["rows"], key=lambda r: r[method])
            else:
                chosen = next((r for r in result["rows"] if r["p"] == ss.manual_order), None)
                if chosen is None:
                    raise ValueError(f"指定次数を推定不能．候補上限は {result['pmax']} ．次数を下げるか記録を増やす必要")
            ss.ar_report = dict(result=result, chosen=chosen, times=sensor_t[:observed_n].copy(),
                                truth=sensor_truth[:observed_n].copy(), y=sensor_y[:observed_n].copy(),
                                method=ss.order_method, intercept=ss.intercept, dt=ss.sensor_dt)
            ss.playing = False
            st.rerun()
        except ValueError as exc:
            st.warning(str(exc))
    if observed_n < 12:
        st.caption("学習には12点以上の記録が必要．")
    if "ar_report" in ss:
        show_ar_report(ss.ar_report)
    if was_playing and not ss.playing:
        st.rerun()  # Refresh buttons and disable periodic reruns after completion.


#with st.expander("このデモで比較すること"):
#    st.markdown("""
#- **良好な観測**：短い記録周期、小さな観測誤差で学習します。
#- **過学習を比較**：次数の選び方を「学習誤差最小」「BICで自動選択」「手動指定」で切り替え、再学習します。高次モデルの学習誤差と検証誤差を比較します。
#- **信号がノイズに埋もれる**：小さな振幅に大きな観測誤差が重なると、BICがAR(0)などの単純なモデルを選ぶ場合があります。
#
#1ステップ予測は直前の**観測値**を毎回使います。連続予測は学習区間の末尾から出発し、以後は**自分の予測値**を使います。
#検証区間は次数の自動選択に使いません。この画面を見ながら設定を調整した場合、最終確認には別の乱数シードなどで再検証してください。
#自由振動は時間とともに減衰するため、後半の信号が小さくなって誤差も小さく見える場合があります。ゼロ予測とも比較します。
#ノイズのない線形自由振動の標本列は原理上AR(2)で表せますが、観測ノイズが加わった記録では選択次数が2になるとは限りません。
#    """)
animation()
#st.caption("計算は線形運動方程式の解析解を使用．描画は約0.1秒間隔を目標に更新し，再生時刻は実経過時間から求める．")
