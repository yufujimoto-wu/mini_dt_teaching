import math
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from discovery import true_state_at as state_at, true_path
from discovery import records, derivative_data, stlsq, rollout, NAMES

st.set_page_config(page_title='台車の運動から方程式を探す：SINDy', layout='wide')
st.markdown('<style>.block-container{padding-top:1.5rem}div[data-testid="stMetricValue"]{font-size:1.6rem}</style>', unsafe_allow_html=True)
DEFAULTS = dict(m=1.0, k=4.0, c=0.3, x0=0.5, v0=0.0, duration=20.0, speed=1.0,
                sensor_dt=0.1, noise=0.005, seed=42, fraction=0.65,
                mode='理想条件：正確な変位・速度・加速度', window=11, degree=3, threshold=0.03, quadratic=False, q=2.0)
ss = st.session_state
for key, value in DEFAULTS.items():
    ss.setdefault(key, value)
for key, value in dict(t=0.0, seek=0.0, playing=False, anchor=time.monotonic(), rate=1.0).items():
    ss.setdefault(key, value)

def clear_model():
    ss.pop('report', None)

def advance():
    now = time.monotonic()
    if ss.playing:
        ss.t = min(ss.duration, ss.t + (now-ss.anchor)*ss.rate)
        if ss.t >= ss.duration:
            ss.playing = False
    ss.anchor = now

def restart():
    clear_model()
    ss.t, ss.seek, ss.playing, ss.anchor = 0.0, 0.0, False, time.monotonic()

def reset():
    ss.update(DEFAULTS)
    ss.rate = ss.speed
    restart()

def play():
    if ss.t >= ss.duration:
        ss.t = 0.0
    ss.playing, ss.anchor = True, time.monotonic()

def pause():
    advance()
    ss.playing = False

def seek():
    ss.t, ss.playing, ss.anchor = ss.seek, False, time.monotonic()

def all_records():
    ss.t, ss.seek, ss.playing, ss.anchor = ss.duration, ss.duration, False, time.monotonic()

def speed_change():
    advance()
    ss.rate = ss.speed

def preset(name):
    ss.update(DEFAULTS)
    if name == '② 必要な項まで削る':
        ss.threshold = 0.2
    elif name == '③ ノイズと数値微分':
        ss.mode, ss.noise = '変位センサ：差分で微分', 0.01
    elif name == '④ 平滑化して微分':
        ss.mode, ss.noise = '変位センサ：平滑化して微分', 0.01
    elif name == '⑤ 二次項を発見':
        ss.quadratic, ss.q, ss.degree = True, 2.0, 2
    elif name == '⑥ 二次項を候補から外す':
        ss.quadratic, ss.q, ss.degree = True, 2.0, 1
    ss.rate = ss.speed
    restart()

with st.sidebar:
    st.header('条件設定')
    st.button('初期設定に戻す', on_click=reset, width='stretch')
    with st.expander('授業用の設定例'):
        for name in ['① 理想条件で式を発見', '② 必要な項まで削る', '③ ノイズと数値微分', '④ 平滑化して微分', '⑤ 二次項を発見', '⑥ 二次項を候補から外す']:
            st.button(name, on_click=preset, args=(name,), width='stretch')
        st.caption('設定後に「全時間を記録して停止」→「記録から式を学習する」。')
    st.subheader('物理パラメータ')
    st.slider('質量 m [kg]', .2, 10., step=.1, key='m', on_change=restart)
    st.slider('ばね定数 k [N/m]', .2, 40., step=.2, key='k', on_change=restart)
    st.slider('減衰係数 c [N·s/m]', 0., 20., step=.1, key='c', on_change=restart)
    st.checkbox('真の復元力に二次項を含める', key='quadratic', on_change=restart)
    st.slider('二次の復元力係数 q [N/m²]', -4., 4., step=.2, key='q', on_change=restart, disabled=not ss.quadratic)
    st.caption('真の復元力は kx + qx²。オン・オフで真の対象を変えます。学習側の候補次数とは別の設定です。')
    st.subheader('初期条件')
    st.slider('初期変位 x₀ [m]', -1., 1., step=.05, key='x0', on_change=restart)
    st.slider('初期速度 v₀ [m/s]', -2., 2., step=.1, key='v0', on_change=restart)
    st.subheader('センサの観測条件')
    st.select_slider('記録周期 Δt [s]', options=[.05,.1,.2,.3,.5,.8,1.], key='sensor_dt', on_change=restart)
    st.slider('変位の観測誤差 σ [m]', 0., .1, step=.001, key='noise', on_change=restart, format='%.3f')
    st.number_input('乱数シード', min_value=0, max_value=999999, step=1, key='seed', on_change=restart)
    st.subheader('SINDyの学習')
    st.selectbox('学習に用いる量', ['理想条件：正確な変位・速度・加速度', '変位センサ：差分で微分', '変位センサ：平滑化して微分'], key='mode', on_change=clear_model)
    st.select_slider('平滑化の窓幅 [点]', options=[5,7,9,11,15,21,31], key='window', on_change=clear_model, disabled='平滑化' not in ss.mode)
    st.slider('学習区間の割合', .4, .8, step=.05, key='fraction', on_change=clear_model)
    st.select_slider('候補項の最高次数', options=[1,2,3], key='degree', on_change=clear_model)
    st.slider('項を削るしきい値 λ', 0., 1.2, step=.005, key='threshold', on_change=clear_model, format='%.3f')
    st.caption('しきい値は、候補列と加速度を学習区間のRMSで正規化した係数に適用します。大きいほど項を削ります。')
    st.subheader('再生設定')
    st.slider('計算時間 [s]', 5., 60., step=5., key='duration', on_change=restart)
    st.select_slider('再生速度', options=[.25,.5,1.,2.], key='speed', on_change=speed_change, format_func=lambda x:f'{x:g} 倍')
    st.caption('物理・初期・センサ条件を変えると時刻0へ戻って停止します。')

st.title('台車の運動から方程式を探す：SINDy')
st.caption('候補項から、加速度を説明する少数の項を選びます。このデモでは運動学の関係 ẋ = v を既知とします。')
with st.expander('デモの見方・学習に使う情報'):
    st.markdown('''
真の対象の二次項のオン・オフと、学習側の候補次数は独立に設定できます。⑤で二次項を見つけ、⑥で候補を1次に制限した場合を比較してください。

1. 再生し、記録周期ごとに増える赤い×を確認します。
2. 「記録から式を学習する」を押すと、現在までの記録の前半から式を推定します。後半は学習に使いません。
3. 候補項、しきい値を変えて再学習し、残る項と、学習した式から再計算した波形を比較します。

**理想条件**ではシミュレーションの正確な変位・速度・加速度を使います。赤い×の誤差は学習に影響しません。
**変位センサ条件**では赤い×だけから変位・速度・加速度を用意します。平滑化は学習区間内の局所3次多項式で行い、端の点は除きます。オフライン処理です。

STLSQ：最小二乗で全候補の係数を推定 → 小さい係数の項を除く → 残った項で再推定、を繰り返します。
候補項の種類・単位・観測条件で結果は変わります。しきい値や平滑化は検証結果を見ながら調整するため、最終評価には別のデータも必要です。
''')
buttons = st.columns(4)
buttons[0].button('▶ 再生', on_click=play, disabled=ss.playing, width='stretch')
buttons[1].button('⏸ 一時停止', on_click=pause, disabled=not ss.playing, width='stretch')
buttons[2].button('↶ 先頭に戻る', on_click=restart, width='stretch')
buttons[3].button('全時間を記録して停止', on_click=all_records, width='stretch')
st.slider('時刻を指定 [s]（操作すると一時停止）', 0., ss.duration, step=.05, key='seek', on_change=seek)
st.caption('スライダーは時刻指定用です。現在の再生時刻は下に表示します。')
params = (ss.m,ss.k,ss.c,ss.x0,ss.v0,ss.q if ss.quadratic else 0.0,ss.duration)
limit = max(.1, 1.15*math.sqrt(ss.x0**2+ss.m/ss.k*ss.v0**2))
zeta = ss.c/(2*math.sqrt(ss.m*ss.k))
st.write(f'平衡位置付近の線形化：減衰比 ζ = {zeta:.3f}　固有周期 Tₙ = {2*math.pi*math.sqrt(ss.m/ss.k):.2f} s')
if ss.quadratic:
    st.latex(r'm\ddot{x}+c\dot{x}+kx+q x^2=0')
    st.caption('二次項のある復元力は左右非対称です。表示した周期は平衡位置付近での目安です。')
else:
    st.latex(r'm\ddot{x}+c\dot{x}+kx=0')

@st.cache_data(show_spinner=False)
def data(duration, dt, noise, seed, params):
    return records(duration,dt,noise,seed,params)

try:
    ts, truth_x, truth_v, truth_a, observed = data(ss.duration,ss.sensor_dt,ss.noise,ss.seed,params)
    if ss.quadratic:
        _,true_states = true_path(params)
        limit = max(limit,float(np.max(np.abs(true_states[:,0])))*1.15)
except ValueError as exc:
    ss.playing = False
    st.error(str(exc))
    st.stop()
plot_limit = max(limit, float(np.max(np.abs(observed)))*1.1)

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



def learn(n):
    split = int(n*ss.fraction)
    mode = 'ideal' if ss.mode.startswith('理想') else 'difference' if '差分' in ss.mode else 'smooth'
    if n-split < 3:
        raise ValueError('検証区間に3点以上必要です。記録を増やしてください。')
    td, xd, vd, ad, edge = derivative_data(ts[:n],truth_x[:n],truth_v[:n],truth_a[:n],observed[:n],split,mode,ss.window)
    result = stlsq(xd,vd,ad,ss.degree,ss.threshold)
    predicted, success = rollout(ts[:n], [ss.x0,ss.v0], result['coef'], ss.degree)
    # Reference derivatives are diagnostics only, never inputs in sensor modes.
    indices = np.arange(edge,split-edge) if edge else np.arange(split)
    ss.report = dict(result=result, times=ts[:n].copy(), truth=truth_x[:n].copy(), observed=observed[:n].copy(),
                     split=split, boundary=(ts[split-1]+ts[split])/2, td=td, ad=ad,
                     derivative_rmse=float(np.sqrt(np.mean((ad-truth_a[indices])**2))),
                     predicted=predicted, success=success, params=params, mode=ss.mode,
                     degree=ss.degree, threshold=ss.threshold, training_points=len(td))


def equation(coefficients):
    latex_terms = ['1', 'x', 'v', 'x^2', 'xv', 'v^2', 'x^3', 'x^2v', 'xv^2', 'v^3']
    terms = [rf'({a:.4g}){latex_terms[i] if i else ""}' for i,a in enumerate(coefficients) if abs(a)>1e-12]
    if not terms:
        terms = ['0']
    chunks = [' + '.join(terms[i:i+2]) for i in range(0,len(terms),2)]
    return r'\begin{aligned}\dot x &= v \\ \dot v &= ' + r' \\ &\quad + '.join(chunks) + r'\end{aligned}'


def show_report(r):
    model = r['result']
    coef = model['coef']
    support = np.flatnonzero(np.abs(coef)>1e-12)
    st.subheader('発見した方程式')
    st.write(f"**残った項：{len(support)} 個**　候補 {len(coef)} 個　λ = {r['threshold']:g}　有効な学習点 {r['training_points']} 点")
    st.caption(f"学習条件：{r['mode']}。ẋ = v は既知として固定し、v̇の式だけを探索します。係数は表示時のみ丸めています。")
    st.latex(equation(coef))
    if not len(support):
        st.info('すべての項が除かれました。この式では加速度は0です。しきい値を下げて再学習できます。')
    if model['rank'] < model['columns'] or model['condition'] > 1e8:
        st.warning('候補項の関係が強く重なり、係数を区別しにくい条件です。別の初期条件や候補項の削減でも確かめてください。')
    with st.expander('生成に用いた真の式・係数との比較'):
        m,k,c,_,_,q,_ = r['params']
        st.latex(equation(np.array([0.,-k/m,-c/m,-q/m,0.,0.])))
        st.caption('真の式は答え合わせ用です。SINDyの候補選択に係数を渡していません。発見できるのはk/m・c/m・q/mという比で、m・k・c・qの個別の特定には追加情報が必要です。')
    st.subheader('項を選ぶ過程')
    stage = st.slider('推定ステップ（0：全候補で最小二乗）', 0, max(1,len(model['history'])-1), value=0, key='stage') if len(model['history'])>1 else 0
    if len(model['history'])==1:
        st.caption('今回のしきい値では最初の推定後に除去する項がありませんでした。')
    normalized = model['history'][stage]
    fig = go.Figure(go.Bar(x=NAMES[:len(coef)],y=normalized,marker_color=['#2563eb' if abs(a)>=r['threshold'] and abs(a)>1e-12 else '#94a3b8' for a in normalized]))
    fig.add_hline(y=r['threshold'],line_dash='dot',line_color='#dc2626')
    fig.add_hline(y=-r['threshold'],line_dash='dot',line_color='#dc2626')
    fig.update_layout(height=280,margin=dict(t=15,b=30),xaxis_title='候補項',yaxis_title='正規化した係数')
    st.plotly_chart(fig,width='stretch',key='coefficients')
    st.caption('赤い点線は±λです。小さい係数の項を除き、残った項の係数を再推定します。しきい値の変更後は再学習してください。')
    physical = model['physical_history'][stage]
    st.dataframe(pd.DataFrame({'候補項':NAMES[:len(coef)],'このステップの係数（元の単位）':physical,'最終係数（元の単位）':coef}),hide_index=True)
    st.subheader('学習した式で運動を再計算')
    split = r['split']
    pred = r['predicted'][:,0]
    good = np.all(np.isfinite(pred))
    val_error = float(np.sqrt(np.mean((pred[split:]-r['truth'][split:])**2))) if good else float('inf')
    cols = st.columns(3)
    cols[0].metric('学習：加速度の当てはまり RMSE',f"{model['fit_rmse']:.4g} m/s²")
    cols[1].metric('検証：変位の連続予測 RMSE',f'{val_error:.4g} m')
    cols[2].metric('学習用加速度と真値の差 RMSE',f"{r['derivative_rmse']:.4g} m/s²")
    st.caption('加速度の当てはまりと変位の予測誤差は単位が異なります。真値との比較は仮想実験の参考値で、項の選択には使いません。')
    tt = r['times']
    dense_t = np.linspace(0,float(tt[-1]),max(500,len(tt)))
    dense_x = [state_at(float(ti),*r['params'])[0] for ti in dense_t]
    fig = go.Figure()
    fig.add_vrect(x0=0,x1=r['boundary'],fillcolor='#dbeafe',opacity=.4,line_width=0,annotation_text='学習区間',annotation_position='top left')
    fig.add_vrect(x0=r['boundary'],x1=float(tt[-1]),fillcolor='#fef3c7',opacity=.4,line_width=0,annotation_text='検証区間',annotation_position='top right')
    fig.add_trace(go.Scatter(x=dense_t,y=dense_x,mode='lines',name='真の変位',line=dict(color='#2563eb',width=2)))
    fig.add_trace(go.Scatter(x=tt,y=r['observed'],mode='markers',name='変位センサ記録',marker=dict(symbol='x',color='#dc2626',size=7)))
    fig.add_trace(go.Scatter(x=tt,y=pred,mode='lines',name='発見した式の連続計算',line=dict(color='#ea580c',dash='dash',width=3)))
    scale = max(.1,float(np.max(np.abs(r['observed']))),float(np.max(np.abs(r['truth']))))*1.2
    fig.update_layout(height=420,margin=dict(t=90,b=45),legend=dict(orientation='h',y=1.25,yanchor='bottom'),xaxis_title='時間 t [s]',yaxis_title='変位 x [m]',yaxis_range=[-scale,scale])
    st.plotly_chart(fig,width='stretch',key='discovered_trajectory')
    st.caption('初期変位・初期速度は左パネルの設定値を共通に与えています。時刻0から連続計算し、検証区間で観測値による修正は行いません。')
    if not r['success'] or np.any(np.abs(pred[np.isfinite(pred)])>scale):
        st.warning('学習した式の計算が表示範囲を超えています。数値発散時は以後の計算を打ち切り、検証誤差をinfと表示します。')
    with st.expander('数値微分で得た加速度・記録データ'):
        fig = go.Figure()
        ta = [-(r['params'][2]*state_at(float(ti),*r['params'])[1]+r['params'][1]*state_at(float(ti),*r['params'])[0]+r['params'][5]*state_at(float(ti),*r['params'])[0]**2)/r['params'][0] for ti in r['td']]
        fig.add_trace(go.Scatter(x=r['td'],y=ta,name='真の加速度（比較用）'))
        fig.add_trace(go.Scatter(x=r['td'],y=r['ad'],name='学習に用いた加速度',mode='lines+markers'))
        fig.update_layout(height=300,legend=dict(orientation='h'),xaxis_title='時間 [s]',yaxis_title='加速度 [m/s²]')
        st.plotly_chart(fig,width='stretch',key='derivatives')
        table = pd.DataFrame({'time_s':tt,'observed_m':r['observed'],'true_m':r['truth'],'identified_m':pred,'split':['train']*split+['validation']*(len(tt)-split)})
        st.download_button('記録・再計算結果をCSV保存',table.to_csv(index=False).encode('utf-8-sig'),file_name='sindy_result.csv',mime='text/csv',key='download_sindy')


@st.fragment(run_every=.1 if ss.playing else None)
def animation():
    was_playing = ss.playing
    advance()
    t = ss.t
    x,v = state_at(t,*params)
    n = int(np.searchsorted(ts,t+1e-9,side='right'))
    cols = st.columns(4)
    cols[0].metric('現在の時刻',f'{t:.2f} s')
    cols[1].metric('変位 x(t)',f'{x:+.3f} m')
    cols[2].metric('速度 v(t)',f'{v:+.3f} m/s')
    cols[3].metric('記録済みの観測点',f'{n} 点')
    left,right = st.columns([1,1.15])
    with left:
        st.subheader('台車の運動')
        st.markdown(cart_svg(x),unsafe_allow_html=True)
    with right:
        st.subheader('変位の時間変化')
        dense_t = np.linspace(0,t,max(2,math.ceil(t/.01)))
        xx = [state_at(float(ti),*params)[0] for ti in dense_t]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dense_t,y=xx,mode='lines',name='真の変位',line=dict(color='#2563eb',width=3)))
        fig.add_trace(go.Scatter(x=ts[:n],y=observed[:n],mode='markers',name='変位センサ記録',marker=dict(color='#dc2626',symbol='x',size=9,line=dict(width=2))))
        fig.add_vline(x=t,line_dash='dot',line_color='#94a3b8')
        fig.update_layout(height=350,margin=dict(l=45,r=15,t=50,b=45),legend=dict(orientation='h',x=0,y=1.02,yanchor='bottom'),uirevision='fixed',xaxis=dict(title='時間 t [s]',range=[0,ss.duration],fixedrange=True),yaxis=dict(title='変位 x [m]',range=[-plot_limit,plot_limit],fixedrange=True))
        st.plotly_chart(fig,width='stretch',key='displacement',config={'displayModeBar':False})
    st.progress(min(1.,t/ss.duration))
    if ss.mode.startswith('理想'):
        st.caption('現在は理想条件です。学習には正確な変位・速度・加速度を使います。赤い×は比較用のセンサ記録です。')
    else:
        st.caption('現在は変位センサ条件です。赤い×から変位・速度・加速度を用意して学習します。')
    st.subheader('記録から加速度の式を学習')
    st.latex(r'\dot v=\theta_0+\theta_1x+\theta_2v+\theta_3x^2+\theta_4xv+\cdots')
    st.caption('現在までの記録の前半で学習し、後半で連続計算を評価します。学習すると再生を一時停止します。')
    if st.button('記録から式を学習する',key='fit_sindy',disabled=n<15):
        try:
            learn(n)
            ss.pop('stage',None)
            ss.playing = False
            st.rerun()
        except (ValueError,np.linalg.LinAlgError) as exc:
            st.warning(str(exc))
    if n<15:
        st.caption('まず15点以上記録してください。候補数や窓幅によっては、さらに多くの記録が必要です。')
    if 'report' in ss:
        show_report(ss.report)
    if was_playing and not ss.playing:
        st.rerun()

animation()
st.caption('線形・二次ばねの自由振動を対象に、SINDyのSTLSQを体験する教材です。学習した式はRK4で数値積分します。')
