import math
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from model import at
import numpy as np
import pandas as pd
from ensemble import simulate_ensemble, values_at, event_counts, wilson, PARAMS

st.set_page_config(page_title='モンテカルロ法による水位予測',layout='wide')
st.markdown('<style>.block-container{padding-top:1.3rem}div[data-testid="stMetricValue"]{font-size:1.6rem}</style>',unsafe_allow_html=True)
DEFAULTS=dict(area=5.0,river_area=5.0,tc=.8,tr=.5,h0=.5,s0=5.0,v0=2.5,peak=40.,onset=.5,width=2.,duration=6.,threshold=1.4,speed=1.,n=200,seed=42,peak_spread=.3,onset_spread=.3,width_spread=.2,tc_spread=.3,tr_spread=.3,initial_spread=0.,member=1,draw_count=30,horizon=3.)
ss=st.session_state
for k,v in DEFAULTS.items():ss.setdefault(k,v)
for k,v in dict(t=0.,seek=0.,playing=False,anchor=time.monotonic(),rate=1.).items():ss.setdefault(k,v)


def advance():
    now=time.monotonic()
    if ss.playing:
        ss.t=min(ss.duration,ss.t+(now-ss.anchor)*ss.rate/20)
        if ss.t>=ss.duration:ss.playing=False
    ss.anchor=now

def restart():
    ss.horizon=min(ss.horizon,ss.duration)
    ss.t,ss.seek,ss.playing,ss.anchor=0.,0.,False,time.monotonic()

def reset():
    ss.update(DEFAULTS);ss.rate=ss.speed;restart()

def play():
    if ss.t>=ss.duration:ss.t=0.
    ss.playing,ss.anchor=True,time.monotonic()

def pause():
    advance();ss.playing=False

def seek():
    ss.t,ss.playing,ss.anchor=ss.seek,False,time.monotonic()

def speed_change():
    advance();ss.rate=ss.speed

def show_all():
    ss.t,ss.seek,ss.playing,ss.anchor=ss.duration,ss.duration,False,time.monotonic()

def preset(name):
    ss.update(DEFAULTS)
    if name=='① ばらつきなし':
        for key in ['peak_spread','onset_spread','width_spread','tc_spread','tr_spread','initial_spread']:ss[key]=0.
    elif name=='② 降雨だけにばらつき':
        ss.tc_spread=ss.tr_spread=0.
    elif name=='④ 標本数を増やす':ss.n=1000
    ss.rate=ss.speed;restart()

with st.sidebar:
    st.header('条件設定')
    st.button('初期設定に戻す',on_click=reset,width='stretch')
    with st.expander('授業用の設定例'):
        for name in ['① ばらつきなし','② 降雨だけにばらつき','③ 降雨と時定数にばらつき','④ 標本数を増やす']:
            st.button(name,on_click=preset,args=(name,),width='stretch')
    st.subheader('流域・河道のパラメータ')
    st.slider('流域面積 A [km²]',.5,20.,step=.5,key='area',on_change=restart)
    st.slider('河道の有効水面積 Aᵣ [万m²]',1.,30.,step=1.,key='river_area',on_change=restart)
    st.slider('流域の時定数 τc [h]',.1,4.,step=.1,key='tc',on_change=restart)
    st.slider('河道の時定数 τr [h]',.1,4.,step=.1,key='tr',on_change=restart)
    st.slider('基準となる水位 h₀ [m]',0.,3.,step=.1,key='h0',on_change=restart)
    st.subheader('初期貯留')
    st.slider('流域の初期貯留 S₀（等価水深）[mm]',0.,30.,step=1.,key='s0',on_change=restart)
    st.slider('河道の初期貯留 V₀ [万m³]',0.,20.,step=.5,key='v0',on_change=restart)
    #st.caption('S₀は等価水深×流域面積で水量へ換算。h₀は水位の基準で、初期水位はh₀+V₀/Aᵣです。')
    st.subheader('有効降雨')
    st.slider('ピークの有効降雨強度 [mm/h]',0.,100.,step=5.,key='peak',on_change=restart)
    st.slider('降雨開始時刻 [h]',0.,6.,step=.25,key='onset',on_change=restart)
    st.slider('降雨継続時間 [h]',.5,6.,step=.25,key='width',on_change=restart)
    #st.caption('降雨開始から継続時間の半分でピークに達し、その後0へ戻ります。流出に寄与する有効降雨を直接与えます。')
    st.subheader('モンテカルロ法の設定')
    st.select_slider('サンプル数 N',options=[50,100,200,500,1000],key='n',on_change=restart)
    st.number_input('乱数シード',min_value=0,max_value=999999,step=1,key='seed',on_change=restart)
    st.slider('降雨ピークのばらつき ±割合',0.,.8,step=.05,key='peak_spread',on_change=restart)
    st.slider('降雨開始時刻のずれ ±時間 [h]',0.,1.,step=.1,key='onset_spread',on_change=restart)
    st.slider('降雨継続時間のばらつき ±割合',0.,.8,step=.05,key='width_spread',on_change=restart)
    st.slider('流域の時定数 τc のばらつき ±割合',0.,.8,step=.05,key='tc_spread',on_change=restart)
    st.slider('河道の時定数 τr のばらつき ±割合',0.,.8,step=.05,key='tr_spread',on_change=restart)
    st.slider('初期貯留 S₀・V₀ のばらつき ±割合',0.,.8,step=.05,key='initial_spread',on_change=restart)
    st.caption('中心値の±指定割合の一様分布。0.30なら中心値の70〜130%。開始時刻は±指定時間。各変数は独立に抽出します。')
    if ss.member>ss.n:ss.member=1
    st.number_input('模式図に表示するシナリオ番号',min_value=1,max_value=ss.n,step=1,key='member')
    st.select_slider('細線で描くサンプル数',options=[10,30,50,100],key='draw_count')
    st.subheader('表示と再生')
    st.slider('表示する基準水位 [m]',.5,10.,step=.1,key='threshold')
    st.slider('計算時間 [h]',3.,12.,step=1.,key='duration',on_change=restart)
    st.slider('期間超過確率の評価終了時刻 [h]',.5,ss.duration,step=.5,key='horizon')
    st.select_slider('再生速度',options=[.25,.5,1.,2.,4.],key='speed',on_change=speed_change,format_func=lambda x:f'{x:g} 倍')
    #st.caption('1倍ではモデルの1時間を約20秒で再生。表示基準以外のモデル条件を変えると時刻0へ戻ります。')

st.title('モンテカルロ法による水位予測')
st.caption('降雨・初期貯留・時定数をサンプルごとに変え、複数の水位の未来を同時に計算します。再生は計算済みの時間軌跡の表示で、観測による予測更新は行いません。')
#st.caption('流域と河道をそれぞれ一つの貯留として表す、講義用の確定的なモデルです。')
buttons=st.columns(4)
buttons[0].button('▶ 再生',on_click=play,disabled=ss.playing,width='stretch')
buttons[1].button('⏸ 一時停止',on_click=pause,disabled=not ss.playing,width='stretch')
buttons[2].button('↶ 先頭に戻る',on_click=restart,width='stretch')
buttons[3].button('全時間を表示',on_click=show_all,width='stretch')
st.slider('時刻を指定 [h]（操作すると一時停止）',0.,ss.duration,step=.05,key='seek',on_change=seek)

BASE_KEYS=['area','river_area','tc','tr','h0','s0','v0','peak','onset','width','duration']
base={key:ss[key] for key in BASE_KEYS}
spreads=dict(peak=ss.peak_spread,onset=ss.onset_spread,width=ss.width_spread,tc=ss.tc_spread,tr=ss.tr_spread,s0=ss.initial_spread,v0=ss.initial_spread)

@st.cache_data(show_spinner=False,max_entries=4)
def calculate(base,spreads,n,seed):
    return simulate_ensemble(base,spreads,n,seed)

with st.spinner('複数の水位軌跡を計算しています…'):
    ensemble=calculate(base,spreads,ss.n,ss.seed)
times=ensemble['t'];heights=ensemble['h']
selected=ss.member-1
sample={key:float(ensemble['draws'][key][selected]) for key in PARAMS}
volumes=(heights[selected]-ss.h0)*ss.river_area*10000
z=(times-sample['onset'])/sample['width']
data=dict(t=times,s=ensemble['s'][selected],v=volumes,h=heights[selected],rain=sample['peak']*np.maximum(0,1-np.abs(2*z-1)),qin=ensemble['s'][selected]/(sample['tc']*3600),qout=volumes/(sample['tr']*3600))
area=ss.area*1e6
smax=max(1.,float(ensemble['s'].max())/area*1000*1.15)
hmax=max(ss.threshold*1.15,float(heights.max())*1.15,ss.h0+.5)
flowmax=max(1.,float(data['qin'].max()),float(data['qout'].max()),ss.area*sample['peak']/3.6)
low,median,high=np.quantile(heights,[.05,.5,.95],axis=0)
mean=heights.mean(axis=0)
instant_prob=(heights>ss.threshold).mean(axis=0)
cumulative_prob=(np.maximum.accumulate(heights,axis=1)>ss.threshold).mean(axis=0)
window_count,maxima=event_counts(heights,times,ss.threshold,ss.horizon)
ci=wilson(window_count,ss.n)
st.info(f'予測開始から {ss.horizon:g} 時間までに一度でも基準を超える割合：{window_count/ss.n:.1%}（{window_count}/{ss.n} 本）')
st.caption(f'MC標本誤差の95%区間（Wilson）：{ci[0]:.1%}〜{ci[1]:.1%}。仮定した入力分布・モデルの誤りは含みません。超過は h > {ss.threshold:g} m として計算します。')


def diagram(state):
    depth=state['s']/area*1000
    sy=300-155*depth/smax
    hy=300-155*state['h']/hmax
    by=300-155*ss.threshold/hmax
    base=300-155*ss.h0/hmax
    widths=[(2+13*q/flowmax) if q>1e-9 else 0 for q in [ss.area*state['rain']/3.6,state['qin'],state['qout']]]
    def arrow(x1,x2,y,width):
        if width == 0:
            return f'<line x1="{x1}" x2="{x2}" y1="{y}" y2="{y}" stroke="#cbd5e1" stroke-dasharray="4 4"/>'
        return f'<line x1="{x1}" x2="{x2-12}" y1="{y}" y2="{y}" stroke="#1684cd" stroke-width="{width:.1f}"/><path d="M{x2-15},{y-11} L{x2},{y} L{x2-15},{y+11}Z" fill="#1684cd"/>'
    return f'''<svg viewBox="0 0 850 405" width="100%" role="img" aria-label="流域と河道の貯留。水位 {state['h']:.2f} メートル">
    <rect x="1" y="1" width="848" height="403" rx="12" fill="#f8fafc"/>
    <text x="260" y="40" text-anchor="middle" font-size="23" fill="#166534">流域の貯留 S</text>
    <text x="590" y="40" text-anchor="middle" font-size="23" fill="#6b21a8">河道の貯留 V</text>
    <rect x="150" y="{sy:.2f}" width="220" height="{300-sy:.2f}" fill="#7dd3fc"/>
    <path d="M150 130 V300 H370 V130" fill="none" stroke="#166534" stroke-width="5"/>
    <line x1="150" x2="370" y1="{sy:.2f}" y2="{sy:.2f}" stroke="#0284c7" stroke-width="3"/>
    <rect x="480" y="{hy:.2f}" width="220" height="{300-hy:.2f}" fill="#38bdf8"/>
    <rect x="480" y="{base:.2f}" width="220" height="{300-base:.2f}" fill="#e2e8f0"/>
    <path d="M480 130 V300 H700 V130" fill="none" stroke="#6b21a8" stroke-width="5"/>
    <line x1="480" x2="700" y1="{hy:.2f}" y2="{hy:.2f}" stroke="#0284c7" stroke-width="3"/>
    <line x1="470" x2="715" y1="{by:.2f}" y2="{by:.2f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="7 5"/>
    <text x="590" y="112" text-anchor="middle" font-size="16" fill="#b91c1c">基準水位 {ss.threshold:.1f} m（赤破線）</text>
    <line x1="480" x2="700" y1="{base:.2f}" y2="{base:.2f}" stroke="#64748b" stroke-width="1" stroke-dasharray="3 3"/>
    <text x="60" y="155" text-anchor="middle" font-size="19" fill="#0c4a6e">有効降雨</text>
    <text x="60" y="182" text-anchor="middle" font-size="18" fill="#0c4a6e">{state['rain']:.1f} mm/h</text>
    {arrow(30,135,225,widths[0])}{arrow(380,470,225,widths[1])}{arrow(710,825,225,widths[2])}
    <text x="425" y="180" text-anchor="middle" font-size="19" fill="#0c4a6e">Qᵢₙ</text>
    <text x="425" y="205" text-anchor="middle" font-size="17" fill="#0c4a6e">{state['qin']:.1f} m³/s</text>
    <text x="765" y="180" text-anchor="middle" font-size="19" fill="#0c4a6e">Qₒᵤₜ</text>
    <text x="765" y="205" text-anchor="middle" font-size="17" fill="#0c4a6e">{state['qout']:.1f} m³/s</text>
    <text x="260" y="335" text-anchor="middle" font-size="18" fill="#166534">S = {state['s']/10000:.2f} 万m³</text>
    <text x="260" y="362" text-anchor="middle" font-size="17" fill="#475569">等価水深 {depth:.2f} mm</text>
    <text x="590" y="335" text-anchor="middle" font-size="18" fill="#6b21a8">V = {state['v']/10000:.2f} 万m³</text>
    <text x="590" y="362" text-anchor="middle" font-size="19" fill="#075985">h = {state['h']:.2f} m</text>
    <text x="765" y="265" text-anchor="middle" font-size="18" fill="#475569">下流へ</text>
    </svg>'''


@st.fragment(run_every=.15 if ss.playing else None)
def animation():
    was_playing=ss.playing
    advance()
    state=at(data,ss.t)
    now_h,idx=values_at(heights,times,ss.t)
    past_count,_=event_counts(heights,times,ss.threshold,ss.t)
    metrics=st.columns(4)
    metrics[0].metric('再生時刻',f'{ss.t:.2f} h')
    metrics[1].metric('この時刻の平均水位',f'{now_h.mean():.2f} m')
    metrics[2].metric('この時刻で超過する割合',f'{np.mean(now_h>ss.threshold):.1%}')
    metrics[3].metric('再生時刻までに一度でも超過',f'{past_count/ss.n:.1%}')
    left,right=st.columns([1.1,1.])
    with left:
        st.subheader(f'シナリオ {ss.member} の貯留')
        st.markdown(diagram(state),unsafe_allow_html=True)
        st.caption(f'模式図は平均ではなく選択した1本です。橙色の線と対応します。現在の水位 {state["h"]:.2f} m。')
        st.caption(f'このシナリオの降雨ピーク {sample["peak"]:.1f} mm/h、開始 {sample["onset"]:.2f} h、継続 {sample["width"]:.2f} h、τc {sample["tc"]:.2f} h、τr {sample["tr"]:.2f} h。')
    n=idx+1
    tt=np.r_[times[:n],ss.t]
    def until(values):return np.r_[values[:n],np.interp(ss.t,times,values)]
    with right:
        st.subheader('複数の水位予測')
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=tt,y=until(low),mode='lines',line=dict(width=0),showlegend=False,hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=tt,y=until(high),mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba(37,99,235,0.17)',name='各時刻の5–95%帯'))
        drawn=min(ss.draw_count,ss.n)
        # All N samples enter probabilities, even when only a subset is rendered.
        for i in range(drawn):
            fig.add_trace(go.Scatter(x=tt,y=np.r_[heights[i,:n],now_h[i]],mode='lines',line=dict(color='rgba(59,130,246,.20)',width=1),showlegend=False,hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=tt,y=until(mean),name='平均水位',line=dict(color='#1d4ed8',width=3)))
        fig.add_trace(go.Scatter(x=tt,y=np.r_[heights[selected,:n],now_h[selected]],name=f'シナリオ {ss.member}',line=dict(color='#ea580c',width=2)))
        fig.add_hline(y=ss.threshold,line_color='#dc2626',line_dash='dash')
        fig.add_vline(x=ss.t,line_color='#94a3b8',line_dash='dot')
        fig.update_layout(height=370,margin=dict(t=65,l=45,r=15,b=45),legend=dict(orientation='h',y=1.04,yanchor='bottom',font=dict(size=11)),uirevision='mc',xaxis=dict(title='予測開始からの時間 [h]',range=[0,ss.duration],fixedrange=True),yaxis=dict(title='水位 [m]',range=[0,hmax],fixedrange=True))
        st.plotly_chart(fig,width='stretch',key='water_mc',config={'displayModeBar':False})
    st.progress(min(1.,ss.t/ss.duration))
    st.caption(f'細線は {drawn} 本、確率の計算には全 {ss.n} 本を使用。帯は各時刻の分位点で、軌跡全体の90%を覆う保証や平均値の信頼区間ではありません。')
    st.subheader('基準水位の超過確率')
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=times[:n],y=instant_prob[:n]*100,name='各時刻で超過',line=dict(color='#2563eb',width=3)))
    fig.add_trace(go.Scatter(x=times[:n],y=cumulative_prob[:n]*100,name='その時刻までに一度でも超過',line=dict(color='#dc2626',width=3)))
    fig.add_vline(x=ss.horizon,line_color='#64748b',line_dash='dot',annotation_text='評価終了時刻')
    fig.update_layout(height=260,margin=dict(t=55,b=40),legend=dict(orientation='h',y=1.05,yanchor='bottom'),xaxis=dict(title='予測開始からの時間 [h]',range=[0,ss.duration]),yaxis=dict(title='超過する割合 [%]',range=[0,105]))
    st.plotly_chart(fig,width='stretch',key='exceedance')
    st.caption('赤線は一度超えたシナリオをその後も数えるので下がりません。「期間中に一度でも超える確率」は、各時刻の超過確率の最大値とは異なります。再生に伴い新しい観測を条件とした予測へ更新しているわけではありません。')
    with st.expander('有効降雨・流量・貯留量の推移（選択シナリオ）'):
        fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.09,subplot_titles=['有効降雨','河道への流入と下流への流出','流域と河道の貯留水量'])
        for row,keys,names,colors in [(1,['rain'],['有効降雨'],['#0284c7']),(2,['qin','qout'],['Qᵢₙ','Qₒᵤₜ'],['#16a34a','#9333ea']),(3,['s','v'],['流域 S','河道 V'],['#16a34a','#9333ea'])]:
            for key,name,color in zip(keys,names,colors):
                vals=until(data[key])/(10000 if row==3 else 1)
                fig.add_trace(go.Scatter(x=tt,y=vals,name=name,line=dict(color=color)),row=row,col=1)
        for row,unit in [(1,'mm/h'),(2,'m³/s'),(3,'万m³')]:fig.update_yaxes(title_text=unit,row=row,col=1)
        fig.update_xaxes(range=[0,ss.duration]);fig.update_xaxes(title_text='時間 [h]',row=3,col=1)
        fig.update_layout(height=620,margin=dict(t=60),legend=dict(orientation='h',y=1.08))
        st.plotly_chart(fig,width='stretch',key='diagnostics')
    with st.expander('抽出したパラメータ・期間内最高水位・データ保存'):
        params=pd.DataFrame(ensemble['draws'])
        params.insert(0,'scenario',np.arange(1,ss.n+1))
        params['max_h_in_window_m']=maxima
        params['exceeded']=maxima>ss.threshold
        st.dataframe(params,hide_index=True)
        st.caption('peak:mm/h、onset/width/tc/tr:h、s0:等価水深mm、v0:万m³。max_h_in_window_mは予測開始から評価終了時刻までの最大水位です。')
        st.download_button('シナリオと期間超過の結果をCSV保存',params.to_csv(index=False).encode('utf-8-sig'),file_name='mc_scenarios.csv',mime='text/csv',key='download_samples')
        summary=pd.DataFrame(dict(time_h=times,mean_h_m=mean,median_h_m=median,q05_h_m=low,q95_h_m=high,instant_exceedance=instant_prob,cumulative_exceedance=cumulative_prob))
        st.download_button('水位分布・超過確率の時系列をCSV保存',summary.to_csv(index=False).encode('utf-8-sig'),file_name='mc_summary.csv',mime='text/csv',key='download_summary')
        st.caption('CSVには計算期間全体の結果を保存します。水収支残差の最大値：'+f'{ensemble["max_balance"]:.2e} m³')
    with st.expander('講義資料の式・単位・見方'):
        a,b=st.columns(2)
        a.latex(r'\frac{dS}{dt}=Ar_e-Q_{\mathrm{in}},\quad Q_{\mathrm{in}}=\frac{S}{\tau_c}')
        b.latex(r'\frac{dV}{dt}=Q_{\mathrm{in}}-Q_{\mathrm{out}},\quad Q_{\mathrm{out}}=\frac{V}{\tau_r}')
        st.latex(r'h=h_0+\frac{V}{A_r}')
        st.latex(r'\widehat p_T=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\!\left\{\max_{0\le t\le T}h^{(i)}(t)>h_{\mathrm{crit}}\right\}')
        st.markdown('''S,Vはm³、流量はm³/s。内部では秒で計算します。各サンプルでパラメータを一度抽出し、時刻を通じて一貫した三角形の有効降雨と同じパラメータを使います。

ばらつきは中心値±指定幅の独立な一様分布です。幅は標準偏差ではありません。開始時刻が負なら予測開始前から雨が降っているシナリオです。初期貯留は別に指定した分布から与え、過去の雨から逆算しません。

描画する本数と計算する本数は別です。同じシードなら再生・再描画でサンプルは変わりません。Nを増やしたときも先頭のサンプルは同じです。

超過判定は30秒刻みの数値軌跡に基づきます。標本誤差と時間離散化の誤差があり、0%や100%も真の確率の断定ではありません。分布の仮定、変数間の独立性、モデルの妥当性は別途検証が必要です。

灰色点線はh₀。水色の貯留Vはh₀からの水位増分に対応します。潮位・背水・氾濫は対象外です。''')
    if was_playing and not ss.playing:st.rerun()

animation()
