import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ensemble import simulate_two,selected_data,at,probability

st.set_page_config(page_title='2流域の降雨相関と合流後の水位',layout='wide')
st.markdown('<style>.block-container{padding-top:1.3rem}div[data-testid="stMetricValue"]{font-size:1.55rem}</style>',unsafe_allow_html=True)
DEFAULTS=dict(area1=2.5,area2=2.5,river_area=5.,tc1=.8,tc2=.8,tr=.5,h0=.5,s0=5.,v0=2.5,peak=40.,spread=.5,onset=.5,width=2.,duration=6.,threshold=1.7,horizon=3.,rho=0.,n=200,seed=42,member=1,draw_count=30,speed=1.)
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
def show_all():
    ss.t,ss.seek,ss.playing,ss.anchor=ss.duration,ss.duration,False,time.monotonic()
def speed_change():
    advance();ss.rate=ss.speed
def change_rho():
    # Keep the displayed time so correlation comparisons can be made in place.
    advance();ss.playing=False

def preset(rho):
    ss.rho=rho;change_rho()

with st.sidebar:
    st.header('条件設定')
    st.button('初期設定に戻す',on_click=reset,width='stretch')
    st.subheader('2流域の降雨ピークの相関')
    st.slider('相関の目標値 ρ',-1.,1.,step=.1,key='rho',on_change=change_rho)
    presets=st.columns(3)
    for col,label,value in zip(presets,['負 −1','無相関 0','正 +1'],[-1.,0.,1.]):
        col.button(label,on_click=preset,args=(value,),width='stretch')
    st.caption('降雨強度の標本は固定し、流域1と流域2の組合せだけを変更。実際の標本相関は画面に表示します。')
    st.subheader('共通の降雨強度分布')
    st.slider('ピーク強度の中心 [mm/h]',0.,100.,step=5.,key='peak',on_change=restart)
    st.slider('ピーク強度のばらつき ±割合',0.,.8,step=.05,key='spread',on_change=restart)
    st.caption(f'両流域とも {ss.peak*(1-ss.spread):.1f}〜{ss.peak*(1+ss.spread):.1f} mm/h の一様分布。')
    st.select_slider('サンプル数 N',options=[50,100,200,500,1000],key='n',on_change=restart)
    st.number_input('乱数シード',min_value=0,max_value=999999,step=1,key='seed',on_change=restart)
    with st.expander('降雨時刻・流域・河道の条件'):
        st.slider('共通の降雨開始時刻 [h]',0.,6.,step=.25,key='onset',on_change=restart)
        st.slider('共通の降雨継続時間 [h]',.5,6.,step=.25,key='width',on_change=restart)
        st.slider('流域1の面積 A₁ [km²]',.5,10.,step=.5,key='area1',on_change=restart)
        st.slider('流域2の面積 A₂ [km²]',.5,10.,step=.5,key='area2',on_change=restart)
        st.slider('流域1の時定数 τ₁ [h]',.1,4.,step=.1,key='tc1',on_change=restart)
        st.slider('流域2の時定数 τ₂ [h]',.1,4.,step=.1,key='tc2',on_change=restart)
        st.slider('河道の時定数 τr [h]',.1,4.,step=.1,key='tr',on_change=restart)
        st.slider('有効水面積 Aᵣ [万m²]',1.,30.,step=1.,key='river_area',on_change=restart)
        st.slider('水位の換算基準 h₀ [m]',0.,3.,step=.1,key='h0',on_change=restart)
        st.slider('各流域の初期貯留の等価水深 [mm]',0.,30.,step=1.,key='s0',on_change=restart)
        st.slider('河道の初期貯留 V₀ [万m³]',0.,20.,step=.5,key='v0',on_change=restart)
    st.subheader('評価と表示')
    st.slider('基準水位 [m]',.5,10.,step=.1,key='threshold')
    st.slider('計算時間 [h]',3.,12.,step=1.,key='duration',on_change=restart)
    st.slider('評価終了時刻 T [h]',.5,ss.duration,step=.5,key='horizon')
    if ss.member>ss.n:ss.member=1
    st.number_input('模式図のシナリオ番号',min_value=1,max_value=ss.n,step=1,key='member')
    st.select_slider('細線で描く本数',options=[10,30,50,100],key='draw_count')
    st.select_slider('再生速度',options=[.25,.5,1.,2.,4.],key='speed',on_change=speed_change,format_func=lambda x:f'{x:g} 倍')

st.title('2流域の降雨相関と合流後の水位')
st.caption('それぞれの降雨強度の分布を変えずに、2流域で強い雨が重なる関係だけを変えます。降雨時刻・継続時間・物理パラメータはシナリオ間で固定です。')
buttons=st.columns(4)
buttons[0].button('▶ 再生',on_click=play,disabled=ss.playing,width='stretch')
buttons[1].button('⏸ 一時停止',on_click=pause,disabled=not ss.playing,width='stretch')
buttons[2].button('↶ 先頭に戻る',on_click=restart,width='stretch')
buttons[3].button('全時間を表示',on_click=show_all,width='stretch')
st.slider('時刻を指定 [h]（操作すると一時停止）',0.,ss.duration,step=.05,key='seek',on_change=seek)
BASE_KEYS=['area1','area2','river_area','tc1','tc2','tr','h0','s0','v0','peak','spread','onset','width','duration']
base={key:ss[key] for key in BASE_KEYS}
@st.cache_data(show_spinner=False,max_entries=6)
def calculate(base,n,seed,rho):return simulate_two(base,n,seed,rho)
with st.spinner('2流域から合流する水位軌跡を計算しています…'):
    result=calculate(base,ss.n,ss.seed,ss.rho)
times=result['t'];heights=result['h']
data=selected_data(result,ss.member-1,base)
lo,hi=np.quantile(heights,[.05,.95],axis=0);mean=heights.mean(axis=0)
p_window,maxima=probability(heights,times,ss.threshold,ss.horizon)
actual='未定義（強度のばらつき0）' if result['corr'] is None else f"{result['corr']:+.4f}"
st.info(f'0〜{ss.horizon:g}時間に一度でも基準水位 {ss.threshold:g} m を超える割合：{p_window:.1%}（{round(p_window*ss.n)}/{ss.n} 本）　実際の降雨ピークの標本相関：{actual}')
st.caption('相関を変えても再生時刻を保持します。各流域の標本平均・標本分布は完全に同じです。無相関は独立性の保証を意味しません。')
# Use common bounds for every correlation so visual scaling cannot mimic an effect.
scale_result=calculate(base,ss.n,ss.seed,1.)
hmax=max(ss.threshold*1.15,float(scale_result['h'].max())*1.15,ss.h0+.5)
smax1=max(1.,float(np.max(np.asarray(result['profiles'][0][0]['s'])+result['p1'].max()*result['profiles'][0][1]['s']))*1.1)
smax2=max(1.,float(np.max(np.asarray(result['profiles'][1][0]['s'])+result['p2'].max()*result['profiles'][1][1]['s']))*1.1)


def diagram(state):
    # Two tributaries merge into one river reservoir.
    def tank(x,y,w,h,depth,maximum,color):
        level=y+h-h*.85*max(0.,depth)/maximum
        return f'<rect x="{x}" y="{level:.2f}" width="{w}" height="{y+h-level:.2f}" fill="#7dd3fc"/><path d="M{x} {y} V{y+h} H{x+w} V{y}" fill="none" stroke="{color}" stroke-width="4"/>'
    a=tank(145,75,190,90,state['s1'],smax1,'#15803d')
    b=tank(145,245,190,90,state['s2'],smax2,'#7c3aed')
    vheight=max(.1,hmax-ss.h0)
    c=tank(555,140,185,145,state['h']-ss.h0,vheight,'#075985')
    threshold_y=285-145*.85*(ss.threshold-ss.h0)/vheight
    return f'''<svg viewBox="0 0 850 410" width="100%" role="img" aria-label="2流域から合流後の水位 {state['h']:.2f} メートル">
    <rect width="850" height="410" rx="12" fill="#f8fafc"/>{a}{b}{c}
    <text x="240" y="45" text-anchor="middle" font-size="23" fill="#166534">流域1 S₁</text>
    <text x="240" y="215" text-anchor="middle" font-size="23" fill="#6b21a8">流域2 S₂</text>
    <text x="647" y="100" text-anchor="middle" font-size="23" fill="#075985">合流後の河道 V</text>
    <text x="60" y="110" text-anchor="middle" font-size="18" fill="#166534">雨 r₁</text>
    <text x="60" y="139" text-anchor="middle" font-size="17">{state['rain1']:.1f} mm/h</text>
    <text x="60" y="280" text-anchor="middle" font-size="18" fill="#6b21a8">雨 r₂</text>
    <text x="60" y="309" text-anchor="middle" font-size="17">{state['rain2']:.1f} mm/h</text>
    <path d="M335 125 H430 V210 H550 M335 295 H430 V210" fill="none" stroke="#1684cd" stroke-width="6"/>
    <path d="M535 198 L555 210 L535 222Z" fill="#1684cd"/>
    <path d="M740 220 H815 M802 208 L822 220 L802 232Z" fill="none" stroke="#1684cd" stroke-width="5"/>
    <text x="400" y="95" text-anchor="middle" font-size="17">Q₁={state['qin1']:.1f}</text>
    <text x="400" y="330" text-anchor="middle" font-size="17">Q₂={state['qin2']:.1f}</text>
    <text x="490" y="190" text-anchor="middle" font-size="18">Q₁+Q₂</text>
    <text x="785" y="265" text-anchor="middle" font-size="17">下流へ</text>
    <text x="645" y="330" text-anchor="middle" font-size="24" fill="#075985">h={state['h']:.2f} m</text>
    <line x1="550" x2="745" y1="{threshold_y:.2f}" y2="{threshold_y:.2f}" stroke="#dc2626" stroke-dasharray="6 5" stroke-width="2"/>
    <text x="425" y="389" text-anchor="middle" font-size="16" fill="#64748b">選択シナリオの模式図。流量の単位はm³/s。水面の倍率は再生中固定。</text>
    </svg>'''


def marginal_plots():
    st.subheader('分布は同じ、組合せが違う')
    left,right=st.columns(2)
    bound=[ss.peak*(1-ss.spread),ss.peak*(1+ss.spread)]
    if bound[0]==bound[1]:bound=[ss.peak-1,ss.peak+1]
    with left:
        fig=go.Figure(go.Scatter(x=result['p1'],y=result['p2'],mode='markers',marker=dict(color='#2563eb',size=5,opacity=.5),name='降雨ピークの組'))
        fig.update_layout(title='同じシナリオの降雨ピークの組合せ',height=310,margin=dict(t=45,b=45),xaxis=dict(title='流域1 [mm/h]',range=bound),yaxis=dict(title='流域2 [mm/h]',range=bound))
        st.plotly_chart(fig,width='stretch',key='rain_pairs')
    with right:
        edges=np.linspace(bound[0],bound[1],11)
        counts1,_=np.histogram(result['p1'],edges);counts2,_=np.histogram(result['p2'],edges)
        centers=(edges[:-1]+edges[1:])/2
        fig=go.Figure()
        fig.add_trace(go.Bar(x=centers,y=counts1,name='流域1',marker_color='#16a34a'))
        fig.add_trace(go.Bar(x=centers,y=counts2,name='流域2',marker_color='#9333ea'))
        fig.update_layout(title='各流域の強度分布（標本数も同じ）',barmode='group',height=310,margin=dict(t=45,b=45),legend=dict(orientation='h'),xaxis_title='降雨ピーク [mm/h]',yaxis_title='本数')
        st.plotly_chart(fig,width='stretch',key='rain_marginals')
    st.caption(f'両流域の標本平均はともに {result["p1"].mean():.2f} mm/h。相関を変えても並べ替え前の標本値とヒストグラムは変わりません。')


@st.fragment(run_every=.15 if ss.playing else None)
def animation():
    was_playing=ss.playing;advance()
    state=at(data,ss.t)
    i=int(np.searchsorted(times,ss.t,side='right'))
    i=max(1,min(len(times),i))
    tt=np.r_[times[:i],ss.t]
    def until(v):return np.r_[v[:i],np.interp(ss.t,times,v)]
    now=np.array([np.interp(ss.t,times,row) for row in heights])
    cols=st.columns(4)
    cols[0].metric('再生時刻',f'{ss.t:.2f} h')
    cols[1].metric('平均水位',f'{now.mean():.2f} m')
    cols[2].metric('この時刻の水位の標準偏差',f'{now.std():.3f} m')
    cols[3].metric('この時刻で超過する割合',f'{np.mean(now>ss.threshold):.1%}')
    left,right=st.columns([1.1,1.])
    with left:
        st.subheader(f'シナリオ {ss.member} の合流')
        st.markdown(diagram(state),unsafe_allow_html=True)
        st.caption(f'降雨ピーク：流域1 {result["p1"][ss.member-1]:.1f}、流域2 {result["p2"][ss.member-1]:.1f} mm/h。橙色の軌跡と対応します。')
    with right:
        st.subheader('合流後の水位予測')
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=tt,y=until(lo),line=dict(width=0),showlegend=False,hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=tt,y=until(hi),line=dict(width=0),fill='tonexty',fillcolor='rgba(37,99,235,.18)',name='各時刻の5–95%帯'))
        for row in heights[:min(ss.draw_count,ss.n)]:
            fig.add_trace(go.Scatter(x=tt,y=until(row),line=dict(color='rgba(37,99,235,.20)',width=1),showlegend=False,hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=tt,y=until(mean),name='平均水位',line=dict(color='#1d4ed8',width=3)))
        fig.add_trace(go.Scatter(x=tt,y=until(heights[ss.member-1]),name=f'シナリオ {ss.member}',line=dict(color='#ea580c',width=2)))
        fig.add_hline(y=ss.threshold,line_color='#dc2626',line_dash='dash')
        fig.add_vline(x=ss.t,line_color='#94a3b8',line_dash='dot')
        fig.update_layout(height=355,margin=dict(t=65,l=45,r=15,b=40),legend=dict(orientation='h',y=1.03,yanchor='bottom',font=dict(size=11)),uirevision='two',xaxis=dict(title='予測開始からの時間 [h]',range=[0,ss.duration],fixedrange=True),yaxis=dict(title='水位 [m]',range=[0,hmax],fixedrange=True))
        st.plotly_chart(fig,width='stretch',key='water_paths',config={'displayModeBar':False})
    st.progress(ss.t/ss.duration)
    st.caption(f'確率は全 {ss.n} 本で計算。5–95%帯は各時刻の分位点で、軌跡全体の同時被覆区間ではありません。再生は計算済みの軌跡を順に表示する操作です。')
    st.subheader('基準水位の超過確率')
    instantaneous=(heights>ss.threshold).mean(axis=0)
    cumulative=(np.maximum.accumulate(heights,axis=1)>ss.threshold).mean(axis=0)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=times[:i],y=instantaneous[:i]*100,name='各時刻で超過',line=dict(color='#2563eb',width=3)))
    fig.add_trace(go.Scatter(x=times[:i],y=cumulative[:i]*100,name='その時刻までに一度でも超過',line=dict(color='#dc2626',width=3)))
    fig.add_vline(x=ss.horizon,line_dash='dot',line_color='#64748b')
    fig.update_layout(height=250,margin=dict(t=50,b=40),legend=dict(orientation='h',y=1.03,yanchor='bottom'),xaxis=dict(title='時間 [h]',range=[0,ss.duration]),yaxis=dict(title='超過する割合 [%]',range=[0,105]))
    st.plotly_chart(fig,width='stretch',key='exceedance')
    marginal_plots()
    with st.expander('相関だけを変えた3条件の比較'):
        rows=[]
        for rho in [-1.,0.,1.]:
            r=calculate(base,ss.n,ss.seed,rho)
            prob,mx=probability(r['h'],r['t'],ss.threshold,ss.horizon)
            rows.append({'相関目標':rho,'標本相関':r['corr'],'期間超過率':prob,'期間最高水位の平均 [m]':mx.mean(),'期間最高水位の標準偏差 [m]':mx.std()})
        st.dataframe(pd.DataFrame(rows),hide_index=True,column_config={'期間超過率':st.column_config.NumberColumn(format='%.3f')})
        st.caption('期間超過率は0〜1の割合です。正の相関で常に超過率が上がるわけではなく、基準水位の位置や流域条件によって変わります。')
    with st.expander('データ保存'):
        frame=pd.DataFrame({'scenario':np.arange(1,ss.n+1),'peak1_mm_h':result['p1'],'peak2_mm_h':result['p2'],'max_h_in_window_m':maxima,'exceeded':maxima>ss.threshold})
        st.download_button('降雨の組合せ・期間超過結果をCSV保存',frame.to_csv(index=False).encode('utf-8-sig'),file_name='two_catchments_samples.csv',mime='text/csv',key='download_samples')
        summary=pd.DataFrame({'time_h':times,'mean_h_m':mean,'q05_h_m':lo,'q95_h_m':hi,'instant_exceedance':instantaneous,'cumulative_exceedance':cumulative})
        st.download_button('水位分布・超過率の時系列をCSV保存',summary.to_csv(index=False).encode('utf-8-sig'),file_name='two_catchments_summary.csv',mime='text/csv',key='download_summary')
    with st.expander('講義資料の式・相関の作り方・見方'):
        st.latex(r'\frac{dS_j}{dt}=A_jr_j-\frac{S_j}{\tau_j},\quad Q_j=\frac{S_j}{\tau_j}\quad(j=1,2)')
        st.latex(r'\frac{dV}{dt}=Q_1+Q_2-\frac{V}{\tau_r},\qquad h=h_0+\frac{V}{A_r}')
        st.markdown('''降雨ピークの一様分布をN個の等確率区間に分け、その中点を共通の標本にします。相関のある正規乱数の順位で標本を並べ替え、標本相関が目標に近づくよう調整します。流域1の標本の並びは固定し、流域2だけを組み替えます。相関±1は完全な同順・逆順です。標本分布を固定するための層化・順位組替えによるモンテカルロ実験で、各組が独立な単純無作為抽出ではありません。通常の二項分布に基づく信頼区間は表示していません。

相関は降雨ピーク強度のPearson標本相関です。無相関は独立性を保証しません。相関係数だけで依存構造が一意に決まるわけではありません。

基準超過はh > 基準水位。時間格子は10秒以下で、期間0〜Tの最大水位から超過を数えます。有限標本と数値計算の近似を含む授業用の推定値です。

各シナリオで降雨時刻・継続時間と物理条件は固定です。初期S₁,S₂は指定等価水深×各面積。V₀は合流後の河道全体の貯留です。計算では線形性を用いて各流域からの応答を重ね合わせます。

この線形モデルでは、各流域の標本平均入力が同じなら平均水位も相関によらず同じです。平均が同じでも、水位のばらつきと超過率は変わります。''')
    if was_playing and not ss.playing:st.rerun()

animation()
