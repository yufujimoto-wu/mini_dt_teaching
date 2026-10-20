import math
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from model import simulate, at

st.set_page_config(page_title='貯留と流出による水位変化',layout='wide')
st.markdown('<style>.block-container{padding-top:1.3rem}div[data-testid="stMetricValue"]{font-size:1.6rem}</style>',unsafe_allow_html=True)
DEFAULTS=dict(area=5.0,river_area=5.0,tc=.8,tr=.5,h0=.5,s0=5.0,v0=2.5,peak=40.,onset=.5,width=2.,duration=6.,threshold=1.4,speed=1.)
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
    if name=='② 流域からゆっくり流出':ss.tc=2.
    elif name=='③ 河道から抜けにくい':ss.tr=1.5
    elif name=='④ 雨がなくても貯留から流れる':ss.peak=0.
    ss.rate=ss.speed;restart()

with st.sidebar:
    st.header('条件設定')
    st.button('初期設定に戻す',on_click=reset,width='stretch')
    with st.expander('授業用の設定例'):
        for name in ['① 降雨から水位への遅れ','② 流域からゆっくり流出','③ 河道から抜けにくい','④ 雨がなくても貯留から流れる']:
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
    st.subheader('有効降雨（三角形の波形）')
    st.slider('ピークの有効降雨強度 [mm/h]',0.,100.,step=5.,key='peak',on_change=restart)
    st.slider('降雨開始時刻 [h]',0.,6.,step=.25,key='onset',on_change=restart)
    st.slider('降雨継続時間 [h]',.5,6.,step=.25,key='width',on_change=restart)
    #st.caption('降雨開始から継続時間の半分でピークに達し、その後0へ戻ります。流出に寄与する有効降雨を直接与えます。')
    st.subheader('表示と再生')
    st.slider('表示する基準水位 [m]',.5,10.,step=.1,key='threshold')
    st.slider('計算時間 [h]',3.,12.,step=1.,key='duration',on_change=restart)
    st.select_slider('再生速度',options=[.25,.5,1.,2.,4.],key='speed',on_change=speed_change,format_func=lambda x:f'{x:g} 倍')
    #st.caption('1倍ではモデルの1時間を約20秒で再生。表示基準以外のモデル条件を変えると時刻0へ戻ります。')

st.title('貯留と流出による水位の時間変化')
#st.caption('流域と河道をそれぞれ一つの貯留として表す、講義用の確定的なモデルです。')
with st.expander('講義資料の式・単位・見方'):
    a,b=st.columns(2)
    a.latex(r'\frac{dS}{dt}=Ar_e-Q_{\mathrm{in}},\qquad Q_{\mathrm{in}}=\frac{S}{\tau_c}')
    b.latex(r'\frac{dV}{dt}=Q_{\mathrm{in}}-Q_{\mathrm{out}},\qquad Q_{\mathrm{out}}=\frac{V}{\tau_r}')
    st.latex(r'h=h_0+\frac{V}{A_r}')
    st.markdown('''流域の水量Sと河道の水量Vはm³，流量はm³/s，式内の時間は秒．左パネルのkm²・万m²・mm/h・時間は内部で換算．

**流域**は有効降雨によって水が増え，S/τcで河道へ流出．**河道**は流入と下流への流出の差で貯留が変化''')
buttons=st.columns(4)
buttons[0].button('▶ 再生',on_click=play,disabled=ss.playing,width='stretch')
buttons[1].button('⏸ 一時停止',on_click=pause,disabled=not ss.playing,width='stretch')
buttons[2].button('↶ 先頭に戻る',on_click=restart,width='stretch')
buttons[3].button('全時間を表示',on_click=show_all,width='stretch')
st.slider('時刻を指定 [h]（操作すると一時停止）',0.,ss.duration,step=.05,key='seek',on_change=seek)
#st.caption('スライダーは時刻指定用です。再生中の現在時刻は数値と縦線で表示します。')

@st.cache_data(show_spinner=False)
def calculate(area,river_area,tc,tr,h0,s0,v0,peak,onset,width,duration):
    return simulate(area*1e6,river_area*1e4,tc,tr,h0,s0*area*1000,v0*1e4,peak,onset,width,duration)

data=calculate(ss.area,ss.river_area,ss.tc,ss.tr,ss.h0,ss.s0,ss.v0,ss.peak,ss.onset,ss.width,ss.duration)
area=ss.area*1e6
smax=max(1.,max(data['s'])/area*1000*1.15)
hmax=max(ss.threshold*1.15,max(data['h'])*1.15,ss.h0+.5)
flowmax=max(1.,max(data['qin']),max(data['qout']),ss.area*ss.peak/3.6)
#if ss.onset>=ss.duration:st.info('降雨開始が計算期間の外にあります。この区間では初期貯留からの流出だけを表示します。')


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
#    <text x="425" y="393" text-anchor="middle" font-size="15" fill="#64748b">左右は異なる尺度。水面の表示倍率と流量矢印の尺度は再生中固定。</text>
    </svg>'''


@st.fragment(run_every=.1 if ss.playing else None)
def animation():
    was_playing=ss.playing
    advance()
    state=at(data,ss.t)
    metrics=st.columns(4)
    metrics[0].metric('現在の時刻',f'{ss.t:.2f} h')
    metrics[1].metric('対象地点の水位 h',f"{state['h']:.2f} m")
    metrics[2].metric('河道への流入 Qᵢₙ',f"{state['qin']:.2f} m³/s")
    metrics[3].metric('下流への流出 Qₒᵤₜ',f"{state['qout']:.2f} m³/s")
    left,right=st.columns([1.1,1.])
    with left:
        st.subheader('流域から河道へ水がたまる様子')
        st.markdown(diagram(state),unsafe_allow_html=True)
        change=state['qin']-state['qout']
        st.write('**河道の水位：'+('上昇中**' if change>1e-6 else '低下中**' if change < -1e-6 else 'ほぼ一定**')+f'　流入−流出 = {change:+.2f} m³/s')
        st.caption('h₀は水位の換算基準（灰色点線）．河道の貯留Vはh₀からの水位増分に対応．')
    with right:
        st.subheader('水位の時間変化')
        # Slice by actual time, including additional rainfall breakpoints.
        import bisect
        n=bisect.bisect_right(data['t'],ss.t)
        tt=data['t'][:n]+[ss.t];hh=data['h'][:n]+[state['h']]
        fig=go.Figure(go.Scatter(x=tt,y=hh,name='水位 h',mode='lines',line=dict(color='#2563eb',width=3)))
        fig.add_trace(go.Scatter(x=[ss.t],y=[state['h']],name='現在',mode='markers',marker=dict(size=9,color='#2563eb'),showlegend=False))
        fig.add_hline(y=ss.threshold,line_color='#dc2626',line_dash='dash',annotation_text='基準水位',annotation_position='top right')
        fig.add_vline(x=ss.t,line_color='#94a3b8',line_dash='dot')
        fig.update_layout(height=340,margin=dict(t=45,l=45,r=15,b=45),legend=dict(orientation='h',y=1.02,yanchor='bottom'),uirevision='water',xaxis=dict(title='時間 [h]',range=[0,ss.duration],fixedrange=True),yaxis=dict(title='水位 [m]',range=[0,hmax],fixedrange=True))
        st.plotly_chart(fig,width='stretch',key='water_level',config={'displayModeBar':False})
    st.progress(min(1.,ss.t/ss.duration))
    if state['h']>=ss.threshold:st.info('現在の水位が表示基準以上．')
    else:st.caption(f"表示基準まで {ss.threshold-state['h']:.2f} m")
    with st.expander('有効降雨・流量・貯留量の推移'):
        fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.09,subplot_titles=['有効降雨','河道への流入と下流への流出','流域と河道の貯留水量'])
        for row,keys,names,colors in [(1,['rain'],['有効降雨'],['#0284c7']),(2,['qin','qout'],['Qᵢₙ','Qₒᵤₜ'],['#16a34a','#9333ea']),(3,['s','v'],['流域 S','河道 V'],['#16a34a','#9333ea'])]:
            for key,name,color in zip(keys,names,colors):
                yy=data[key][:n]+[state[key]]
                if row==3:yy=[v/10000 for v in yy]
                fig.add_trace(go.Scatter(x=tt,y=yy,name=name,line=dict(color=color)),row=row,col=1)
        for row,unit in [(1,'mm/h'),(2,'m³/s'),(3,'万m³')]:fig.update_yaxes(title_text=unit,row=row,col=1)
        fig.update_xaxes(range=[0,ss.duration]);fig.update_xaxes(title_text='時間 [h]',row=3,col=1)
        fig.update_layout(height=620,margin=dict(t=60),legend=dict(orientation='h',y=1.08))
        st.plotly_chart(fig,width='stretch',key='diagnostics')
        st.caption(f"現在までの累積有効降雨量：{state['input']/10000:.3f} 万m³ ／ 累積下流流出量：{state['output']/10000:.3f} 万m³．水収支の数値残差：{state['balance']:.2e} m³")
    if ss.t>=ss.duration:
        peak_index=max(range(len(data['h'])),key=lambda i:data['h'][i])
        st.write(f"計算期間内の最高水位：**{data['h'][peak_index]:.2f} m**（時刻 {data['t'][peak_index]:.2f} h）")
        import csv,io
        buf=io.StringIO();writer=csv.writer(buf);writer.writerow(['time_h','catchment_storage_m3','river_storage_m3','water_level_m','effective_rain_mm_h','qin_m3_s','qout_m3_s'])
        writer.writerows(zip(*(data[k] for k in ['t','s','v','h','rain','qin','qout'])))
        st.download_button('計算結果をCSV保存',buf.getvalue().encode('utf-8-sig'),file_name='water_storage.csv',mime='text/csv',key='download_water')
    if was_playing and not ss.playing:st.rerun()

animation()
#st.caption('授業用の簡略モデル．内部では10秒以下の区間ごとに降雨を一定近似し，2段の線形貯留式を解析的に更新．')
