from dataclasses import asdict
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from model import Conditions, simulate, metrics

st.set_page_config(page_title='Mini Energy Class Room Digital Twin', layout='wide', initial_sidebar_state='expanded')
st.markdown("""<style>
.block-container {max-width:1600px; padding:3rem 1.5rem .5rem;}
[data-testid="stVerticalBlock"] {gap:.35rem;}
h1 {font-size:1.5rem !important; padding:0 0 .35rem !important; line-height:1.5 !important;}
h2,h3 {font-size:1.05rem !important; padding:.25rem 0 !important;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.45rem;}
.kpi {border-bottom:1px solid #e2e8ee;padding:10px 0;}
.kpi-label {font-size:13px;color:#526272;}
.kpi-value {font-size:25px;font-weight:650;color:#007c91;line-height:1.5;}
.kpi-value small {font-size:13px;font-weight:400;}
.kpi-base {font-size:12px;color:#607080;}
</style>""", unsafe_allow_html=True)
st.title('Mini Energy Class Room Digital Twin')
st.caption('教室の予冷による室温・電力使用量の比較 ｜ 授業 9:00–11:00')

with st.sidebar:
    st.subheader('冷房の運転方法')
    lead = st.slider('授業の何分前から開始するか', 0, 90, 45, 5, key='lead')
    setpoint = st.slider('両案共通の設定温度（℃）', 24.0, 28.0, 26.0, 0.5, key='setpoint')
    st.subheader('教室・外部条件')
    people = st.slider('授業中の人数（人）',0,80,40,5)
    outdoor = st.slider('外気温（℃）',25.0,38.0,33.0,0.5)
    initial = st.slider('7:30の室温（℃）',24.0,36.0,31.0,0.5)
    with st.expander('設備・評価条件'):
        capacity = st.number_input('空調の冷却能力（kW）',5.0,30.0,20.0,1.0)
        threshold = st.number_input('比較の目安となる室温（℃）',26.0,30.0,28.0,0.5)
        factor = st.number_input('CO₂排出係数（kg-CO₂/kWh）',0.0,1.5,0.4,0.05)
        st.caption('室温の目安は比較用の設定値。快適性には湿度・気流・個人差も関係。')
    st.caption('教室・外部条件は両案に共通。')

c = Conditions(initial_temp=initial, outdoor_temp=outdoor, people=people,
               capacity=capacity, threshold=threshold, emission_factor=factor)
base = simulate(c, 0, setpoint)
trial = simulate(c, lead, setpoint)
mb, mt = metrics(base,c), metrics(trial,c)
start = datetime(2026,1,1,9)
label = (start-timedelta(minutes=lead)).strftime('%H:%M')
st.caption(f'基準案：9:00開始 ／ 比較案：{label}開始（{lead}分前） ｜ 両案とも設定 {setpoint:.1f}℃')

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                    row_heights=[0.65,0.35])
colors = ['#7b8593','#007c91']
for df, name, color, dash in [(base,'基準案：9:00開始',colors[0],'dash'),(trial,f'比較案：{label}開始',colors[1],'solid')]:
    for row,field in [(1,'temperature'),(2,'power')]:
        fig.add_trace(go.Scatter(x=df.minute,y=df[field],name=name,legendgroup=name,
                      showlegend=row==1,mode='lines',line=dict(color=color,width=3,dash=dash),
                      customdata=[(start+timedelta(minutes=int(m))).strftime('%H:%M') for m in df.minute],
                      hovertemplate='%{customdata}<br>%{y:.2f}'+(' ℃' if row==1 else ' kW')+'<extra>%{fullData.name}</extra>'),row=row,col=1)
fig.add_hline(y=threshold,line_dash='dot',line_color='#b85c22',annotation_text=f'比較の目安 {threshold:.1f}℃',row=1,col=1)
fig.add_hline(y=setpoint,line_dash='dot',line_color='#8ba7bb',annotation_text=f'設定 {setpoint:.1f}℃',row=1,col=1)
for row in [1,2]:
    fig.add_vrect(x0=0,x1=120,fillcolor='#edf5fb',opacity=.5,line_width=0,layer='below',row=row,col=1)
    fig.add_vline(x=0,line_color='#415a77',line_width=1,row=row,col=1)
fig.add_annotation(x=0,y=1.05,xref='x',yref='paper',text='授業開始 9:00',showarrow=False,xanchor='left')
ticks=list(range(-90,121,30))
fig.update_xaxes(tickvals=ticks,ticktext=[(start+timedelta(minutes=t)).strftime('%H:%M') for t in ticks],range=[-90,120])
fig.update_yaxes(title_text='室温（℃）',row=1,col=1)
fig.update_yaxes(title_text='消費電力（kW）',rangemode='tozero',row=2,col=1)
fig.update_layout(height=425,margin=dict(l=10,r=30,t=50,b=10),paper_bgcolor='white',plot_bgcolor='white',
                  legend=dict(orientation='h',y=1.16,x=0),font=dict(family='sans-serif',size=13),hovermode='x unified')
main_view, saved_view, model_view, export_view = st.tabs(['室温・電力の比較','保存案との比較','モデルの仕組み','データ保存'])
with main_view:
    chart_area, kpi_area = st.columns([3.5,1], gap='medium')
    with chart_area:
        st.plotly_chart(fig,width='stretch',config={'displayModeBar':False})
        st.caption('青い背景：授業中 ｜ 電力・CO₂の集計：7:30–11:00')
    with kpi_area:
        st.subheader('比較案の結果')
        for title,key,unit,fmt in zip(['授業開始時の室温','目安を超える時間（授業中）','合計電力使用量','CO₂排出量'],['start','hot','energy','co2'],['℃','分','kWh','kg-CO₂'],['.1f','.0f','.1f','.1f']):
            st.markdown(f'<div class="kpi"><div class="kpi-label">{title}</div>'
                        f'<div class="kpi-value">{mt[key]:{fmt}} <small>{unit}</small></div>'
                        f'<div class="kpi-base">基準案 {mb[key]:{fmt}} {unit} ／ 差 {mt[key]-mb[key]:+{fmt}}</div></div>',unsafe_allow_html=True)
        st.caption('CO₂：一定の排出係数で換算')

with saved_view:
    st.caption('保存するのは開始時刻と設定温度。人数・外気温などは現在の共通条件で再計算。')
    if st.button('現在の運転案を保存',key='save'):
        st.session_state.saved = {'lead':lead,'setpoint':setpoint}
    if 'saved' in st.session_state:
        s=st.session_state.saved
        saved=simulate(c,s['lead'],s['setpoint']); ms=metrics(saved,c)
        st.write(f"保存案：{s['lead']}分前開始・設定{s['setpoint']:.1f}℃ ／ 現在案：{lead}分前開始・設定{setpoint:.1f}℃")
        f=go.Figure()
        for df,name,color in [(saved,'保存案','#8064ad'),(trial,'現在案','#007c91')]:
            f.add_trace(go.Scatter(x=df.minute,y=df.temperature,name=name,line=dict(color=color,width=3)))
        f.add_hline(y=threshold,line_dash='dot')
        f.add_vline(x=0,line_dash='dot')
        f.update_xaxes(tickvals=ticks,ticktext=[(start+timedelta(minutes=t)).strftime('%H:%M') for t in ticks],title='時刻')
        f.update_layout(height=280,margin=dict(t=20,b=10),yaxis_title='室温（℃）',legend=dict(orientation='h'))
        st.plotly_chart(f,width='stretch',config={'displayModeBar':False})
        st.dataframe(pd.DataFrame({'指標':['授業開始時の室温（℃）','目安を超える時間（分）','電力使用量（kWh）','CO₂排出量（kg-CO₂）'],
            '保存案':[round(ms[k],1) for k in ['start','hot','energy','co2']],
            '現在案':[round(mt[k],1) for k in ['start','hot','energy','co2']]}),hide_index=True,width='stretch')

with model_view:
    st.markdown('''**室温の変化 = 外部から入る熱 ＋ 人から出る熱 − 空調で取り除く熱**

・教室全体を一つの温度で表現し、1分ごとの熱収支から次の室温を計算

・外気温と日射は一定、人数は授業開始時に増加。空調は冷却能力の上限内で設定温度に向けて出力を調整

・実測データとの接続前のシミュレーション。現実の教室への適用には、温度・電力の観測、教室に合わせたモデルの調整、運転後の検証が必要

・今回の計算では湿度、座席ごとの温度差、機器の起動損失などを省略。予冷による省エネを保証するものではない''')
    st.caption('計算上の設定：有効熱容量 2.5 kWh/℃、外部との熱交換 0.6 kW/℃、日射等の熱 1.5 kW、人体の発熱 0.075 kW/人、冷房効率 3.5。授業用の設定値。')

with export_view:
    export=pd.DataFrame({'授業開始からの分':base.minute,'基準案_室温_C':base.temperature,'比較案_室温_C':trial.temperature,
                         '基準案_消費電力_kW':base.power,'比較案_消費電力_kW':trial.power})
    st.download_button('２案の時系列をCSVで保存',export.to_csv(index=False).encode('utf-8-sig'),'classroom_comparison.csv','text/csv')
    st.json({'共通条件':asdict(c),'比較案_開始分前':lead,'両案設定温度':setpoint},expanded=False)
