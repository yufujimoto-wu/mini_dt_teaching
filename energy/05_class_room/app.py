from dataclasses import asdict
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from model import Conditions, simulate, metrics

st.set_page_config(page_title='教室の予冷と室温', layout='wide')
st.markdown('''<style>
.block-container {max-width:1180px; padding-top:2rem;}
h1 {font-size:1.8rem !important;} h2 {font-size:1.25rem !important;}
[data-testid="stMetricValue"] {font-size:1.55rem;}
</style>''', unsafe_allow_html=True)
st.title('教室の予冷と室温')
st.caption('授業前の冷房開始時刻による、室温と電力使用量の比較｜授業 9:00–11:00')

a,b = st.columns(2)
with a:
    lead = st.slider('比較案：授業の何分前から冷房を開始するか', 0, 90, 45, 5, key='lead')
with b:
    setpoint = st.slider('両案共通の設定温度（℃）', 24.0, 28.0, 26.0, 0.5, key='setpoint')

with st.expander('教室・外部条件', expanded=False):
    st.caption('変更した条件を両案に共通して適用。7:30の室温から、空調停止中の温度変化も計算。')
    x,y,z = st.columns(3)
    initial = x.slider('7:30の室温（℃）',24.0,36.0,31.0,0.5)
    outdoor = y.slider('外気温（℃）',25.0,38.0,33.0,0.5)
    people = z.slider('授業中の人数（人）',0,80,40,5)
    capacity = x.slider('空調の冷却能力（kW）',5.0,30.0,20.0,1.0)
    threshold = y.slider('比較の目安となる室温（℃）',26.0,30.0,28.0,0.5)
    factor = z.number_input('電力のCO₂排出係数（kg-CO₂/kWh）',0.0,1.5,0.4,0.05)
    st.caption('目安の室温は比較用の設定値。快適性には湿度・気流・個人差なども関係。')

c = Conditions(initial_temp=initial, outdoor_temp=outdoor, people=people,
               capacity=capacity, threshold=threshold, emission_factor=factor)
base = simulate(c, 0, setpoint)
trial = simulate(c, lead, setpoint)
mb, mt = metrics(base,c), metrics(trial,c)
start = datetime(2026,1,1,9)
label = (start-timedelta(minutes=lead)).strftime('%H:%M')
st.markdown(f'**基準案：9:00開始**　／　**比較案：{label}開始（{lead}分前）**　｜　設定温度 {setpoint:.1f}℃・{people}人・外気温 {outdoor:.1f}℃')

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
fig.update_layout(height=490,margin=dict(l=10,r=30,t=50,b=10),paper_bgcolor='white',plot_bgcolor='white',
                  legend=dict(orientation='h',y=1.17,x=0),font=dict(family='sans-serif',size=13),hovermode='x unified')
st.plotly_chart(fig,width='stretch',config={'displayModeBar':False})

st.subheader('比較結果')
cols=st.columns(4)
for col,title,key,unit,fmt in zip(cols,['授業開始時の室温','授業中に目安を超える時間','合計電力使用量','電力使用に伴うCO₂排出量'],['start','hot','energy','co2'],['℃','分','kWh','kg-CO₂'],['.1f','.0f','.1f','.1f']):
    col.metric(title,f'{mt[key]:{fmt}} {unit}',f'{mt[key]-mb[key]:+{fmt}} {unit}（基準案との差）',delta_color='off')
    col.caption(f'基準案 {mb[key]:{fmt}} {unit}')
st.caption('電力・CO₂の集計期間：7:30–11:00。温度超過時間：9:00–11:00。CO₂は一定の排出係数による換算。')
if lead == 0:
    st.info('開始時刻が同じため、２案の結果は一致。')
elif mt['start'] > threshold:
    st.info('比較案でも、授業開始時に目安の室温を超過。開始時刻・設定温度・空調能力の関係を確認。')

with st.expander('保存した運転案との比較'):
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

with st.expander('モデルの仕組みと、現実の教室との接続'):
    st.markdown('''**室温の変化 = 外部から入る熱 ＋ 人から出る熱 − 空調で取り除く熱**

・教室全体を一つの温度で表現し、1分ごとの熱収支から次の室温を計算

・外気温と日射は一定、人数は授業開始時に増加。空調は冷却能力の上限内で設定温度に向けて出力を調整

・実測データとの接続前のシミュレーション。現実の教室への適用には、温度・電力の観測、教室に合わせたモデルの調整、運転後の検証が必要

・今回の計算では湿度、座席ごとの温度差、機器の起動損失などを省略。予冷による省エネを保証するものではない''')
    st.caption('計算上の設定：有効熱容量 2.5 kWh/℃、外部との熱交換 0.6 kW/℃、日射等の熱 1.5 kW、人体の発熱 0.075 kW/人、冷房効率 3.5。授業用の設定値。')

with st.expander('計算結果のダウンロード'):
    export=pd.DataFrame({'授業開始からの分':base.minute,'基準案_室温_C':base.temperature,'比較案_室温_C':trial.temperature,
                         '基準案_消費電力_kW':base.power,'比較案_消費電力_kW':trial.power})
    st.download_button('２案の時系列をCSVで保存',export.to_csv(index=False).encode('utf-8-sig'),'classroom_comparison.csv','text/csv')
    st.json({'共通条件':asdict(c),'比較案_開始分前':lead,'両案設定温度':setpoint},expanded=False)
