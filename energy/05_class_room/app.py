from dataclasses import replace
from datetime import datetime,timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dataclasses import dataclass
import numpy as np
import pandas as pd

INSULATION = {'高断熱':0.15, '標準':0.45, '低断熱':0.90}
PROFILES = {'授業前が低い':(0.20,0.60), '時間帯で一定':(0.40,0.40), '授業前が高い':(0.60,0.20)}
@dataclass(frozen=True)
class Conditions:
    initial_temp: float = 30.0
    outdoor_temp: float = 33.0
    people: int = 40
    capacity: float = 20.0
    capacitance: float = 2.5
    envelope: float = 0.15
    solar: float = 1.5
    cop: float = 3.5
    lower: float = 24.0
    upper: float = 28.0
    ci_before: float = 0.20
    ci_during: float = 0.60


def simulate(c, lead, pre_temp=24.5, limit=1.0):
    """10–16h; class 13–16h. Limit is class-time capacity fraction."""
    if not (0<=lead<=180 and 0<limit<=1 and c.capacitance>0 and c.cop>0):
        raise ValueError('Invalid parameters')
    temp=c.initial_temp
    energy=carbon=0.
    rows=[]
    for minute in range(-180,181):
        occupied=0<=minute<180
        gain=(c.envelope+(0.15 if occupied else 0.05))*(c.outdoor_temp-temp)+c.solar+(0.075*c.people if occupied else 0)
        target=c.upper-0.2 if minute>=0 else pre_temp
        cap=c.capacity*(limit if minute>=0 else 1)
        cooling=float(np.clip(gain+c.capacitance*(temp-target)/(10/60),0,cap)) if -lead<=minute<180 else 0.
        power=cooling/c.cop
        ci=c.ci_before if minute<0 else c.ci_during
        rows.append((minute,temp,power,ci,energy,carbon,gain,cooling))
        if minute<180:
            energy+=power/60
            carbon+=power*ci/60
            temp+=(gain-cooling)/c.capacitance/60
    return pd.DataFrame(rows,columns=['minute','temperature','power','ci','energy','co2','heat_gain','cooling'])


def metrics(d,c):
    t=d.loc[d.minute>=0,'temperature']
    bad=(t<c.lower-1e-8)|(t>c.upper+1e-8)
    # Duration represented by intervals; check all endpoints for feasibility as well.
    a=t.to_numpy()
    outside=((a[:-1]+a[1:])/2<c.lower)|((a[:-1]+a[1:])/2>c.upper)
    return dict(energy=float(d.energy.iloc[-1]),co2=float(d.co2.iloc[-1]),
        outside=int(outside.sum()),feasible=not bool(bad.any()),
        start=float(t.iloc[0]),end=float(t.iloc[-1]),
        minimum=float(t.min()),maximum=float(t.max()),
        peak=float(d.loc[(d.minute>=0)&(d.minute<180),'power'].max()),
        total_peak=float(d.power.max()))


def baseline(c):
    """Latest start that meets the same class temperature band; monotonic bisection."""
    target=c.upper-0.2
    last=simulate(c,180,target,1.)
    if not metrics(last,c)['feasible']:
        return last,180
    lo,hi=0,180
    while lo<hi:
        mid=(lo+hi)//2
        if metrics(simulate(c,mid,target,1.),c)['feasible']: hi=mid
        else: lo=mid+1
    return simulate(c,lo,target,1.),lo


st.set_page_config(page_title='Mini Energy Class Room Digital Twin',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container {max-width:1600px;padding:3rem 1.5rem .4rem;}
[data-testid="stVerticalBlock"] {gap:.3rem;}
h1 {font-size:1.45rem !important;padding:0 0 .3rem !important;line-height:1.5 !important;}
h2,h3 {font-size:1.02rem !important;padding:.2rem 0 !important;}
.kpi {border-bottom:1px solid #e2e8ee;padding:7px 0;}
.kpi-label {font-size:13px;color:#526272;}.kpi-value {font-size:24px;font-weight:650;color:#007c91;}
.kpi-value small,.kpi-base {font-size:12px;font-weight:400;color:#526272;}
</style>''',unsafe_allow_html=True)
st.title('Mini Energy Class Room Digital Twin')
st.caption('室温の条件を満たしながら、冷房を低炭素な時間帯へ移す ｜ 授業 13:00–16:00')
with st.sidebar:
    st.subheader('予冷と授業中の運転')
    lead=st.slider('授業の何分前から予冷するか',0,180,120,15,key='lead')
    pre=st.slider('予冷の設定温度（℃）',24.,27.5,24.5,.5,key='pre')
    limit=st.slider('授業中の冷却能力の上限（%）',20,100,60,5,key='limit')/100
    st.subheader('建物と電力の条件')
    insulation=st.selectbox('外皮の断熱性能',list(INSULATION),key='insulation')
    profile=st.selectbox('時間別のCO₂排出係数',list(PROFILES),key='profile')
    with st.expander('教室・設備の詳細'):
        outdoor=st.slider('外気温（℃）',28.,38.,33.,.5)
        people=st.slider('授業中の人数',0,80,40,5)
        initial=st.slider('10:00の室温（℃）',27.,35.,30.,.5)
        capacity=st.slider('最大冷却能力（kW）',10.,30.,20.,1.)
        mass=st.selectbox('建物・家具の有効熱容量', ['小：1.5 kWh/℃','中：2.5 kWh/℃','大：4.0 kWh/℃'],index=1)
    st.caption('両案共通の室温条件：授業中24〜28℃\n\n断熱性能と熱容量は別の特性。')
ci0,ci1=PROFILES[profile]
c=Conditions(initial_temp=initial,outdoor_temp=outdoor,people=people,capacity=capacity,
             capacitance={'小':1.5,'中':2.5,'大':4.}[mass[0]],envelope=INSULATION[insulation],ci_before=ci0,ci_during=ci1)
@st.cache_data(show_spinner=False)
def reference(c): return baseline(c)
b,bl=reference(c);p=simulate(c,lead,pre,limit)
mb,mp=metrics(b,c),metrics(p,c)
clock=lambda m:(datetime(2026,1,1,13)+timedelta(minutes=int(m))).strftime('%H:%M')
st.caption(f'通常運転：{clock(-bl)}開始・27.8℃維持 ／ 予冷運転：{clock(-lead)}開始・{pre:.1f}℃ → 授業中27.8℃、能力上限{limit:.0%}')
main,compare,explain=st.tabs(['運転と排出量','条件による効果の違い','判断の前提・モデル'])
with main:
    if not mb['feasible']:
        st.warning('通常運転も室温条件を満たせない設定。設備能力・外気温・人数を確認。削減効果の比較は未成立。')
    elif not mp['feasible']:
        st.warning('予冷案は室温条件を満たさないため、CO₂が減っても採用候補とは評価できない。')
    else:
        d=mp['co2']-mb['co2']
        st.caption(f'両案とも室温条件を充足 ｜ 予冷によるCO₂の変化 {d:+.2f} kg-CO₂（{d/mb["co2"]*100:+.1f}%）')
    chart,kpi=st.columns([3.5,1],gap='medium')
    with chart:
        fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.09,row_heights=[.47,.32,.21])
        for df,name,color,dash in [(b,'通常運転','#7b8593','dash'),(p,'予冷運転','#007c91','solid')]:
            for row,col,unit in [(1,'temperature','℃'),(2,'power','kW')]:
                fig.add_trace(go.Scatter(x=df.minute,y=df[col],name=name,legendgroup=name,showlegend=row==1,
                    line=dict(color=color,width=2.5,dash=dash),customdata=[clock(m) for m in df.minute],
                    hovertemplate='%{customdata}<br>%{y:.2f} '+unit+'<extra>%{fullData.name}</extra>'),row=row,col=1)
        fig.add_trace(go.Scatter(x=p.minute,y=p.ci,name='CO₂排出係数',line=dict(color='#aa6b22',shape='hv'),showlegend=False,
            customdata=[clock(m) for m in p.minute],hovertemplate='%{customdata}<br>%{y:.2f} kg-CO₂/kWh<extra></extra>'),row=3,col=1)
        fig.add_hrect(y0=24,y1=28,fillcolor='#d6eee2',opacity=.4,line_width=0,layer='below',row=1,col=1)
        fig.add_hline(y=28,line_dash='dot',line_color='#8aaa97',annotation_text='室温条件 24〜28℃',row=1,col=1)
        for row in [1,2,3]: fig.add_vline(x=0,line_dash='dot',line_color='#596a78',row=row,col=1)
        fig.add_annotation(x=0,y=1.04,xref='x',yref='paper',text='授業開始 13:00',showarrow=False)
        fig.update_xaxes(tickvals=list(range(-180,181,60)),ticktext=[clock(m) for m in range(-180,181,60)],range=[-180,180])
        for row,title in [(1,'室温（℃）'),(2,'消費電力（kW）'),(3,'排出係数')]: fig.update_yaxes(title_text=title,row=row,col=1)
        fig.update_yaxes(range=[0,.7],row=3,col=1)
        fig.update_yaxes(rangemode='tozero',row=2,col=1)
        fig.update_layout(height=430,margin=dict(l=5,r=15,t=38,b=5),legend=dict(orientation='h',y=1.14),
                          font=dict(size=12),plot_bgcolor='white',paper_bgcolor='white',hovermode='x unified')
        st.plotly_chart(fig,width='stretch',config={'displayModeBar':False})
        st.caption('集計10:00–16:00（予冷を含む）｜ 排出係数：kg-CO₂/kWh。時間帯の値は条件比較用の設定。')
    with kpi:
        st.subheader('予冷案の結果')
        for title,key,unit,fmt in [('CO₂排出量','co2','kg-CO₂','.2f'),('電力使用量','energy','kWh','.2f'),('授業中の最大電力','peak','kW','.2f'),('室温条件を外れる時間','outside','分','.0f')]:
            st.markdown(f'<div class="kpi"><div class="kpi-label">{title}</div><div class="kpi-value">{mp[key]:{fmt}} <small>{unit}</small></div><div class="kpi-base">通常 {mb[key]:{fmt}} ／ 差 {mp[key]-mb[key]:+{fmt}}</div></div>',unsafe_allow_html=True)
        st.caption(f'授業中 {mp["minimum"]:.1f}〜{mp["maximum"]:.1f}℃\n\n全時間の最大電力：通常{mb["total_peak"]:.1f}／予冷{mp["total_peak"]:.1f} kW')
with compare:
    st.subheader('同じ運転方法を、断熱性能と排出係数の異なる条件で比較')
    st.caption('人数・外気温・予冷設定は現在の入力値。各条件で通常運転の開始時刻を再計算。差は「予冷 − 通常」。')
    rows=[]
    for ins,u in INSULATION.items():
        for prof,(a,z) in PROFILES.items():
            cc=replace(c,envelope=u,ci_before=a,ci_during=z)
            bb,_=reference(cc);pp=simulate(cc,lead,pre,limit);bm,pm=metrics(bb,cc),metrics(pp,cc)
            valid=bm['feasible'] and pm['feasible']
            rows.append({'断熱性能':ins,'排出係数':prof,'室温条件':'両案充足' if valid else '比較未成立',
                '電力量の差（kWh）':round(pm['energy']-bm['energy'],2),'CO₂の差（kg）':round(pm['co2']-bm['co2'],2),
                'CO₂変化率':f'{(pm["co2"]/bm["co2"]-1)*100:+.1f}%' if valid else '—',
                '予冷案の条件外時間（分）':pm['outside']})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch',height=365)
    st.markdown('**比較の視点**：電力量の増加と時間移動による排出削減の関係 ／ 断熱改修と運転変更の組合せ ／ 室温条件を満たせない運転の除外')
with explain:
    st.markdown('''**判断の目的**

授業中の室温を24〜28℃に保ちながら、冷房に伴うCO₂排出量を低減。温度範囲は比較のための設定で、快適性・健康の保証値ではない。

**比較する運転**

- 通常運転：27.8℃を目標に冷房。授業中の室温条件を満たす最も遅い開始時刻を1分刻みで探索（最大3時間前）。最適なCO₂運転を探索するものではない。
- 予冷運転：授業前に低めの温度まで冷やし、授業中は27.8℃へ設定を戻して冷却能力を制限。冷房を前倒しする代わりに、冷やした建物への熱流入が増える可能性。
- 両案で同じ初期温度・外気温・人数・設備・温度範囲。室温条件の未達は排出削減と分けて評価。

**計算と評価範囲**

- 室温変化＝（外皮・換気からの熱＋日射等＋人の熱−冷却）÷有効熱容量。1分刻みの熱収支。
- 外皮の熱の伝わりやすさ：高断熱0.15／標準0.45／低断熱0.90 kW/℃。換気等は別途、授業前0.05／授業中0.15 kW/℃。断熱性能のみを変えても換気量・熱容量は不変。
- 建物・家具の熱容量は一つの室温にまとめた近似。日射等1.5 kW、人体0.075 kW/人、冷房効率3.5。実測による校正前の設定。
- CO₂＝各時刻の電力使用量×その時刻の排出係数の合計。授業前／授業中の係数は0.20／0.60、0.40／0.40、0.60／0.20。実際の地域・日のデータではなく、低炭素時間帯は常に授業前とは限らない。
- 10〜16時の空調電力のみを集計。断熱工事・設備製造の排出量や、電力系統の限界排出係数に基づく因果的な削減効果は評価対象外。
- 授業中の最大電力と全時間の最大電力を区別。建物全体のピーク削減には他の電力需要との合算が必要。
- 湿度・座席差・壁と空気の温度差を省略。実際の運用には観測によるモデル校正、時間別排出係数の確認、実行後の効果検証が必要。
''')
    st.caption(f'16:00の室温：通常{mb["end"]:.2f}℃／予冷{mp["end"]:.2f}℃。終了時に残る冷熱の差と、その後の運転は未精算。')
    st.markdown('[予冷と建物の蓄熱を用いた需要移動の研究（Berkeley Lab）](https://eta-publications.lbl.gov/publications/demand-shifting-thermal-mass-large-0)')
