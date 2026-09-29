"""Teaching-only urban flood DT. Run: streamlit run app.py.

Pure model functions intentionally do not import Streamlit or read session state.
"""
from dataclasses import dataclass
import numpy as np

N = 10
PUMP_COUNT = 3
PUMP_DROP = 0.25  # m, same-cell reduction over the fixed intervention interval
THRESHOLD = 0.10  # m, damage starts above this depth
STRATEGIES = ("空間を均等にカバー", "重要地区を重点観測", "低地を重点観測")


@dataclass(frozen=True)
class City:
    xy: np.ndarray
    elevation: np.ndarray
    weight: np.ndarray
    prior: np.ndarray
    truth: np.ndarray
    noise: np.ndarray


def make_city(scenario=0):
    """Known terrain/exposure/prior; unknown actual rain and drainage anomaly."""
    yy, xx = np.mgrid[0:N, 0:N]
    x, y = xx.ravel(), yy.ravel()
    xy = np.column_stack((x, y)).astype(float)
    elevation = 0.3 + 0.7 * (x / 9) + 0.25 * np.cos(y / 2)
    drainage = 0.12 + 0.06 * (y / 9)
    weight = np.ones(N * N)
    def importance_hill(cx, cy, peak, sx, sy):
        return peak * np.exp(-0.5 * (((x-cx)/sx)**2 + ((y-cy)/sy)**2))

    # Smooth exposure around the city centre, hospital/shelter, and housing.
    weight += importance_hill(4.0, 4.0, 5.0, 1.6, 1.3)
    weight += importance_hill(7.5, 1.5, 10.0, 0.8, 0.8)
    weight += importance_hill(2.0, 7.0, 3.5, 1.5, 1.2)
    # Depth = max(0, accumulated rain + terrain ponding - drainage).
    ponding = 0.22 * (elevation.max() - elevation)
    prior = np.maximum(0, 0.20 + ponding - drainage)
    broad = 0.24 * np.exp(-((x - 2)**2 + (y - 6)**2) / 16)
    local = (0.36, 0.12, 0.46)[scenario] * np.exp(-((x - 7.6)**2 + (y - 1.4)**2) / 1.5)
    rain = 0.20 + broad + local
    drainage_error = 0.07 * np.sin(x * 0.7 + y * 0.45 + scenario)
    truth = np.maximum(0, rain + ponding - drainage + drainage_error)
    noise = np.random.default_rng(2026 + scenario).normal(size=N*N)
    return City(xy, elevation, weight, prior, truth, noise)


def sensor_order(city, strategy):
    """Nested layouts based ONLY on pre-event known data; never truth."""
    dist = np.sum((city.xy[:, None] - city.xy[None, :])**2, axis=2)
    if strategy == STRATEGIES[0]:
        priority = np.ones(N*N)
        first = 44
    elif strategy == STRATEGIES[1]:
        priority = city.weight
        first = int(np.argmax(priority))
    elif strategy == STRATEGIES[2]:
        priority = 1 + 5 * (city.elevation.max() - city.elevation)
        first = int(np.argmin(city.elevation))
    else:
        raise ValueError(strategy)
    chosen = [first]
    while len(chosen) < N*N:
        score = np.min(dist[:, chosen], axis=1) * priority
        score[chosen] = -1
        chosen.append(int(np.argmax(score)))
    return np.array(chosen)


def observe(city, sensors, sigma):
    # Fixed standard noise per cell: changing layout/count does not redraw data.
    return np.maximum(0, city.truth[sensors] + sigma * city.noise[sensors])


def estimate(city, sensors, values):
    """Prior + spatially correlated residual interpolation (length 1.8 cells).

    Fixed regularizer for every design/precision; sensor locations are exact.
    No truth, exposure, or realized error is read by this estimator.
    """
    d2 = np.sum((city.xy[:, None] - city.xy[sensors][None, :])**2, axis=2)
    cross = np.exp(-d2 / (2 * 1.8**2))
    coeff = np.linalg.solve(cross[sensors] + np.eye(len(sensors))*0.01,
                            values - city.prior[sensors])
    return np.maximum(0, city.prior + cross @ coeff)


def damage(depth, weight):
    return weight * np.maximum(depth - THRESHOLD, 0)


def gains(depth, weight):
    return damage(depth, weight) - damage(np.maximum(depth - PUMP_DROP, 0), weight)


def dispatch(depth, weight):
    # Stable index breaks ties; additive, disjoint effects => exact optimal top-3.
    ids = np.array(list(CANDIDATES))
    return ids[np.argsort(-gains(depth, weight)[ids], kind="stable")[:PUMP_COUNT]]


def evaluate(city, pumps):
    pumps = np.asarray(pumps, dtype=int)
    if len(pumps) > PUMP_COUNT or len(set(pumps.tolist())) != len(pumps):
        raise ValueError("ポンプは異なる地点に最大3台です")
    if any(int(i) not in CANDIDATES for i in pumps):
        raise ValueError("都市外の地点です")
    post = city.truth.copy()
    post[pumps] = np.maximum(0, post[pumps] - PUMP_DROP)
    baseline = float(damage(city.truth, city.weight).sum())
    loss = float(damage(post, city.weight).sum())
    oracle = dispatch(city.truth, city.weight)
    best = float(gains(city.truth, city.weight)[oracle].sum())
    return dict(baseline=baseline, loss=loss, avoided=baseline-loss,
                regret=max(0.0, best - (baseline-loss)), oracle_avoided=best)


def run_design(city, count, strategy, sigma):
    sensors = sensor_order(city, strategy)[:count]
    values = observe(city, sensors, sigma)
    prediction, _ = estimate_sources(city.xy, city.prior, make_reports(city), sensors, values, sigma, "通報＋センサ・現地測定")
    pumps = dispatch(prediction, city.weight)
    result = evaluate(city, pumps)
    result.update(sensors=sensors, values=values, prediction=prediction, pumps=pumps,
                  rmse=float(np.sqrt(np.mean((prediction-city.truth)**2))))
    return result


def cell_name(i):
    return f"{chr(65 + int(i)//N)}{int(i)%N + 1:02d}"


# Fixed feasible districts, independent of the hidden flood realization.
CANDIDATES = {12:"北西住宅", 18:"病院東側", 27:"病院南側", 44:"中心市街地",
              55:"駅前", 62:"西部住宅", 73:"避難所周辺", 87:"南東住宅"}
SOURCES = ("事前情報のみ", "通報を追加", "通報＋センサ・現地測定")
STEPS = ("1. 支援要請の整理", "2. 観測・推定による見直し", "3. 実績条件での比較", "4. 観測設計の比較")

@dataclass(frozen=True)
class Report:
    cell: int
    kind: str
    text: str
    value: float | None = None
    sigma: float = .08
    count: int = 1
    time: str = "10:00"


def make_reports(city):
    """Synthetic evidence generated once; inference receives reports, never truth.

    Duplicate calls are displayed but a shared incident contributes only once.
    Numeric field measurement noise is separate from fixed sensor noise.
    """
    reports = [Report(55, "冠水の通報", "駅前の道路が冠水。水深・範囲は未確認", count=4),
               Report(18, "冠水の通報", "病院東側の出入口前で冠水。場所・時刻の分かる写真あり"),
               Report(62, "冠水の通報", "住宅地の交差点で冠水。周辺への広がりは未確認", count=2),
               Report(73, "支援要請", "避難所周辺の排水支援要請。水深の記載なし"),
               Report(44, "支援要請", "中心市街地から排水支援要請。水深の記載なし")]
    for cell, error in ((27, .025), (62, -.035)):
        value = round(max(0., float(city.truth[cell]) + error), 2)
        reports.append(Report(cell, "現地測定", f"担当者による水深測定：約{value:.2f} m", value, .06))
    return reports


def estimate_sources(xy, prior, reports, sensors, values, sigma, source):
    """Spatial prior + noisy numeric evidence + one-sided qualitative evidence.

    'Flooded' supplies only a soft positive-depth constraint, not a guessed depth.
    A small 0.02 m detection threshold is an explicit teaching assumption.
    Source coverage is descriptive, NOT a calibrated confidence probability.
    """
    from scipy.optimize import minimize
    prior = np.asarray(prior, dtype=float)
    if source == SOURCES[0]:
        return prior.copy(), np.zeros(len(prior), dtype=int)
    numeric = []
    qualitative = sorted({r.cell for r in reports if r.kind == "冠水の通報"})
    if source == SOURCES[2]:
        numeric += [(r.cell, r.value, r.sigma) for r in reports if r.kind == "現地測定"]
        numeric += [(int(i), float(v), max(float(sigma), .005)) for i,v in zip(sensors, values)]
    d2 = np.sum((xy[:,None]-xy[None,:])**2, axis=2)
    covariance = .18**2 * np.exp(-d2/(2*1.8**2)) + np.eye(len(prior))*1e-6
    chol = np.linalg.cholesky(covariance)
    ids = np.array([n[0] for n in numeric], dtype=int)
    vals = np.array([n[1] for n in numeric])
    sd = np.array([n[2] for n in numeric])
    q = np.array(qualitative, dtype=int)
    def objective(z):
        h = prior + chol @ z
        residual = (h[ids]-vals)/sd
        # An inequality likelihood; reports do not force a precise 0.02 m depth.
        below = np.minimum(h[q]-.02, 0.)/.06
        score = .5*(z@z + residual@residual + below@below)
        grad = z + chol[ids].T@(residual/sd) + chol[q].T@(below/.06)
        return score, grad
    fit = minimize(objective, np.zeros(len(prior)), jac=True, method="L-BFGS-B",
                   options={"maxiter":1000,"ftol":1e-11,"gtol":1e-7})
    if not fit.success:
        raise RuntimeError("状態推定の収束を確認できませんでした: " + fit.message)
    direct = sorted(set(qualitative + ids.tolist()))
    coverage = np.zeros(len(prior), dtype=int)
    if direct:
        coverage[np.min(d2[:,direct],axis=1) <= 1.8**2] = 1
        coverage[direct] = 2
    return np.maximum(0, prior+chol@fit.x), coverage


def map_figure(city, field, reports, sensors, values, pumps, coverage=None):
    import plotly.graph_objects as go
    fig = go.Figure()
    if coverage is not None:
        z=coverage.reshape(N,N); scale=[[0,"#e5e7eb"],[.49,"#e5e7eb"],[.5,"#bfdbfe"],[.99,"#bfdbfe"],[1,"#2563eb"]]
        fig.add_trace(go.Heatmap(z=z, zmin=0,zmax=2,colorscale=scale,
            colorbar=dict(tickvals=[0,1,2],ticktext=["主に事前情報","周辺に直接情報","直接情報あり"]),
            hovertemplate="列%{x}・行%{y}<extra></extra>"))
    else:
        z=np.full((N,N),np.nan) if field is None else np.asarray(field).reshape(N,N)
        fig.add_trace(go.Heatmap(z=z,zmin=0,zmax=1,colorscale="Blues",xgap=2,ygap=2,
            colorbar=dict(title="浸水深 m"), hovertemplate="浸水深 %{z:.3f} m<extra></extra>"))
    for r in reports:
        shown = f"{r.time} {r.text}（{r.count}件）"
        fig.add_trace(go.Scatter(x=[r.cell%N],y=[r.cell//N],mode="markers",
            marker=dict(symbol="diamond-open" if r.kind!="現地測定" else "square",size=15,color="#9a3412",line=dict(width=2)),
            text=[shown],hovertemplate="%{text}<extra></extra>",showlegend=False))
    if len(sensors):
        fig.add_trace(go.Scatter(x=np.asarray(sensors)%N,y=np.asarray(sensors)//N,mode="markers+text",
            text=[f"{v:.2f}" for v in values],textposition="top center",textfont=dict(color="#713f12"),
            marker=dict(size=8,color="#eab308"),name="センサ",hovertemplate="センサ %{text} m<extra></extra>"))
    for i,name in CANDIDATES.items():
        fig.add_trace(go.Scatter(x=[i%N],y=[i//N],mode="markers+text",text=[cell_name(i)],textposition="bottom center",
            marker=dict(symbol="square-open",size=24,color="#be123c" if i in pumps else "#334155",line=dict(width=3 if i in pumps else 1)),
            hovertext=[f"{name}｜重要度 {city.weight[i]:.2f}"],hovertemplate="%{hovertext}<extra></extra>",showlegend=False))
    fig.update_layout(height=460,margin=dict(l=10,r=10,t=5,b=5),plot_bgcolor="#e5e7eb",showlegend=False,
        xaxis=dict(range=[-.5,9.5],tickvals=list(range(10)),ticktext=list(range(1,11)),fixedrange=True),
        yaxis=dict(range=[9.5,-.5],tickvals=list(range(10)),ticktext=list("ABCDEFGHIJ"),scaleanchor="x",fixedrange=True))
    return fig


def report_rows(reports):
    return [{"地区":CANDIDATES[r.cell],"時刻":r.time,"種類":r.kind,"内容":r.text,
             "件数":r.count,"推定への利用":"水深の数値（誤差あり）" if r.kind=="現地測定" else "冠水の条件のみ" if r.kind=="冠水の通報" else "要請として表示"} for r in reports]


def main():
    import streamlit as st
    st.set_page_config(page_title="Mini Urban Flood Digital Twin",layout="wide")
    st.markdown("""<style>.block-container{padding-top:1.5rem;max-width:1500px}h1{font-size:1.7rem!important}
    [data-testid=stMetricValue]{font-size:1.6rem}</style>""",unsafe_allow_html=True)
    st.title("Mini Urban Flood Digital Twin")
    st.caption("複数の排水支援要請に対する優先配分 ｜ 同じ10:00時点の情報を順に追加")
    with st.sidebar:
        scenario=st.selectbox("豪雨ケース",[0,1,2],format_func=lambda i:["基本：局地豪雨","病院周辺の雨が弱い","病院周辺の雨が強い"][i],key="new_scenario")
    if st.session_state.get("case_id") != scenario:
        for key in ("initial_plan","initial_prediction","review_plan","review_prediction","review_config","design_before"):
            st.session_state.pop(key,None)
        st.session_state["allocation"]=[]
        st.session_state["flow"]=STEPS[0]
        st.session_state["open_results"]=False
        st.session_state["case_id"]=scenario
    stage=st.radio("進行",STEPS,horizontal=True,key="flow",label_visibility="collapsed")
    with st.sidebar:
        st.caption("設置・排水が可能な８地区から３地区を選択")
        count=st.slider("センサ数",3,30,3,key="new_count")
        strategy=st.selectbox("配置戦略",STRATEGIES,key="new_strategy")
        sigma=st.selectbox("センサ誤差 σ（m）",[0.,.03,.10,.20],index=1,key="new_sigma")
        st.caption("条件変更後も手動の配分を保持。推定案の反映はボタンで実行。")
    city=make_city(scenario); reports=make_reports(city)
    sensors=sensor_order(city,strategy)[:count]; values=observe(city,sensors,sigma)
    if stage==STEPS[0]:
        source=SOURCES[1]
        visible_reports=[r for r in reports if r.kind!="現地測定"]
    else:
        source=st.radio("推定に使う情報",SOURCES,index=2,horizontal=True,key="source_set")
        visible_reports=[] if source==SOURCES[0] else [r for r in reports if r.kind!="現地測定" or source==SOURCES[2]]
    used_sensors=sensors if source==SOURCES[2] else np.array([],dtype=int)
    used_values=values if source==SOURCES[2] else np.array([])
    prediction,coverage=estimate_sources(city.xy,city.prior,reports,sensors,values,sigma,source)
    config=(scenario,count,strategy,sigma,source)
    if stage==STEPS[0]:
        st.info("通報・支援要請と地域の重要度を確認し、当初の配分を選択。未把握の地区も比較対象。")
    else:
        st.caption("青の濃さは推定水深。推定の確かさを表す色ではない。通報なしを水深ゼロとして扱わない。")
    left,right=st.columns([1.6,1],gap="large")
    with right:
        st.subheader("支援候補地区への配分")
        if stage!=STEPS[0] and st.button("推定に基づく配分案を反映"):
            st.session_state["allocation"]=dispatch(prediction,city.weight).tolist()
        pumps=st.multiselect("３地区を選択",list(CANDIDATES),format_func=lambda i:f"{cell_name(i)} {CANDIDATES[i]}",max_selections=3,key="allocation")
        rows=[{"地区":f"{cell_name(i)} {name}","重要度":round(float(city.weight[i]),1),
               **({"推定水深 m":round(float(prediction[i]),2),"推定軽減量":round(float(gains(prediction,city.weight)[i]),2)} if stage!=STEPS[0] else {})} for i,name in CANDIDATES.items()]
        st.dataframe(rows,hide_index=True,width="stretch")
        if stage==STEPS[0]:
            if st.button("当初案を保存",disabled=len(pumps)!=3,type="primary"):
                st.session_state["initial_plan"]=list(pumps)
                st.session_state["initial_prediction"]=prediction.copy()
                st.session_state.pop("review_plan",None)
                st.success("当初案を保存")
        else:
            if st.button("見直し案を保存",disabled=len(pumps)!=3 or "initial_plan" not in st.session_state,type="primary"):
                st.session_state["review_plan"]=list(pumps)
                st.session_state["review_prediction"]=prediction.copy()
                st.session_state["review_config"]=config
                st.success("見直し案を保存")
        if "initial_plan" in st.session_state:
            st.caption("当初案："+" / ".join(CANDIDATES[i] for i in st.session_state["initial_plan"]))
        else:
            st.caption("第１段階で３地区を選び、当初案を保存")
    revealed=False
    if stage in (STEPS[2],STEPS[3]):
        revealed=st.toggle("実際の浸水状況と評価を開示",key="open_results")
    with left:
        views=["通報・観測情報"] if stage==STEPS[0] else ["推定された浸水状況","通報・観測情報","情報の所在"]
        if revealed: views.append("実際の浸水状況")
        view=st.radio("地図表示",views,horizontal=True,key="map_"+str(STEPS.index(stage))+str(revealed),label_visibility="collapsed")
        field=city.truth if view=="実際の浸水状況" else prediction if view=="推定された浸水状況" else None
        st.plotly_chart(map_figure(city,field,visible_reports,used_sensors,used_values,pumps,coverage if view=="情報の所在" else None),width="stretch",key="map")
        st.caption("枠：支援候補地区／赤枠：選択地区／茶色のひし形：通報・要請／茶色の四角：現地測定／黄色：センサ")
        if view=="情報の所在":
            st.caption("直接情報のある地点と、その近傍（1.8区画以内）を区別した表示。信頼確率や推定誤差の表示ではない。支援要請だけの地点は直接の水深情報に含めない。")
    with st.expander("通報・現地報告の内容",expanded=stage==STEPS[0]):
        st.dataframe(report_rows(visible_reports),hide_index=True,width="stretch")
        st.caption("重複通報は一つの事象として推定に利用。支援要請の件数を水深や重要度に変換しない。")
    if stage==STEPS[2]:
        if all(k in st.session_state for k in ("initial_plan","review_plan")):
            if revealed:
                a=evaluate(city,st.session_state["initial_plan"]); b=evaluate(city,st.session_state["review_plan"])
                c=st.columns(3)
                c[0].metric("当初案の被害軽減",f"{a['avoided']:.2f}")
                c[1].metric("見直し案の被害軽減",f"{b['avoided']:.2f}",f"{b['avoided']-a['avoided']:+.2f}")
                err=float(np.sqrt(np.mean((st.session_state['review_prediction']-city.truth)**2)))
                c[2].metric("保存した見直し時の推定RMSE",f"{err:.3f} m")
                st.caption("保存時の推定条件：" + str(st.session_state["review_config"][1:]))
                st.caption("同じ豪雨・候補地区・３台で評価。保存した案の比較（現在編集中の選択とは別）。見直し案："+" / ".join(CANDIDATES[i] for i in st.session_state["review_plan"]))
                with st.expander("完全情報下の最良配分との比較"):
                    st.write({"当初案との差":round(a['regret'],3),"見直し案との差":round(b['regret'],3)})
        else: st.info("当初案と見直し案を保存すると比較が可能")
    if stage==STEPS[3]:
        st.subheader("観測設計の比較")
        target=st.radio("変更する条件",["センサ数","配置戦略","センサ誤差"],horizontal=True)
        signature=(scenario,source,target,*(v for j,v in enumerate((count,strategy,sigma)) if j!=["センサ数","配置戦略","センサ誤差"].index(target)))
        if st.session_state.get("design_signature")!=signature:
            st.session_state.pop("design_before",None)
        st.session_state["design_signature"]=signature
        if source!=SOURCES[2]: st.info("センサ条件の効果を見る場合は「通報＋センサ・現地測定」を選択")
        if st.button("現在の観測設計を保存"):
            st.session_state["design_before"]=(config,prediction.copy())
        before=st.session_state.get("design_before")
        if before:
            rows=[]
            for label,cfg,h in (("変更前",*before),("変更後",config,prediction)):
                chosen=dispatch(h,city.weight)
                row={"条件":label,"センサ数":cfg[1],"配置":cfg[2],"誤差σ":cfg[3],"推定からの配分案":" / ".join(CANDIDATES[i] for i in chosen)}
                if revealed: row.update({"RMSE m":round(float(np.sqrt(np.mean((h-city.truth)**2))),3),"実際の被害軽減":round(evaluate(city,chosen)['avoided'],3)})
                rows.append(row)
            st.dataframe(rows,hide_index=True,width="stretch")
            for col, label, cfg, h in zip(st.columns(2), ("変更前", "変更後"), (before[0], config), (before[1], prediction)):
                with col:
                    st.caption(label + "の推定と配分案")
                    ss = sensor_order(city,cfg[2])[:cfg[1]] if cfg[4]==SOURCES[2] else np.array([],dtype=int)
                    vv = observe(city,ss,cfg[3])
                    rr = [] if cfg[4]==SOURCES[0] else [r for r in reports if r.kind!="現地測定" or cfg[4]==SOURCES[2]]
                    st.plotly_chart(map_figure(city,h,rr,ss,vv,dispatch(h,city.weight)),width="stretch",key="compare_"+label)
            st.caption("比較対象以外の条件を変えると保存を解除。通報・候補地区・豪雨は前後で共通。各条件の推定から自動で配分案を計算。")
    with st.expander("モデルと情報の読み方"):
        st.markdown("""- 冠水通報は「その場所で水がたまっている」という片側の条件として利用。検出の目安を0.02 mとする教育用の仮定であり、通報を特定の水深へ置換しない。既に事前推定がこの条件を満たす場合、補正が生じないこともある。
- 数値のある現地測定は誤差0.06 m、センサは設定した誤差として利用。地形・排水の事前分布を空間的関係で補正。観測値と推定値の完全一致は保証しない。
- 全情報は10:00を対象とする設定。到着までの予測・道路移動・水の流動は今回の計算対象外。
- 各候補地区は代表１区画。１台で代表区画の水深を最大0.25 m低減。周辺全体への排水効果は計算しない。
- 被害＝重要度 × max(水深−0.10 m, 0) の総和（相対指標）。重要度は人の価値の点数ではなく、施設・生活への影響を簡略化した設定。
- 全域のRMSEと被害軽減は結果開示後だけ表示。真値は推定・候補地区選定には利用しない。通報・測定の生成と事後評価にのみ利用。
- 教育用の仮想モデル。実際の災害対応の判断には使用しない。""")


if __name__ == "__main__":
    main()
