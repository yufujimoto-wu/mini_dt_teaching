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
STAGES = ("1｜真値と配置", "2｜少数の観測", "3｜DTで推定", "4｜センサ数", "5｜配置戦略")


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
    return np.argsort(-gains(depth, weight), kind="stable")[:PUMP_COUNT]


def evaluate(city, pumps):
    pumps = np.asarray(pumps, dtype=int)
    if len(pumps) > PUMP_COUNT or len(set(pumps.tolist())) != len(pumps):
        raise ValueError("ポンプは異なる地点に最大3台です")
    if np.any((pumps < 0) | (pumps >= N*N)):
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
    prediction = estimate(city, sensors, values)
    pumps = dispatch(prediction, city.weight)
    result = evaluate(city, pumps)
    result.update(sensors=sensors, values=values, prediction=prediction, pumps=pumps,
                  rmse=float(np.sqrt(np.mean((prediction-city.truth)**2))))
    return result


def cell_name(i):
    return f"{chr(65 + int(i)//N)}{int(i)%N + 1:02d}"


def toggle_pump(pumps, cell):
    """Toggle one valid cell, preserving the three-truck limit."""
    updated = list(pumps)
    if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < N*N:
        return updated
    if cell in updated:
        updated.remove(cell)
    elif len(updated) < PUMP_COUNT:
        updated.append(cell)
    return updated


def grid_payload(city, field, sensors, values, pumps, exposure=False):
    """Only visible data cross the iframe boundary; never send hidden truth."""
    from plotly.colors import sample_colorscale
    observations = {int(i): f"{v:.2f}" for i,v in zip(sensors,values)}
    palette = "YlOrBr" if exposure else "Blues"
    shown = city.weight if exposure else field
    maximum = 12 if exposure else 1
    colors = sample_colorscale(palette, np.clip(shown/maximum,0,1).tolist()) if shown is not None else ["#e8edf2"]*(N*N)
    cells = []
    for i in range(N*N):
        value = f"重要度 {city.weight[i]:.2f}" if exposure else f"水深 {shown[i]:.3f} m" if shown is not None else f"観測値 {observations[i]} m（真値は非表示）" if i in observations else "未観測・真値は非表示"
        description = f"{cell_name(i)} ｜ {value} ｜ 重要度 {city.weight[i]:.2f}"
        if i in observations:
            description += f" ｜ 観測 {observations[i]} m"
        cells.append(dict(name=cell_name(i), color=colors[i], important=bool(city.weight[i]>=6.0),
                          selected=i in pumps, observed=observations.get(i), description=description))
    ramp = sample_colorscale(palette,[0,.25,.5,.75,1])
    return dict(cells=cells,legend="重要度" if exposure else "浸水深",range="0–12" if exposure else "0–1 m",
                ramp="linear-gradient(to right, "+", ".join(ramp)+")")



def apply_grid_event(state, event):
    """Consume each click once; a rejected fourth truck does not change provenance."""
    if not isinstance(event, dict) or not event.get("token") or event["token"] == state.get("grid_event"):
        return False
    state["grid_event"] = event["token"]
    updated = toggle_pump(state["pumps"], event.get("cell"))
    if updated != state["pumps"]:
        state["pumps"] = updated
        state["dispatch_source"] = {"kind": "manual"}
    return True


def condition_label(conditions):
    scenario, count, strategy, sigma = conditions
    return f"{['基本：局地豪雨', '病院周辺の雨が弱い', '病院周辺の雨が強い'][scenario]} / {count}台 / {strategy} / σ={sigma:.2f} m"


def comparison_rows(before, after):
    """Use the same model evaluation for every displayed comparison value."""
    rows=[]
    for label, conditions in (("変更前",before),("変更後（現在）",after)):
        scenario,count,strategy,sigma=conditions
        r=run_design(make_city(scenario),count,strategy,sigma)
        rows.append({"時点":label,"センサ数":count,"配置戦略":strategy,"σ (m)":sigma,
                     "推定誤差 (m)":r["rmse"],"自動派遣先":" / ".join(cell_name(i) for i in r["pumps"]),
                     "減らせた被害":r["avoided"]})
    return rows


def comparison_message(before, after):
    if before[0] != after[0]:
        return "豪雨ケースも変わっています。観測条件の効果を比べるには同じ豪雨に戻すか、変更前を保存し直してください。"
    if tuple(before)==tuple(after):
        return "変更前と同じ条件です。サイドバーで比較対象の値を変えてください。"
    a,b=comparison_rows(before,after)
    error=b["推定誤差 (m)"]-a["推定誤差 (m)"]
    benefit=b["減らせた被害"]-a["減らせた被害"]
    error_text="推定誤差は減った" if error < -1e-9 else "推定誤差は増えた" if error > 1e-9 else "推定誤差は同じ"
    same=set(a["自動派遣先"].split(" / "))==set(b["自動派遣先"].split(" / "))
    dispatch_text="派遣先は同じ" if same else "派遣先が変わった"
    benefit_text=f"減らせた被害は{abs(benefit):.2f}増えた" if benefit > 1e-9 else f"減らせた被害は{abs(benefit):.2f}減った" if benefit < -1e-9 else "減らせた被害は同じ"
    return f"{error_text}。{dispatch_text}。{benefit_text}。"


def show_before_after(conditions):
    import streamlit as st
    import streamlit.components.v1 as components
    from pathlib import Path
    st.caption("① 現在の条件を変更前として保存 → ② サイドバーで条件を変更 → ③ 推定・派遣先・被害を比較")
    st.caption("ここでは前後それぞれのDT推定から3台を自動派遣して比較します。手動で保持している配置は変更しません。")
    before=st.session_state.get("comparison_before")
    if before is None:
        st.info("まず、比較の出発点にしたい条件をサイドバーで選び、「変更前」に保存してください。")
        st.caption("現在："+condition_label(conditions))
        return
    same_weather=before[0]==conditions[0]
    message=comparison_message(before,conditions)
    if same_weather:
        st.info(message)
    else:
        st.warning(message)
    rows=comparison_rows(before,conditions)
    grid=components.declare_component("urban_flood_grid",path=str(Path(__file__).parent/"flood_grid"))
    for col,row,config in zip(st.columns(2,gap="large"),rows,(before,conditions)):
        with col:
            st.subheader(row["時点"])
            st.caption(condition_label(config))
            c1,c2=st.columns(2)
            c1.metric("推定の誤差 ↓",f"{row['推定誤差 (m)']:.3f} m")
            c2.metric("減らせた被害 ↑",f"{row['減らせた被害']:.2f}")
            st.write("自動派遣先："+row["自動派遣先"])
            city=make_city(config[0])
            r=run_design(city,*config[1:])
            grid(**grid_payload(city,r["prediction"],r["sensors"],r["values"],r["pumps"]),
                 interactive=False,ack=None,max_width=310,key="comparison_"+row["時点"],default=None)
    st.caption("地図の色＝DT推定の浸水深（共通0〜1 m） / 黄色＝観測値 / 赤×＝その推定からの自動派遣先。被害削減は真値で評価。")
    with st.expander("前後の数値表・CSV"):
        st.dataframe(rows,hide_index=True,width="stretch")
        import csv,io
        buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=["豪雨ケース"]+list(rows[0]));writer.writeheader()
        for row,config in zip(rows,(before,conditions)):
            writer.writerow({"豪雨ケース":config[0],**row})
        st.download_button("前後比較CSV",buf.getvalue().encode("utf-8-sig"),"before_after.csv","text/csv")


def comparison_sidebar():
    import streamlit as st
    st.subheader("何を比較する？")
    target=st.radio("比較する項目",["センサ数","配置戦略"],index=1,key="cmp_target",label_visibility="collapsed")
    if target=="センサ数":
        count=st.slider("センサ数",3,30,12,key="cmp_count")
    else:
        strategy=st.selectbox("配置戦略",STRATEGIES,key="cmp_strategy")
    save_area=st.container()
    change_area=st.container()
    st.divider()
    st.subheader("前後で揃える条件")
    summary_area=st.container()
    with st.expander("共通条件を設定"):
        st.caption("共通条件を変更すると、保存した変更前をリセットします。")
        if target=="センサ数":
            strategy=st.selectbox("固定する配置戦略",STRATEGIES,key="cmp_strategy")
        else:
            count=st.slider("固定するセンサ数",3,30,12,key="cmp_count")
        scenario=st.selectbox("豪雨ケース",[0,1,2],format_func=lambda i:condition_label((i,0,"",0)).split(" / ")[0],key="cmp_scenario")
        sigma=st.selectbox("観測ノイズ σ（m）",[0.,.03,.10,.20],index=1,key="cmp_sigma")
    conditions=(scenario,count,strategy,sigma)
    signature=(target,scenario,sigma,strategy if target=="センサ数" else count)
    if st.session_state.get("comparison_signature",signature)!=signature:
        st.session_state.pop("comparison_before",None)
        st.session_state["comparison_reset_notice"]=True
    st.session_state["comparison_signature"]=signature
    with save_area:
        if st.button("現在の設定を「変更前」に保存",key="save_before",type="primary"):
            st.session_state["comparison_before"]=conditions
            st.session_state["comparison_reset_notice"]=False
    with change_area:
        before=st.session_state.get("comparison_before")
        if before is None:
            st.caption("変更前：未保存")
        else:
            st.caption("変更前："+(f"{before[1]}台" if target=="センサ数" else before[2]))
        st.caption("変更後："+(f"{count}台" if target=="センサ数" else strategy))
        if st.session_state.get("comparison_reset_notice"):
            st.caption("比較項目・共通条件が変わったため、変更前をリセットしました。保存し直してください。")
    with summary_area:
        st.caption(f"配置戦略：{strategy}" if target=="センサ数" else f"センサ数：{count}台")
        st.caption("豪雨ケース："+condition_label(conditions).split(" / ")[0])
        st.caption(f"観測ノイズ：σ = {sigma:.2f} m")
    return conditions


def main():
    import streamlit as st
    st.set_page_config(page_title="Mini Urban Flood DT",page_icon="🌧️",layout="wide",initial_sidebar_state="expanded")
    st.markdown("""<style>.block-container{padding-top:2rem;padding-bottom:.7rem;padding-left:2rem;padding-right:2rem;max-width:1450px}
    h1{font-size:1.6rem!important} [data-testid=stMetricValue]{font-size:1.4rem}
    [data-testid=stVerticalBlock]{gap:.55rem}</style>""",unsafe_allow_html=True)
    st.title("Mini Urban Flood Digital Twin")
    with st.sidebar:
        mode = st.radio("モード", ["派遣を考える", "観測設計を比較する"], key="mode")
    comparing = mode == "観測設計を比較する"
    st.subheader(mode)
    # Keep hidden widgets' selections when switching modes or introductory stages.
    for key in ("stage", "count", "sigma", "strategy", "reveal_2", "reveal_3", "reveal_4", "reveal_5", "scenario", "cmp_target", "cmp_count", "cmp_strategy", "cmp_scenario", "cmp_sigma"):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    if st.session_state.get("stage") not in (None, *STAGES):
        st.session_state["stage"] = STAGES[0]
    stage = None
    if not comparing:
        stage_label = st.radio("講義Stage", STAGES, horizontal=True, label_visibility="collapsed", key="stage")
        stage = STAGES.index(stage_label)+1
    observation_unlocked = comparing or stage >= 4
    strategy_unlocked = comparing or stage >= 5
    prompts = ["すべて見えるなら、ポンプ3台をどこへ？ 深い地点と守る価値の高い地点は同じ？",
               "見えるのは3地点だけ。観測のない地区を、どう判断する？",
               "観測のない場所は推定。滑らかな地図は、正しさを保証する？",
               "配置の順番を固定してセンサを追加。推定を見て、派遣先を見直す？",
               "同じ台数でも、都市全体を見るか、重要地区を見るか？"]
    if not comparing:
        st.info(prompts[stage-1])
    with st.sidebar:
        st.divider()
        if comparing:
            scenario,count,strategy,sigma=comparison_sidebar()
        else:
            st.subheader("観測できる情報")
            count=st.slider("センサ数",3,30,12 if observation_unlocked else 3,key="count" if observation_unlocked else "count_intro",disabled=not observation_unlocked)
            strategy=st.selectbox("配置戦略",STRATEGIES,key="strategy" if strategy_unlocked else "strategy_intro",disabled=not strategy_unlocked)
            with st.expander("豪雨・観測精度の設定"):
                scenario=st.selectbox("豪雨ケース",[0,1,2],format_func=lambda i:condition_label((i,0,"",0)).split(" / ")[0],key="scenario")
                sigma=st.selectbox("観測ノイズ σ（m）",[0.,.03,.10,.20],index=1,key="sigma" if observation_unlocked else "sigma_intro",disabled=not observation_unlocked)
            if not observation_unlocked:
                count,sigma=3,.03
                st.caption("Stage 1〜3：3台・均等カバー・σ = 0.03 m")
            elif not strategy_unlocked:
                st.caption("Stage 4：均等カバーで台数を比較")
            if not strategy_unlocked:
                strategy=STRATEGIES[0]
        st.divider()
        st.caption("ポンプ車：3台\n\n1台で1区画の水深を最大0.25 m低減")
    city = make_city(scenario)
    result = run_design(city,count,strategy,sigma)
    conditions = (scenario, count, strategy, sigma)
    st.session_state.setdefault("pumps", [])
    st.session_state.setdefault("dispatch_source", {"kind": "manual"})
    pumps = st.session_state["pumps"]
    source = st.session_state["dispatch_source"]
    if comparing:
        show_before_after(conditions)
        return
    left, right = st.columns([3.2,1],gap="large")
    with right:
        st.subheader("派遣・結果")
        st.write(f"**派遣 {len(pumps)} / {PUMP_COUNT} 台**")
        st.caption(" / ".join(cell_name(i) for i in pumps) or "派遣先は未選択")
        if source["kind"] == "manual":
            st.caption("自分で選択した派遣結果" if pumps else "未選択：地図をクリックして配置")
        elif source["kind"] == "oracle":
            st.caption("Oracleから自動派遣した配置")
            st.caption("選択時：" + condition_label(source["conditions"]).split(" / ")[0])
            if source["conditions"][0] != scenario:
                st.caption("豪雨ケース変更前の配置を保持中")
        else:
            st.caption("DT推定から自動派遣した配置")
            st.caption("選択時：" + condition_label(source["conditions"]))
            if tuple(source["conditions"]) != conditions:
                st.caption("条件変更前の配置を保持中。再計算は下のボタンから。")
        if stage >= 3:
            st.caption("現在の観測条件：" + condition_label(conditions))
        if stage==1 and st.button("Oracleの最適配置",key="oracle"):
            st.session_state["pumps"] = dispatch(city.truth,city.weight).tolist()
            st.session_state["dispatch_source"] = {"kind":"oracle", "conditions":conditions}
            st.rerun()
        if stage>=3 and st.button("DT推定から自動派遣",key="auto_dispatch"):
            st.session_state["pumps"] = result["pumps"].tolist()
            st.session_state["dispatch_source"] = {"kind":"dt", "conditions":conditions}
            st.rerun()
        if st.button("選択をクリア",key="clear"):
            st.session_state["pumps"] = []
            st.session_state["dispatch_source"] = {"kind":"manual"}
            st.rerun()
        if len(pumps)==PUMP_COUNT:
            st.caption("地図で1台解除してから別の区画を選択")
        revealed = stage == 1
        if stage in (2,3,4,5):
            revealed = st.toggle("結果を開示（教員・事後評価）",key=f"reveal_{stage}")
        evaluation = evaluate(city,pumps)
        if revealed:
            st.metric("実現した被害削減",f"{evaluation['avoided']:.2f}")
            st.metric("Oracleとの差 ↓",f"{evaluation['regret']:.2f}")
            st.caption(f"対策なし {evaluation['baseline']:.2f} → 対策後 {evaluation['loss']:.2f} ｜ Oracle削減 {evaluation['oracle_avoided']:.2f}")
            if stage>=3:
                st.caption(f"全100地点の推定RMSE：{result['rmse']:.3f} m（事後の真値で評価）")
                prior_result = evaluate(city, dispatch(city.prior, city.weight))
                st.caption(f"観測なしの派遣と比べた追加削減：{evaluation['avoided'] - prior_result['avoided']:+.2f}（この豪雨での実現値）")
        else:
            st.caption("評価は非表示。派遣を考えてから「結果を開示」。")
    with left:
        options=["真の浸水","重要度"] if stage==1 else (["観測のみ","重要度"] if stage==2 else ["DT推定","観測のみ","重要度"])
        if revealed and stage!=1:
            options.append("真の浸水（事後）")
        view=st.radio("地図",options,horizontal=True,key=f"view_{stage}",label_visibility="collapsed")
        field=city.truth if view.startswith("真") else result["prediction"] if view=="DT推定" else None
        sensors=result["sensors"] if stage!=1 else np.array([],dtype=int)
        values=result["values"] if stage!=1 else np.array([])
        from pathlib import Path
        import streamlit.components.v1 as components
        grid = components.declare_component("urban_flood_grid",path=str(Path(__file__).parent/"flood_grid"))
        ack = st.session_state.get("grid_event")
        event = grid(**grid_payload(city,field,sensors,values,pumps,exposure=view=="重要度"),
                     interactive=True,ack=ack,key="shared_grid",default=None)
        if apply_grid_event(st.session_state,event):
            st.rerun()
        st.caption("橙枠：重要度6以上。重要度は市街地・病院周辺・住宅地に緩やかに分布。黄色の数値：観測、赤×：派遣。灰色は未観測。深さの色尺度は0–1 mで固定。")
    st.caption("教育用の仮想モデル ｜ 各ポンプは担当1区画の水深を最大0.25 m低減。被害＝重要度 × max(水深 − 0.10 m, 0) の総和（相対指標）。")
    with st.expander("モデル・比較の読み方"):
        st.markdown("""**Reality → Observation → Estimation → Decision → Outcome**

- 既知：地形・排水能力の基準値・重要度・予報に基づく事前推定。未知：実際の局地雨・排水偏差。
- 観測残差を距離に応じて補間し、事前推定を補正。未観測＝水深ゼロではありません。
- 自動派遣は推定上の被害削減が大きい3区画。評価には常に実際の浸水深を使います。
- Oracleとの差＝真値で最適に派遣した削減量 − 実際の派遣の削減量。小さいほどよい判断です。
- **Best Observation for State Estimation ≠ Best Observation for Decision Making**：ここで比べるのは3つの候補戦略です。センサ配置の大域的最適解を求めた結果ではありません。
- **Accuracy ≠ Decision Value**：RMSEは全地区を等しく扱います。判断では地区の重要度と、ポンプによる削減可能量が効きます。
- この1ケースの被害削減は事後の実現値です。期待Value of Informationや導入費用を含む純便益ではありません。複数事象での検証は次の学習課題です。
- 水の移動、下水網、ポンプ移動時間、再浸水は省略。実際の防災計画には使用できません。""")


if __name__ == "__main__":
    main()
