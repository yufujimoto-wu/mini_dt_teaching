"""Teaching-only urban flood DT. Run: streamlit run app.py.

Pure model functions intentionally do not import Streamlit or read session state.
"""
from dataclasses import dataclass
import numpy as np

N = 10
PUMP_COUNT = 3
PUMP_DROP = 0.25  # m, upper bound per cell over a fixed intervention interval
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
    known_drainage = .10 * np.exp(-((x-1.5)**2+(y-6.5)**2)/14)
    prior = np.maximum(0, 0.20 + ponding - drainage-known_drainage)
    # Synthetic event: unobserved eastern flooding and better-than-expected western drainage.
    broad = .38 * np.exp(-((x-7.0)**2+(y-5.5)**2)/9)
    local = (.32, .08, .48)[scenario] * np.exp(-((x-7.7)**2+(y-1.3)**2)/3)
    relief = .14 * np.exp(-((x-1.5)**2+(y-6.5)**2)/14)
    truth = np.maximum(0, prior+broad+local-relief)
    noise = np.random.default_rng(2026 + scenario).normal(size=N*N)
    return City(xy, elevation, weight, prior, truth, noise)


def sensor_order(city, strategy):
    """Nested layouts based ONLY on pre-event known data; never truth."""
    dist = np.sum((city.xy[:, None] - city.xy[None, :])**2, axis=2)
    if strategy == STRATEGIES[0]:
        priority = np.ones(N*N)
        first = 55
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
    cell_gains=damage(depth, weight) - damage(np.maximum(depth - PUMP_DROP, 0), weight)
    result=np.zeros(N*N)
    for anchor,cells in AREAS.items(): result[anchor]=cell_gains[cells].sum()
    return result


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
    post = after_pumping(city.truth,pumps)
    baseline = float(damage(city.truth, city.weight).sum())
    loss = float(damage(post, city.weight).sum())
    oracle = dispatch(city.truth, city.weight)
    best = float(gains(city.truth, city.weight)[oracle].sum())
    return dict(baseline=baseline, loss=loss, avoided=baseline-loss,
                regret=max(0.0, best - (baseline-loss)), oracle_avoided=best)


def run_design(city, count, strategy, sigma):
    sensors = sensor_order(city, strategy)[:count]
    values = observe(city, sensors, sigma)
    prediction, _ = estimate_sources(city.xy, city.prior, make_reports(city), sensors, values, sigma, "センサ＋通報")
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
# Fixed, disjoint drainage service areas, independent of observations and truth.
def rectangle(y0,y1,x0,x1):
    return [y*N+x for y in range(y0,y1) for x in range(x0,x1)]
AREAS = {
    12: rectangle(0,4,0,4),
    18: rectangle(0,3,8,10),
    27: rectangle(0,4,6,8)+rectangle(3,4,8,10),
    44: rectangle(0,4,4,6)+rectangle(4,6,3,5),
    55: rectangle(4,7,5,10),
    62: rectangle(4,7,0,3)+rectangle(6,7,3,5),
    73: rectangle(7,10,0,5),
    87: rectangle(7,10,5,10),
}
AREA_OWNER = {cell: anchor for anchor,cells in AREAS.items() for cell in cells}


def after_pumping(depth, pumps):
    ids=list(map(int,pumps))
    if len(ids)>PUMP_COUNT or len(set(ids))!=len(ids) or any(i not in AREAS for i in ids):
        raise ValueError("異なる候補地区を最大3地区まで選択")
    post=np.asarray(depth,dtype=float).copy()
    for anchor in ids:
        region=AREAS[anchor]
        post[region]=np.maximum(0,post[region]-PUMP_DROP)
    return post


SOURCES = ("事前情報のみ", "センサ情報を追加", "センサ＋通報")

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
    """Synthetic reports; duplicate calls describe one event, not independent evidence."""
    reported_depth = round(max(0., float(city.truth[18]) + .035), 1)
    return [Report(55, "冠水の通報", "駅前の道路が冠水。水深・範囲は未確認", count=4),
            Report(18, "水深の通報", f"病院東側の出入口前で約{reported_depth:.1f} mの冠水との通報。目測のため誤差を含む", value=reported_depth, sigma=.08),
            Report(62, "冠水の通報", "住宅地の交差点で冠水。周辺への広がりは未確認", count=2),
            Report(73, "支援要請", "避難所周辺の排水支援要請。水深の記載なし"),
            Report(44, "支援要請", "中心市街地から排水支援要請。水深の記載なし")]


def estimate_sources(xy, prior, reports, sensors, values, sigma, source):
    """Spatial prior + noisy numeric evidence + one-sided qualitative evidence.

    'Flooded' supplies only a soft positive-depth constraint, not a guessed depth.
    A small 0.02 m detection threshold is an explicit teaching assumption.
    Source coverage is descriptive, NOT a calibrated confidence probability.
    """
    prior = np.asarray(prior, dtype=float)
    if source == SOURCES[0]:
        return prior.copy(), np.zeros(len(prior), dtype=int)
    numeric = [(int(i), float(v), max(float(sigma), .005)) for i,v in zip(sensors, values)]
    qualitative = sorted({r.cell for r in reports if r.kind == "冠水の通報"}) if source == SOURCES[2] else []
    if source == SOURCES[2]:
        numeric += [(r.cell, r.value, r.sigma) for r in reports if r.value is not None]
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
    # Convex piecewise-quadratic objective: damped Newton with active
    # one-sided reports. NumPy only, including the linear solve.
    z = np.zeros(len(prior))
    numeric_design = chol[ids] / sd[:, None]
    base_hessian = np.eye(len(prior)) + numeric_design.T @ numeric_design
    for _ in range(100):
        score, grad = objective(z)
        if np.linalg.norm(grad, ord=np.inf) < 1e-7:
            break
        active = q[(prior + chol @ z)[q] < .02]
        qualitative_design = chol[active] / .06
        hessian = base_hessian + qualitative_design.T @ qualitative_design
        direction = np.linalg.solve(hessian, -grad)
        slope = float(grad @ direction)
        step = 1.0
        for _ in range(50):
            candidate = z + step * direction
            candidate_score, _ = objective(candidate)
            if candidate_score <= score + 1e-4 * step * slope + 1e-14:
                z = candidate
                break
            step *= .5
        else:
            raise RuntimeError("状態推定の補正量を計算できませんでした")
    else:
        raise RuntimeError("状態推定の収束を確認できませんでした")
    direct = sorted(set(qualitative + ids.tolist()))
    coverage = np.zeros(len(prior), dtype=int)
    if direct:
        coverage[np.min(d2[:,direct],axis=1) <= 1.8**2] = 1
        coverage[direct] = 2
    return np.maximum(0, prior+chol@z), coverage



VIEWS = ("重要度", "観測・通報による浸水情報", "推定された浸水深", "実際の浸水深（参照）")


def allocation_event(state, event):
    """Acknowledge every event exactly once, validate candidate and scene."""
    if not isinstance(event, dict) or not isinstance(event.get("token"), str):
        return False
    if state.get("click_ack") == event["token"]:
        return False
    state["click_ack"] = event["token"]
    if event.get("scene") != state.get("ui_scene",state.get("ui_case")):
        return True
    cell = event.get("cell")
    if state.get("map_mode")=="追加センサを置く" and state.get("ui_source")!=SOURCES[0]:
        if isinstance(cell,bool) or not isinstance(cell,int) or not 0<=cell<N*N:
            return True
        if cell in state.get("base_sensors",[]):
            state["sensor_note"]="既設センサのある地点。他のマスを選択"
        else:
            state["extra_sensor"]=None if state.get("extra_sensor")==cell else cell
            state["sensor_note"]=""
        return True
    if isinstance(cell, bool) or not isinstance(cell, int) or cell not in CANDIDATES:
        return True
    selected = list(state.get("picked", []))
    if cell in selected:
        selected.remove(cell)
        state["selection_note"] = ""
    elif len(selected) < PUMP_COUNT:
        selected.append(cell)
        state["selection_note"] = ""
    else:
        state["selection_note"] = "３地区を選択済み。選択中の地区をクリックして解除"
    state["picked"] = selected
    return True


def outcome(depth, weight, pumps):
    """All decision KPIs use the chosen evaluation basis; no hidden truth."""
    post=after_pumping(depth,pumps)
    before=damage(depth,weight); after=damage(post,weight)
    total=float(before.sum()); avoided=float((before-after).sum())
    return dict(avoided=avoided, remaining=float(after.sum()),
                rate=100*avoided/total if total else 0.,
                important=float((before-after)[weight>=6].sum()),
                residual=int(np.sum(post>THRESHOLD)),
                dry=int(np.sum((np.asarray(depth)>THRESHOLD)&(post<=THRESHOLD))),
                total=total)


def grid_data(city, prediction, coverage, reports, sensors, values, pumps, view):
    from plotly.colors import sample_colorscale
    numeric={int(i):float(v) for i,v in zip(sensors,values)}
    by_cell={}
    for r in reports: by_cell.setdefault(r.cell,[]).append(r)
    shown=city.weight if view==VIEWS[0] else prediction if view==VIEWS[2] else city.truth if view==VIEWS[3] else None
    palette="YlOrBr" if view==VIEWS[0] else "Blues"
    upper=12 if view==VIEWS[0] else 1
    colors=sample_colorscale(palette,np.clip(shown/upper,0,1).tolist()) if shown is not None else ["#e5e7eb"]*100
    cells=[]
    for i in range(100):
        rr=by_cell.get(i,[])
        name=CANDIDATES.get(i,"")
        owner=AREA_OWNER[i]
        region=AREAS[owner]
        detail=f"{cell_name(i)} ｜ {CANDIDATES[owner]}の排水区域（{len(region)}区画） ｜ 重要度 {city.weight[i]:.1f}"
        if shown is not None and view!=VIEWS[0]: detail+=f" ｜ {view} {shown[i]:.2f} m"
        if i in numeric: detail+=f" ｜ センサ {numeric[i]:.2f} m（10:00）"
        detail+="".join(f" ｜ {r.kind} {r.count}件（{r.time}）：{r.text}" for r in rr)
        label=""
        if view==VIEWS[1]:
            report_value=next((r.value for r in rr if r.value is not None),None)
            value=numeric.get(i,report_value)
            if value is not None:
                colors[i]=sample_colorscale("Blues",[float(np.clip(value,0,1))])[0]
                label=f"{value:.2f}" if i in numeric else f"約{value:.1f}"
            elif any(r.kind=="冠水の通報" for r in rr):
                colors[i]="repeating-linear-gradient(135deg,#dbeafe 0px,#dbeafe 4px,#7da5c7 4px,#7da5c7 6px)"
            else:
                detail+=" ｜ 浸水深の情報なし"
        cells.append(dict(name=cell_name(i),district=name,color=colors[i],candidate=i in CANDIDATES,
                          selected=i in pumps,sensor=i in numeric,report=bool(rr),label=label,description=detail,
                          owner=owner,affected=owner in pumps,
                          edges=[i<N or AREA_OWNER.get(i-N)!=owner, i%N==N-1 or AREA_OWNER.get(i+1)!=owner,
                                 i>=N*(N-1) or AREA_OWNER.get(i+N)!=owner, i%N==0 or AREA_OWNER.get(i-1)!=owner]))
    ramp=sample_colorscale(palette,[0,.25,.5,.75,1])
    return dict(cells=cells,legend="重要度 0–12" if view==VIEWS[0] else "浸水深 0–1 m",
                ramp="linear-gradient(to right,"+",".join(ramp)+")",observation=view==VIEWS[1])


GRID_HTML = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;color:#243247;font:13px system-ui,sans-serif;background:transparent}
.wrap{max-width:390px;margin:auto}#grid{display:grid;grid-template-columns:19px repeat(10,minmax(0,1fr));gap:2px}
.axis{display:flex;align-items:center;justify-content:center;color:#64748b;font-size:11px}
.cell{position:relative;aspect-ratio:1;border:1px solid #dbe2e8;background:#eee;border-radius:3px;padding:1px;min-width:0;color:#172a41;overflow:hidden}
.cell.candidate,.cell.sensor-target{cursor:pointer}.cell.extra-sensor{outline:2px dashed #2563eb;outline-offset:-4px}
.cell.has-report{box-shadow:inset 0 0 0 3px #888}
.cell.has-sensor:before{content:"";position:absolute;inset:0;border:1px solid #111;pointer-events:none;z-index:2}
.cell.selected:after{content:"";position:absolute;inset:4px;border:2px solid #be123c;pointer-events:none;z-index:3}
.area-edge{position:absolute;inset:0;pointer-events:none;z-index:1;border-color:#bd4964;border-style:solid;opacity:0}.cell.affected .area-edge{opacity:1}.cell.area-hover .area-edge{opacity:1;border-color:#8b5cf6;border-style:dashed}
.depth{display:block;font-size:9px;font-weight:700;line-height:1.05;background:rgba(255,255,255,.35)}
.cell:focus-visible{outline:3px solid #2563eb;z-index:1}.cell.candidate:hover{filter:brightness(.92)}
.district{display:block;font-size:clamp(8px,1.7vw,11px);font-weight:700;background:rgba(255,255,255,.35);line-height:1.2;border-radius:2px}
.legend{display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:11px;margin-top:7px}.item{display:inline-flex;align-items:center;gap:5px}
.box{width:13px;height:13px;border:1px solid #111;display:inline-block}.chosen{border:2px solid #be123c}.reported{border:3px solid #888}
#scale{margin-top:7px;font-size:11px;display:flex;align-items:center;gap:7px}.ramp{width:125px;height:10px}
#hint{min-height:36px;font-size:11px;line-height:1.5;margin-top:5px;color:#64748b}
</style></head><body><div class="wrap"><div id="grid"></div>
<div class="legend"><span class="item"><i class="box"></i>センサ</span><span class="item"><i class="box" style="border:2px dashed #2563eb"></i>追加センサ</span><span class="item"><i class="box reported"></i>通報・要請</span><span class="item"><i class="box chosen"></i>派遣先・排水区域</span></div>
<div class="legend" id="observation-legend"><span class="item"><i class="box" style="border:none;background:#e5e7eb"></i>水深の情報なし</span><span class="item"><i class="box" style="border:none;background:repeating-linear-gradient(135deg,#dbeafe 0px,#dbeafe 4px,#7da5c7 4px,#7da5c7 6px)"></i>冠水のみ判明</span><span>数値：m ／ 約：通報による目測</span></div>
<div id="scale"></div><div id="hint"></div></div><script>
let args,pending=null;
const send=(type,data={})=>parent.postMessage({isStreamlitMessage:true,type,...data},'*');
const resize=()=>send('streamlit:setFrameHeight',{height:document.querySelector('.wrap').getBoundingClientRect().height+3});
function render(){
 const grid=document.getElementById('grid');const focus=document.activeElement?.dataset.cell;grid.replaceChildren();
 const axis=t=>{let e=document.createElement('div');e.className='axis';e.textContent=t;grid.append(e)};
 axis('');for(let i=1;i<=10;i++)axis(i);
 args.cells.forEach((c,i)=>{
  if(i%10===0)axis(String.fromCharCode(65+Math.floor(i/10)));
  const sensorMode=args.mode==='追加センサを置く';const selectable=sensorMode||c.candidate;
  let e=document.createElement(selectable?'button':'div');e.className='cell'+(c.candidate?' candidate':'')+(c.selected?' selected':'')+(c.sensor?' has-sensor':'')+(c.report?' has-report':'')+(c.affected?' affected':'')+(sensorMode?' sensor-target':'')+(i===args.extra?' extra-sensor':'');e.style.background=c.color;e.dataset.cell=i;e.dataset.owner=c.owner;e.title=c.description;let edge=document.createElement('span');edge.className='area-edge';edge.style.borderWidth=c.edges.map(b=>b?'2px':'0px').join(' ');e.append(edge);
  if(selectable){e.type='button';e.disabled=pending!==null;e.setAttribute('aria-pressed',String(sensorMode?i===args.extra:c.selected));e.setAttribute('aria-label',c.name+' '+c.district+(sensorMode?' 追加センサ地点':''))}
  if(c.candidate){let t=document.createElement('span');t.className='district';t.textContent=c.district;e.append(t)}
  if(c.label){let t=document.createElement('span');t.className='depth';t.textContent=c.label;e.append(t)}
  e.onmouseenter=e.onfocus=()=>{document.getElementById('hint').textContent=c.description;grid.querySelectorAll('.cell').forEach(el=>el.classList.toggle('area-hover',Number(el.dataset.owner)===c.owner))};
  e.onmouseleave=e.onblur=()=>grid.querySelectorAll('.area-hover').forEach(el=>el.classList.remove('area-hover'));
  if(selectable)e.onclick=()=>{if(pending!==null)return;pending=crypto.randomUUID();send('streamlit:setComponentValue',{value:{cell:i,token:pending,scene:args.scene},dataType:'json'});grid.querySelectorAll('button').forEach(b=>b.disabled=true)};
  grid.append(e);
 });
 document.getElementById('observation-legend').style.display=args.observation?'flex':'none';
 const scale=document.getElementById('scale');scale.replaceChildren();
 let t=document.createElement('span');t.textContent=args.legend;let r=document.createElement('span');r.className='ramp';r.style.background=args.ramp;scale.append(t,r);
 document.getElementById('hint').textContent=args.mode==='追加センサを置く'?'任意のマスをクリックしてセンサを1地点追加。別のマスで移動、同じマスで解除。':'地区名のあるマスをクリックして選択・解除（最大３地区）。マウスを重ねると排水区域と詳細を表示。';
 if(focus!==undefined)grid.querySelector('button[data-cell="'+focus+'"]')?.focus({preventScroll:true});resize();
}
window.addEventListener('message',e=>{if(e.source!==parent||e.data.type!=='streamlit:render')return;args=e.data.args;if(args.ack===pending)pending=null;render()});
new ResizeObserver(resize).observe(document.querySelector('.wrap'));send('streamlit:componentReady',{apiVersion:1});resize();
</script></body></html>"""


def clickable_grid(**kwargs):
    # Embed the component in app.py so Cloud deployment needs only this file.
    import hashlib,tempfile
    from pathlib import Path
    import streamlit.components.v1 as components
    folder=Path(tempfile.gettempdir())/("urban-flood-grid-"+hashlib.sha256(GRID_HTML.encode()).hexdigest()[:12])
    folder.mkdir(exist_ok=True)
    target=folder/"index.html"
    if not target.exists(): target.write_text(GRID_HTML,encoding="utf-8")
    return components.declare_component("flood_allocation_map",path=str(folder))(**kwargs)


def main():
    import streamlit as st
    # Keep the evaluation selection when a map click reruns before the KPI widgets.
    for key in ("ui_basis","ui_view"):
        if key in st.session_state: st.session_state[key]=st.session_state[key]
    st.set_page_config(page_title="Mini Urban Flood Digital Twin",layout="wide")
    st.markdown("""<style>.block-container{padding-top:3rem;padding-bottom:.4rem;padding-left:1rem;padding-right:1rem;max-width:1450px}
    h1{font-size:1.65rem!important;margin-bottom:.1rem!important}h3{font-size:1.1rem!important}
    [data-testid=stVerticalBlock]{gap:.4rem}[data-testid=stMetricValue]{font-size:1.45rem}
    [data-testid=stMetricLabel]{font-size:.8rem}
    .reference{color:#7b8491;font-size:.8rem;line-height:1.5;padding-bottom:8px}
    .st-key-ui_view [role="radiogroup"] label:last-child p,.st-key-ui_basis [role="radiogroup"] label:last-child p{color:#7b8491}
    </style>""",unsafe_allow_html=True)
    st.title("Mini Urban Flood Digital Twin")
    with st.sidebar:
        scenario=st.selectbox("豪雨ケース",[0,1,2],format_func=lambda i:["基本：局地豪雨","病院周辺の雨が弱い","病院周辺の雨が強い"][i],key="ui_weather")
        source=st.radio("利用する情報",SOURCES,index=0,key="ui_source")
        count=st.slider("既設センサ数",3,30,3,key="ui_count",disabled=source==SOURCES[0])
        strategy=st.selectbox("配置戦略",STRATEGIES,key="ui_strategy",disabled=source==SOURCES[0])
        sigma=st.selectbox("センサ誤差 σ（m）",[0.,.03,.10,.20],index=1,key="ui_sigma",disabled=source==SOURCES[0])
    if st.session_state.get("ui_case")!=scenario:
        st.session_state.update(ui_case=scenario,picked=[],selection_note="",extra_sensor=None,sensor_note="")
    if source==SOURCES[0]: st.sidebar.caption("センサ・通報は未使用。重要度から配分を検討。")
    city=make_city(scenario);reports=make_reports(city)
    base_sensors=sensor_order(city,strategy)[:count]
    st.session_state["base_sensors"]=base_sensors.tolist()
    extra=st.session_state.get("extra_sensor")
    if extra in base_sensors:
        extra=None;st.session_state["extra_sensor"]=None
    with st.sidebar:
        mode=st.radio("地図のクリック操作",["派遣先を選ぶ","追加センサを置く"],key="map_mode",disabled=source==SOURCES[0])
        if extra is not None:
            st.caption(f"追加地点：{cell_name(extra)} ｜ センサ合計：{count+1}地点" if source!=SOURCES[0] else f"追加地点：{cell_name(extra)}（未使用）")
            if st.button("追加センサを解除"):
                st.session_state["extra_sensor"]=None;st.session_state["sensor_note"]="";st.rerun()
        elif source!=SOURCES[0] and mode=="追加センサを置く":
            st.caption("任意のマスをクリック。既設センサに加えて1地点を指定。")
        if st.session_state.get("sensor_note"): st.caption(st.session_state["sensor_note"])
    effective_mode=mode if source!=SOURCES[0] else "派遣先を選ぶ"
    scene=f"{scenario}:{strategy}:{count}:{source}:{effective_mode}"
    st.session_state["ui_scene"]=scene
    sensors=np.append(base_sensors,extra).astype(int) if extra is not None else base_sensors
    values=observe(city,sensors,sigma)
    prediction,coverage=estimate_sources(city.xy,city.prior,reports,sensors,values,sigma,source)
    rr=reports if source==SOURCES[2] else []
    ss=sensors if source!=SOURCES[0] else np.array([],dtype=int)
    vv=values if source!=SOURCES[0] else np.array([])
    left,right=st.columns([1.25,1],gap="large")
    with left:
        view=st.radio("地図",VIEWS,horizontal=True,key="ui_view",label_visibility="collapsed")
        if view==VIEWS[3]: st.markdown('<div class="reference">参照用の真値：実運用では都市全体を直接把握できない情報</div>',unsafe_allow_html=True)
        if view==VIEWS[2] and source==SOURCES[0]: st.caption("地形・排水条件に基づく事前推定（観測・通報による補正なし）")
        event=clickable_grid(**grid_data(city,prediction,coverage,rr,ss,vv,st.session_state["picked"],view),
                             ack=st.session_state.get("click_ack"),scene=scene,mode=effective_mode,extra=extra if source!=SOURCES[0] else None,key="allocation_grid",default=None)
        if allocation_event(st.session_state,event): st.rerun()
        st.caption("1台で1地区の排水区域（6〜16区画）に作用。最大3地区。")
    with right:
        pumps=st.session_state["picked"]
        st.subheader(f"配分先 {len(pumps)} / 3地区")
        st.write(" ／ ".join(CANDIDATES[i] for i in pumps) or "地図の候補地区をクリックして選択")
        if st.session_state.get("selection_note"): st.caption(st.session_state["selection_note"])
        b1,b2=st.columns(2)
        if b1.button("推定案を選択",width="stretch",disabled=view!=VIEWS[2],help="推定された浸水深を表示しているときに使用"):
            st.session_state["picked"]=dispatch(prediction,city.weight).tolist();st.session_state["selection_note"]="";st.rerun()
        if b2.button("選択を解除",width="stretch"):
            st.session_state["picked"]=[];st.session_state["selection_note"]="";st.rerun()
        basis=st.radio("KPIの評価条件",["未評価","推定に基づく評価","実際の条件（参照）"],horizontal=True,key="ui_basis")
        actual=basis=="実際の条件（参照）"
        if actual: st.markdown('<div class="reference">参照用の真値で、現在の選択を評価</div>',unsafe_allow_html=True)
        evaluated=basis!="未評価"
        h=city.truth if actual else prediction
        k=outcome(h,city.weight,pumps)
        c1,c2=st.columns(2)
        c1.metric("被害軽減量",f"{k['avoided']:.2f}" if evaluated else "—",help="対策前と対策後の被害指標の差。金額・人数ではない相対指標。")
        c2.metric("被害軽減率",f"{k['rate']:.1f}%" if evaluated else "—",help="全100区画の対策前被害に対する軽減量の割合。")
        c1.metric("対策後に残る被害",f"{k['remaining']:.2f}" if evaluated else "—")
        c2.metric("重要区画の軽減量",f"{k['important']:.2f}" if evaluated else "—",help="重要度６以上の区画の被害軽減量。全体の軽減量の内数。")
        c1.metric("改善した区画数",f"{k['dry']} / 100" if evaluated else "—",help="排水区域への作用により、水深が0.10 m超から0.10 m以下になった区画数。安全判定ではない。")
        c2.metric("被害水準超の区画数",f"{k['residual']} / 100" if evaluated else "—",help="対策後も水深が0.10 mを超える区画数。全100区画を集計。")
        if evaluated: st.caption(f"対策前の被害：{k['total']:.2f} ｜ 重要区画＝重要度６以上")
        if actual:
            st.caption(f"全域の推定RMSE：{np.sqrt(np.mean((prediction-city.truth)**2)):.3f} m（配分ではなく観測・推定の評価）")
        elif evaluated: st.caption("KPIは現在の推定に基づく値。地図の表示切替では評価条件は変化しない。")
        else: st.caption("配分先を選び、評価条件を切り替えて結果を確認。")
    with st.expander("通報の内容",expanded=False):
        st.dataframe([{"地区":CANDIDATES[r.cell],"時刻":r.time,"種類":r.kind,"内容":r.text,"件数":r.count} for r in rr],hide_index=True,width="stretch")
    with st.expander("候補地区の数値",expanded=False):
        st.dataframe([{"地区":CANDIDATES[i],"区域の区画数":len(AREAS[i]),"重要度の合計":round(float(city.weight[AREAS[i]].sum()),1),"区域の平均推定水深 m":round(float(prediction[AREAS[i]].mean()),2),"推定被害軽減":round(float(gains(prediction,city.weight)[i]),2)} for i in CANDIDATES],hide_index=True,width="stretch")
    with st.expander("モデル・KPI・凡例の補足",expanded=False):
        st.markdown("""- 候補８地区から最大３地区を選択。１台が担当する固定の排水区域内の全区画で、水深を最大0.25 m低減。区域は重複せず、各地区に最大１台。地区名のマスは配置地点であり、効果は区域全体に及ぶ。
- 区画面積は同一と仮定。区域によって広さと排水条件が異なり、１台当たりの排水体積を同一にはしていない。流動・ポンプ能力・稼働時間の詳細は省略した教育用の効果モデル。
- 基本ケースは東部の広い浸水と病院周辺の局地雨、西部の想定以上の排水を組み合わせた仮想事例。センサ配置は重要度・地形・位置だけで決定し、実際の浸水深を参照しない。
- 被害＝重要度 × max(水深−0.10 m, 0) の全区画合計。重要度は施設・生活への影響を簡略化した設定。被害軽減量・残る被害は相対指標。
- 「被害水準」はこのモデルの0.10 mという閾値。安全や通行可能性を示す基準ではない。
- 定性的な冠水通報は、水深が正である片側の条件として利用（検出目安0.02 m）。水深を特定値に置換せず、同一事象の重複通報を重ねて加点しない。
- センサ値と目測水深の通報を誤差付きで利用（目測の標準偏差0.08 m）。通報件数を独立した観測数として加算しない。通報のない場所も事前情報と空間的な関係から推定。
- 観測・通報の地図では、水深の情報がない場所は灰色、冠水のみ判明した場所は縞模様。支援要請だけでは水深を補正しない。センサと数値通報が重なる場合はセンサ値を色と数値で表示し、通報はマウスオーバーで確認。
- 事前情報のみの推定図は地形・排水条件に基づく事前推定。観測・通報を使用した推定と区別。
- 地図の重要度・推定・真値は表示切替。右側の評価条件は独立。参照用の真値を見ても自動配分は推定だけを使用。
- 追加センサは既設配置に1地点だけ追加。同じ地点の観測値と誤差は固定し、移動しても乱数を引き直さない。情報を追加しても派遣先は自動変更しない。実際の条件でのKPIは配分を変えたときに変化。
- 全情報は同じ10:00時点。道路移動・水の流動・時間変化は省略。教育用の仮想モデル。""")


if __name__ == "__main__":
    main()
