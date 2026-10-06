"""Small, self-contained PV orientation teaching demo; no downloaded data."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

SEASONS = {"春分頃（3月）": 80, "夏至頃（6月）": 172,
           "秋分頃（9月）": 266, "冬至頃（12月）": 355}
AZIMUTHS = {"南": 180, "南東": 135, "東": 90, "南西": 225, "西": 270, "北": 0}
LATITUDE = 35.7
AREA = 25.0
EFFICIENCY = 0.20


def calculate(day, tilt, azimuth):
    """ENU solar vectors; clock is local solar time, not civil/JST time.

    Declination approximation, synthetic clear-day DNI/DHI, isotropic sky
    and albedo 0.2. Fixed efficiency: output is DC, without inverter clipping.
    """
    hours = np.linspace(0, 24, 289)
    dec = np.deg2rad(23.45 * np.sin(2 * np.pi * (284 + day) / 365))
    lat = np.deg2rad(LATITUDE)
    ha = np.deg2rad(15 * (hours - 12))
    east = -np.cos(dec) * np.sin(ha)
    north = np.cos(lat) * np.sin(dec) - np.sin(lat) * np.cos(dec) * np.cos(ha)
    up = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    sun = np.column_stack((east, north, up))
    beta, gamma = np.deg2rad([tilt, azimuth])
    normal = np.array([np.sin(beta) * np.sin(gamma),
                       np.sin(beta) * np.cos(gamma), np.cos(beta)])
    cos_incidence = sun @ normal
    daylight = up > 0
    elevation = np.maximum(up, 0)
    # Smooth synthetic atmosphere, declining to zero near the horizon.
    dni = np.where(daylight, 900 * np.exp(-0.16 / np.maximum(elevation, 1e-6)), 0)
    dhi = 100 * np.sqrt(elevation)
    ghi = dni * elevation + dhi
    direct = dni * np.maximum(cos_incidence, 0)
    diffuse = dhi * (1 + np.cos(beta)) / 2
    reflected = 0.2 * ghi * (1 - np.cos(beta)) / 2
    poa = direct + diffuse + reflected
    power = EFFICIENCY * AREA * poa / 1000
    horizontal = EFFICIENCY * AREA * ghi / 1000
    return dict(hours=hours, sun=sun, normal=normal, power=power,
                horizontal=horizontal, poa=poa, direct=direct,
                diffuse=diffuse, reflected=reflected,
                incidence=np.rad2deg(np.arccos(np.clip(cos_incidence, -1, 1))),
                altitude=np.rad2deg(np.arcsin(np.clip(up, -1, 1))))


def geometry_chart(data, hour, tilt, azimuth):
    i = int(round(hour * 12))
    sun = data["sun"][i]
    normal = data["normal"]
    beta, gamma = np.deg2rad([tilt, azimuth])
    across = np.array([np.cos(gamma), -np.sin(gamma), 0])
    along = np.array([np.cos(beta) * np.sin(gamma),
                      np.cos(beta) * np.cos(gamma), -np.sin(beta)])
    center = np.array([0., 0., .5])
    corners = np.array([center + a * across + b * along
                        for a, b in [(-.65, -.4), (.65, -.4), (.65, .4), (-.65, .4)]])
    fig = go.Figure(go.Mesh3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
                              i=[0, 0], j=[1, 2], k=[2, 3], color="#286aa0",
                              opacity=.95, hoverinfo="skip", name="パネル"))
    def segment(start, end, color, name, dash="solid"):
        fig.add_trace(go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]],
                                  z=[start[2], end[2]], mode="lines", name=name,
                                  line=dict(color=color, width=6, dash=dash), hoverinfo="skip"))
    segment(center, center + normal * .85, "#536579", "パネル面の法線", "dash")
    if sun[2] > 0:
        endpoint = center + sun * 1.8
        segment(center, endpoint, "#e4a126", "太陽の方向")
        fig.add_trace(go.Scatter3d(x=[endpoint[0]], y=[endpoint[1]], z=[endpoint[2]],
                                  mode="markers", marker=dict(size=12, color="#efad32"),
                                  showlegend=False, hoverinfo="skip"))
        # Cone points along incoming ray, from the sun toward the panel.
        pos = center + sun * .5
        fig.add_trace(go.Cone(x=[pos[0]], y=[pos[1]], z=[pos[2]],
                             u=[-sun[0]], v=[-sun[1]], w=[-sun[2]],
                             sizemode="absolute", sizeref=.22, showscale=False,
                             colorscale=[[0, "#e4a126"], [1, "#e4a126"]], hoverinfo="skip"))
    for position, text in [([0, -1.3, 0], "南"), ([1.3, 0, 0], "東"),
                           ([0, 1.3, 0], "北"), ([-1.3, 0, 0], "西")]:
        fig.add_trace(go.Scatter3d(x=[position[0]], y=[position[1]], z=[position[2]],
                                  text=[text], mode="text", textfont=dict(size=16),
                                  showlegend=False, hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=30),
                      legend=dict(orientation="h", x=0, y=-.08, font=dict(size=12)),
                      scene=dict(xaxis=dict(visible=False, range=[-2, 2]),
                                 yaxis=dict(visible=False, range=[-2, 2]),
                                 zaxis=dict(visible=False, range=[0, 2.5]),
                                 aspectmode="manual", aspectratio=dict(x=1, y=1, z=.7),
                                 camera=dict(eye=dict(x=1.6, y=-2, z=1.4))),
                      paper_bgcolor="white", font=dict(color="#213449"))
    return fig


def output_chart(data, hour, compare):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["hours"], y=data["power"], name="設定したパネル",
                             line=dict(color="#2679a9", width=3),
                             hovertemplate="%{x:.2f}時：%{y:.2f} kW<extra></extra>"))
    if compare:
        fig.add_trace(go.Scatter(x=data["hours"], y=data["horizontal"],
                                 name="水平面の日射をそのまま使う計算",
                                 line=dict(color="#9a94ab", width=2, dash="dot")))
    fig.add_vline(x=hour, line_color="#c08a2a", line_dash="dash")
    fig.update_layout(height=320, margin=dict(l=50, r=15, t=50, b=55),
                      legend=dict(orientation="h", x=0, y=1.20, font=dict(size=11)),
                      hovermode="x unified", paper_bgcolor="white", plot_bgcolor="#fafbfd",
                      font=dict(color="#213449", size=13))
    fig.update_xaxes(title="時刻（太陽時：南中が12時）", range=[4, 20], dtick=2, gridcolor="#e8edf2")
    fig.update_yaxes(title="発電出力（kW）", range=[0, 5.5], gridcolor="#e8edf2")
    return fig


def integrate(values, hours):
    # Compatible with both NumPy 1.x and 2.x.
    return float(np.sum((values[:-1] + values[1:]) * np.diff(hours) / 2))


def main():
    st.set_page_config(page_title="Mini PV Panel Digital Twin", layout="wide")
    st.markdown("""<style>
    .block-container {padding:3rem 1.8rem .5rem;max-width:1450px;}
    .demo-title {font-size:1.6rem;font-weight:700;line-height:1.45;padding:.2rem 0 .6rem;}
    .section-title {font-size:1.05rem;font-weight:600;line-height:1.5;padding:.3rem 0 .4rem;}
    [data-testid="stVerticalBlock"] {gap:.5rem;}
    [data-testid="stCaptionContainer"] p {line-height:1.55;margin:0;}
    [data-testid="stMetricValue"] {font-size:1.7rem;}
    [data-testid="stSidebar"] {min-width:260px;max-width:280px;}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="demo-title">Mini PV Panel Digital Twin</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("パネルの設定")
        season = st.selectbox("季節", list(SEASONS))
        tilt = st.slider("傾斜角（水平面から）", 0, 90, 30, 5, format="%d°")
        direction = st.selectbox("パネルが向く方位", list(AZIMUTHS))
        hour = st.slider("図で確認する時刻（太陽時）", 5.0, 19.0, 12.0, .5, format="%.1f時")
        compare = st.checkbox("角度を考慮しない計算を重ねる", value=False)
        st.caption("東京付近・晴天を仮定\n\nパネル面積25 m²／変換効率20%（固定）")
    data = calculate(SEASONS[season], tilt, AZIMUTHS[direction])
    st.caption(f"{season} ｜ {direction}向き・傾斜{tilt}° ｜ 同じ日の条件で設置角度を変更")
    left, right = st.columns([1, 1.45])
    i = int(round(hour * 12))
    with left:
        st.markdown('<div class="section-title">太陽の方向とパネル面</div>', unsafe_allow_html=True)
        st.plotly_chart(geometry_chart(data, hour, tilt, AZIMUTHS[direction]), use_container_width=True)
        if data["altitude"][i] > 0:
            st.caption(f"{hour:g}時：太陽高度 {data['altitude'][i]:.0f}° ／ 入射角 {data['incidence'][i]:.0f}°（法線との角度）")
        else:
            st.caption(f"{hour:g}時：太陽は地平線の下")
    with right:
        st.markdown('<div class="section-title">１日の発電出力</div>', unsafe_allow_html=True)
        st.plotly_chart(output_chart(data, hour, compare), use_container_width=True)
        st.caption("縦の点線：左で指定した時刻。時刻の変更は１日の計算条件を変更しない。")
    metrics = st.columns(3)
    energy = integrate(data["power"], data["hours"])
    metrics[0].metric("１日の発電量", f"{energy:.1f} kWh")
    metrics[1].metric(f"{hour:g}時の発電出力", f"{data['power'][i]:.2f} kW")
    metrics[2].metric("発電出力が最大の時刻", f"{data['hours'][np.argmax(data['power'])]:.1f}時")
    if compare:
        base = integrate(data["horizontal"], data["hours"])
        st.caption(f"角度を考慮しない計算の１日発電量：{base:.1f} kWh ／ 設定したパネルとの差：{energy - base:+.1f} kWh")
    with st.expander("計算の関係と仮定", expanded=False):
        st.latex(r"\hat P(t)=\eta A\,G_{\mathrm{面}}(t)/1000")
        st.latex(r"G_{\mathrm{面}}=DNI\max(0,\cos\theta)+DHI\frac{1+\cos\beta}{2}+\rho GHI\frac{1-\cos\beta}{2}")
        st.markdown("入射角 θ：太陽の方向とパネル面の法線との角度。傾斜角 β：水平面からの角度。"
                    "DNI：直達日射、DHI：水平面の散乱日射、GHI：水平面の全天日射。地表の反射率 ρ＝0.2。")
        st.caption("晴天条件。太陽位置は緯度35.7°と代表日から計算。"
                   "時刻は太陽時で、時計の日本標準時とは異なる。影・温度による効率変化・機器損失は含まない。"
                   "発電出力は直流出力の推定。比較線は同じ効率・面積で水平面の日射を使う計算（傾斜0°と一致）。")


if __name__ == "__main__":
    main()
