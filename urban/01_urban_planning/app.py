import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from math import sqrt

st.set_page_config(
    page_title="Mini Urban & Resilience Digital Twin",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.7rem;
        padding-bottom: 0.7rem;
        max-width: 1500px;
    }
    h1, h2, h3 {
        margin-top: 0.15rem;
        margin-bottom: 0.25rem;
    }
    div[data-testid="stMetric"] {
        padding-top: 0.05rem;
        padding-bottom: 0.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Mini Urban & Resilience Digital Twin")

# ============================================================
# 1. Virtual city
# ============================================================

hours = np.arange(24)

districts = {
    "A": {
        "name": "住宅地区A",
        "x": 1.0, "y": 3.0,
        "population": 1800,
        "daily_energy": 420,
        "goods": 90,
        "critical": None,
        "profile": "residential",
    },
    "B": {
        "name": "商業地区B",
        "x": 3.0, "y": 3.3,
        "population": 900,
        "daily_energy": 620,
        "goods": 120,
        "critical": None,
        "profile": "commercial",
    },
    "C": {
        "name": "住宅地区C",
        "x": 5.2, "y": 3.1,
        "population": 1500,
        "daily_energy": 390,
        "goods": 80,
        "critical": None,
        "profile": "residential",
    },
    "D": {
        "name": "中心地区D",
        "x": 2.0, "y": 1.7,
        "population": 1200,
        "daily_energy": 700,
        "goods": 140,
        "critical": "市役所",
        "profile": "commercial",
    },
    "E": {
        "name": "病院地区E",
        "x": 4.2, "y": 1.6,
        "population": 700,
        "daily_energy": 560,
        "goods": 110,
        "critical": "病院",
        "profile": "flat",
    },
    "F": {
        "name": "郊外地区F",
        "x": 0.8, "y": 0.3,
        "population": 1000,
        "daily_energy": 260,
        "goods": 70,
        "critical": None,
        "profile": "residential",
    },
    "G": {
        "name": "避難所地区G",
        "x": 3.0, "y": 0.2,
        "population": 800,
        "daily_energy": 300,
        "goods": 100,
        "critical": "避難所",
        "profile": "public",
    },
    "H": {
        "name": "学校地区H",
        "x": 5.3, "y": 0.4,
        "population": 1100,
        "daily_energy": 340,
        "goods": 85,
        "critical": "学校",
        "profile": "school",
    },
}

roads = [
    ("A", "B"),
    ("B", "C"),
    ("A", "D"),
    ("B", "D"),
    ("B", "E"),
    ("C", "E"),
    ("D", "E"),
    ("D", "F"),
    ("D", "G"),
    ("E", "G"),
    ("E", "H"),
    ("F", "G"),
    ("G", "H"),
]

# An external logistics gateway. Goods can reach the city even with no local hub.
EXTERNAL_NODE = "EXT"
EXTERNAL_LOGISTICS = {
    "x": -0.8,
    "y": 1.8,
    "name": "都市外物流センター",
}

# The external logistics center enters the city through A and F.
EXTERNAL_ROADS = [
    (EXTERNAL_NODE, "A"),
    (EXTERNAL_NODE, "F"),
]

# External grid node used only as an educational representation of
# the electricity-supply layer. In normal operation every district can
# purchase any residual demand from this grid.
GRID_NODE = "GRID"
GRID_SUPPLY = {
    "x": 6.15,
    "y": 2.2,
    "name": "系統電力",
}

# Radial distribution-grid topology for the electricity layer.
# The external grid is connected to district E.
# E is the upstream entry point of a radial urban distribution feeder.
# The A-D link is intentionally omitted so the teaching example has no loop.
POWER_LINES = [
    (GRID_NODE, "E"),
    ("E", "B"),
    ("B", "A"),
    ("B", "C"),
    ("E", "G"),
    ("G", "D"),
    ("D", "F"),
    ("G", "H"),
]

def power_node_xy(node):
    if node == GRID_NODE:
        return GRID_SUPPLY["x"], GRID_SUPPLY["y"]
    return districts[node]["x"], districts[node]["y"]

def build_power_adjacency(blocked_lines=None):
    blocked_lines = blocked_lines or set()
    blocked_norm = {tuple(sorted(e)) for e in blocked_lines}
    nodes = [GRID_NODE] + list(districts.keys())
    adj = {n: [] for n in nodes}

    for a, b in POWER_LINES:
        if tuple(sorted((a, b))) in blocked_norm:
            continue
        adj[a].append(b)
        adj[b].append(a)

    return adj

def power_reachable_from_grid(blocked_lines=None):
    adj = build_power_adjacency(blocked_lines)
    reached = set([GRID_NODE])
    stack = [GRID_NODE]

    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in reached:
                reached.add(nxt)
                stack.append(nxt)

    return reached

def power_children_and_parent():
    # Root the radial network at GRID_NODE.
    adj = build_power_adjacency()
    parent = {GRID_NODE: None}
    order = [GRID_NODE]
    for cur in order:
        for nxt in adj[cur]:
            if nxt == parent.get(cur):
                continue
            if nxt in parent:
                continue
            parent[nxt] = cur
            order.append(nxt)

    children = {n: [] for n in parent}
    for node, par in parent.items():
        if par is not None:
            children[par].append(node)

    return parent, children, order

POWER_PARENT, POWER_CHILDREN, POWER_ORDER = power_children_and_parent()

def calculate_power_flows_for_hour(hour, energy_choice, normal_energy, blocked_lines=None):
    """
    Calculate illustrative upstream power flow for the chosen hour.
    Local supply is subtracted at each district; remaining net demand
    is aggregated upstream through the radial distribution topology.
    If a line is blocked, downstream nodes disconnected from GRID have no grid supply.
    """
    blocked_lines = blocked_lines or set()
    reachable = power_reachable_from_grid(blocked_lines)

    local_net = {}
    for d in districts:
        demand_h = float(districts[d]["demand_profile"][hour])
        local_h = float(normal_energy["district_ops"][d]["local_supply"][hour])
        local_net[d] = max(demand_h - local_h, 0.0)

    subtree_need = {n: 0.0 for n in POWER_PARENT}
    for d in districts:
        subtree_need[d] = local_net[d]

    # Accumulate from leaves toward root, but only through intact links.
    blocked_norm = {tuple(sorted(e)) for e in blocked_lines}
    for node in reversed(POWER_ORDER):
        par = POWER_PARENT[node]
        if par is None:
            continue
        if tuple(sorted((node, par))) in blocked_norm:
            continue
        subtree_need[par] += subtree_need[node]

    line_flows = {}
    for a, b in POWER_LINES:
        edge = tuple(sorted((a, b)))
        if edge in blocked_norm:
            line_flows[(a, b)] = None
            continue

        # Determine downstream node using parent relation.
        if POWER_PARENT.get(b) == a:
            downstream = b
        elif POWER_PARENT.get(a) == b:
            downstream = a
        else:
            downstream = b

        if downstream not in reachable:
            line_flows[(a, b)] = 0.0
        else:
            line_flows[(a, b)] = subtree_need[downstream]

    return {
        "local_net": local_net,
        "line_flows": line_flows,
        "reachable": reachable,
    }

# ============================================================
# 2. Demand and generation profiles
# ============================================================

def normalize_profile(values):
    v = np.asarray(values, dtype=float)
    return v / v.sum()

DEMAND_SHAPES = {
    "residential": normalize_profile(
        0.45
        + 0.75 * np.exp(-0.5 * ((hours - 7.0) / 1.8) ** 2)
        + 1.10 * np.exp(-0.5 * ((hours - 19.5) / 2.2) ** 2)
    ),
    "commercial": normalize_profile(
        0.20
        + 1.25 * np.exp(-0.5 * ((hours - 13.0) / 4.0) ** 2)
    ),
    "flat": normalize_profile(
        0.85
        + 0.10 * np.sin((hours - 6) / 24 * 2 * np.pi)
    ),
    "public": normalize_profile(
        0.35
        + 0.70 * np.exp(-0.5 * ((hours - 12.0) / 4.5) ** 2)
        + 0.45 * np.exp(-0.5 * ((hours - 19.0) / 2.8) ** 2)
    ),
    "school": normalize_profile(
        0.12
        + 1.45 * np.exp(-0.5 * ((hours - 12.5) / 3.2) ** 2)
    ),
}

for d, p in districts.items():
    p["demand_profile"] = (
        DEMAND_SHAPES[p["profile"]] * p["daily_energy"]
    )

# Grid CI [kg-CO2/kWh] — known teaching profile.
GRID_CI = (
    0.43
    + 0.06 * np.exp(-0.5 * ((hours - 19.0) / 2.5) ** 2)
    - 0.08 * np.exp(-0.5 * ((hours - 13.0) / 3.5) ** 2)
)

# Each energy-hub type has a fixed teaching-scale output profile.
PV_PROFILE = np.maximum(
    0.0,
    np.sin(np.pi * (hours - 6) / 12)
)
PV_PROFILE = PV_PROFILE / max(PV_PROFILE.sum(), 1e-9) * 250

WIND_RAW = (
    0.60
    + 0.20 * np.cos((hours - 2) / 24 * 2 * np.pi)
    + 0.08 * np.cos((hours - 15) / 12 * 2 * np.pi)
)
WIND_PROFILE = WIND_RAW / WIND_RAW.sum() * 230

STABLE_PROFILE = np.ones(24) * (205 / 24)

TECHNOLOGIES = {
    "なし": {
        "cost": 0,
        "generation": np.zeros(24),
        "ci": 0.0,
        "label": "なし",
    },
    "PV型": {
        "cost": 20,
        "generation": PV_PROFILE,
        "ci": 0.04,
        "label": "PV型",
    },
    "風力型": {
        "cost": 25,
        "generation": WIND_PROFILE,
        "ci": 0.02,
        "label": "風力型",
    },
    "安定電源型": {
        "cost": 35,
        "generation": STABLE_PROFILE,
        "ci": 0.20,
        "label": "安定電源型（コージェネ相当）",
    },
}

# Behind-the-scenes battery attached to every energy hub.
BATTERY_CAPACITY = 90.0
BATTERY_POWER = 45.0
BATTERY_EFF = 0.95

# Logistics hubs
LOGISTICS_HUB_COST = 15

# Teaching-scale logistics constraints.
# A regional logistics hub handles both:
#   1) transfer from the external / another regional hub,
#   2) last-mile pickup & delivery to surrounding districts.
#
# For last-mile service, each hub has a one-day driving-distance budget.
LOGISTICS_DAILY_DISTANCE_CAPACITY = 18.0
LOGISTICS_VEHICLE_CAPACITY_UNITS = 100.0
LOGISTICS_HUB_TRANSFER_MAX_DISTANCE = 8.0

# In disaster mode, an electrically isolated logistics hub remains operational
# only if local generation can support this fraction of its logistics load.
LOGISTICS_MIN_POWER_COVERAGE = 0.70

# Electricity consumed by one regional logistics hub [kWh/h].
# Refrigeration/IT create a base load; loading activity increases daytime demand.
LOGISTICS_POWER_PROFILE = (
    2.0
    + 3.8 * np.exp(-0.5 * ((hours - 11.0) / 3.0) ** 2)
    + 2.8 * np.exp(-0.5 * ((hours - 17.0) / 2.6) ** 2)
)

# 災害時の地区基礎需要倍率（講義用仮想値）
DISASTER_BASE_DEMAND_MULTIPLIER = {
    "A": 0.80,  # 住宅
    "B": 0.45,  # 商業
    "C": 0.80,  # 住宅
    "D": 0.90,  # 中心・行政
    "E": 1.00,  # 病院：低下しない
    "F": 0.75,  # 郊外住宅
    "G": 1.20,  # 避難所：増加
    "H": 0.50,  # 学校
}

# 災害時に最低限維持したいサービス率
DISASTER_MIN_SERVICE_RATIO = {
    "A": 0.35,
    "B": 0.30,
    "C": 0.35,
    "D": 0.70,
    "E": 0.85,
    "F": 0.30,
    "G": 0.75,
    "H": 0.45,
}

# Budget
TOTAL_BUDGET = 100

# Simple siting constraints for teaching.
# Dense commercial/central areas cannot host wind; hospital cannot be logistics hub.
ALLOWED_TECH = {
    "A": ["なし", "PV型", "風力型", "安定電源型"],
    "B": ["なし", "PV型", "安定電源型"],
    "C": ["なし", "PV型", "風力型", "安定電源型"],
    "D": ["なし", "PV型", "安定電源型"],
    "E": ["なし", "PV型", "安定電源型"],
    "F": ["なし", "PV型", "風力型", "安定電源型"],
    "G": ["なし", "PV型", "風力型", "安定電源型"],
    "H": ["なし", "PV型", "風力型", "安定電源型"],
}
LOGISTICS_ALLOWED = [d for d in districts if d != "E"]

# ============================================================
# 3. Graph utilities
# ============================================================

def node_xy(node):
    if node == EXTERNAL_NODE:
        return EXTERNAL_LOGISTICS["x"], EXTERNAL_LOGISTICS["y"]
    return districts[node]["x"], districts[node]["y"]


def euclidean_xy(x1, y1, x2, y2):
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def euclidean(a, b):
    xa, ya = node_xy(a)
    xb, yb = node_xy(b)
    return euclidean_xy(xa, ya, xb, yb)


def build_adjacency(blocked_roads=None, include_external=True):
    blocked_roads = blocked_roads or set()
    blocked_norm = {tuple(sorted(e)) for e in blocked_roads}

    nodes = list(districts.keys())
    if include_external:
        nodes.append(EXTERNAL_NODE)

    adj = {d: [] for d in nodes}

    all_links = list(roads)
    if include_external:
        all_links += EXTERNAL_ROADS

    for a, b in all_links:
        if tuple(sorted((a, b))) in blocked_norm:
            continue

        w = euclidean(a, b)
        adj[a].append((b, w))
        adj[b].append((a, w))

    return adj


def shortest_paths_from(source, blocked_roads=None, include_external=True):
    adj = build_adjacency(
        blocked_roads=blocked_roads,
        include_external=include_external,
    )

    dist = {d: float("inf") for d in adj}
    dist[source] = 0.0
    visited = set()

    while True:
        unvisited = [
            (dval, node)
            for node, dval in dist.items()
            if node not in visited
        ]

        if not unvisited:
            break

        current_dist, current = min(unvisited)

        if not np.isfinite(current_dist):
            break

        visited.add(current)

        for nxt, weight in adj[current]:
            candidate = current_dist + weight
            if candidate < dist[nxt]:
                dist[nxt] = candidate

    return dist


def nearest_local_logistics_distance(district, hubs, blocked_roads=None):
    if not hubs:
        return float("inf")

    return min(
        shortest_paths_from(
            hub,
            blocked_roads=blocked_roads,
            include_external=False,
        )[district]
        for hub in hubs
    )


def external_network_distance(district, blocked_roads=None):
    """
    Distance from the external logistics center to a district
    through the same road network shown on the map.
    """
    return shortest_paths_from(
        EXTERNAL_NODE,
        blocked_roads=blocked_roads,
        include_external=True,
    )[district]


def shortest_path_nodes(source, target, blocked_roads=None, include_external=True):
    """Return one shortest path as a list of network nodes."""
    adj = build_adjacency(
        blocked_roads=blocked_roads,
        include_external=include_external,
    )

    dist = {node: float("inf") for node in adj}
    prev = {node: None for node in adj}
    dist[source] = 0.0
    visited = set()

    while True:
        unvisited = [
            (dval, node)
            for node, dval in dist.items()
            if node not in visited
        ]
        if not unvisited:
            break

        current_dist, current = min(unvisited)
        if not np.isfinite(current_dist):
            break

        if current == target:
            break

        visited.add(current)

        for nxt, weight in adj[current]:
            candidate = current_dist + weight
            if candidate < dist[nxt]:
                dist[nxt] = candidate
                prev[nxt] = current

    if not np.isfinite(dist.get(target, float("inf"))):
        return []

    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        if cur == source:
            break
        cur = prev[cur]

    path.reverse()
    return path


def add_network_path(fig, path, *, name, legendgroup, showlegend, line):
    """Draw a path using the exact road-network geometry."""
    if len(path) < 2:
        return

    xs, ys = [], []
    for node in path:
        x, y = node_xy(node)
        xs.append(x)
        ys.append(y)

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            line=line,
            hoverinfo="skip",
        )
    )

# ============================================================
# 4. Energy operation
# ============================================================

def operate_local_energy(demand, generation):
    """
    Rule-based local energy operation.

    Generation first serves local demand.
    Surplus charges the attached battery.
    Stored energy is discharged later when demand exceeds generation.
    Any remaining deficit is supplied from the grid.

    This exact operation is used for all energy KPIs.
    """
    soc = 0.0
    grid = np.zeros(24)
    local_supply = np.zeros(24)
    battery_charge = np.zeros(24)
    battery_discharge = np.zeros(24)
    curtailed = np.zeros(24)

    for h in range(24):
        load = demand[h]
        gen = generation[h]

        direct = min(load, gen)
        local_supply[h] += direct

        surplus = max(gen - direct, 0.0)
        deficit = max(load - direct, 0.0)

        # Charge battery with surplus.
        room = BATTERY_CAPACITY - soc
        charge_input = min(
            surplus,
            BATTERY_POWER,
            room / BATTERY_EFF if BATTERY_EFF > 0 else 0.0,
        )
        stored = charge_input * BATTERY_EFF
        soc += stored
        battery_charge[h] = charge_input
        curtailed[h] = max(surplus - charge_input, 0.0)

        # Discharge battery into remaining deficit.
        available_output = soc * BATTERY_EFF
        discharge_to_load = min(
            deficit,
            BATTERY_POWER,
            available_output,
        )
        soc -= discharge_to_load / BATTERY_EFF if BATTERY_EFF > 0 else 0.0
        battery_discharge[h] = discharge_to_load
        local_supply[h] += discharge_to_load

        grid[h] = max(deficit - discharge_to_load, 0.0)

    return {
        "grid": grid,
        "local_supply": local_supply,
        "battery_charge": battery_charge,
        "battery_discharge": battery_discharge,
        "curtailed": curtailed,
    }

def normal_energy_metrics(energy_choice):
    total_demand = np.zeros(24)
    baseline_grid = np.zeros(24)
    actual_grid = np.zeros(24)
    total_local_supply = np.zeros(24)
    district_ops = {}

    for d, p in districts.items():
        demand = p["demand_profile"]
        total_demand += demand
        baseline_grid += demand

        tech = energy_choice[d]
        generation = TECHNOLOGIES[tech]["generation"]

        op = operate_local_energy(demand, generation)
        op["technology"] = tech
        op["local_ci"] = TECHNOLOGIES[tech]["ci"]
        district_ops[d] = op

        actual_grid += op["grid"]
        total_local_supply += op["local_supply"]

    baseline_co2 = float(np.sum(baseline_grid * GRID_CI))

    grid_co2 = float(np.sum(actual_grid * GRID_CI))
    local_co2 = 0.0
    for d, op in district_ops.items():
        local_co2 += float(
            np.sum(op["local_supply"]) * op["local_ci"]
        )

    actual_co2 = grid_co2 + local_co2
    co2_reduction = baseline_co2 - actual_co2

    local_supply_ratio = (
        100 * total_local_supply.sum() / total_demand.sum()
        if total_demand.sum() > 0 else 0.0
    )

    grid_reduction = baseline_grid.sum() - actual_grid.sum()

    return {
        "district_ops": district_ops,
        "total_demand": total_demand,
        "baseline_grid": baseline_grid,
        "actual_grid": actual_grid,
        "total_local_supply": total_local_supply,
        "baseline_co2": baseline_co2,
        "actual_co2": actual_co2,
        "grid_co2": grid_co2,
        "local_co2": local_co2,
        "co2_reduction": co2_reduction,
        "local_supply_ratio": local_supply_ratio,
        "grid_reduction": grid_reduction,
    }

# ============================================================
# 5. Logistics model
# ============================================================

def logistics_trip_count(goods_units):
    """Number of teaching-scale vehicle trips needed for a district."""
    return max(
        1,
        int(np.ceil(goods_units / LOGISTICS_VEHICLE_CAPACITY_UNITS)),
    )


def logistics_hub_power_status(
    hub,
    energy_choice,
    blocked_power_lines=None,
):
    """
    Is a regional logistics hub electrically operable?

    - If the district remains connected to the upper grid: operable.
    - If disconnected: the local energy hub must cover at least
      LOGISTICS_MIN_POWER_COVERAGE of the logistics equipment load.

    This intentionally isolates the logistics equipment requirement as a
    teaching abstraction so the cross-layer dependency is easy to follow.
    """
    reachable = power_reachable_from_grid(
        blocked_lines=blocked_power_lines,
    )

    if hub in reachable:
        return {
            "operational": True,
            "power_coverage": 1.0,
            "reason": "系統から電力供給可能",
        }

    tech = energy_choice[hub]
    generation = TECHNOLOGIES[tech]["generation"]

    op = operate_local_energy(
        LOGISTICS_POWER_PROFILE,
        generation,
    )

    coverage = (
        float(op["local_supply"].sum())
        / float(LOGISTICS_POWER_PROFILE.sum())
        if LOGISTICS_POWER_PROFILE.sum() > 0
        else 1.0
    )

    return {
        "operational": coverage >= LOGISTICS_MIN_POWER_COVERAGE,
        "power_coverage": coverage,
        "reason": (
            "地域電源で物流設備を維持"
            if coverage >= LOGISTICS_MIN_POWER_COVERAGE
            else "物流設備の電力不足"
        ),
    }


def build_logistics_supply_chain(
    logistics_hubs,
    blocked_roads=None,
    operational_hubs=None,
):
    """
    Build a simple hierarchical urban logistics network.

    1. Long-haul goods arrive at the external logistics center.
    2. Among usable regional hubs, the hub closest to EXT becomes the gateway hub.
    3. Remaining hubs are supplied from the nearest already supplied hub
       when the inter-hub transport distance is within the allowed limit.

    Returns a parent relation and the road-network paths used for trunk transport.
    """
    blocked_roads = blocked_roads or set()

    if operational_hubs is None:
        usable = list(logistics_hubs)
    else:
        usable = [
            h for h in logistics_hubs
            if operational_hubs.get(h, False)
        ]

    # Only hubs physically reachable from the external center can seed the chain.
    ext_dist = {
        h: external_network_distance(
            h,
            blocked_roads=blocked_roads,
        )
        for h in usable
    }

    reachable_from_ext = [
        h for h in usable
        if np.isfinite(ext_dist[h])
    ]

    if not reachable_from_ext:
        return {
            "supplied_hubs": [],
            "unsupplied_hubs": usable,
            "parent": {},
            "trunk_paths": {},
            "gateway": None,
            "trunk_distance": 0.0,
        }

    gateway = min(
        reachable_from_ext,
        key=lambda h: ext_dist[h],
    )

    supplied = [gateway]
    parent = {gateway: EXTERNAL_NODE}
    trunk_paths = {
        gateway: shortest_path_nodes(
            EXTERNAL_NODE,
            gateway,
            blocked_roads=blocked_roads,
            include_external=True,
        )
    }

    trunk_distance = ext_dist[gateway]
    remaining = [h for h in usable if h != gateway]

    progress = True
    while remaining and progress:
        progress = False

        for h in list(remaining):
            candidates = []

            for source in supplied:
                dmap = shortest_paths_from(
                    source,
                    blocked_roads=blocked_roads,
                    include_external=False,
                )
                dist = dmap[h]

                if (
                    np.isfinite(dist)
                    and dist <= LOGISTICS_HUB_TRANSFER_MAX_DISTANCE
                ):
                    candidates.append((dist, source))

            if not candidates:
                continue

            dist, source = min(candidates)
            supplied.append(h)
            parent[h] = source
            trunk_paths[h] = shortest_path_nodes(
                source,
                h,
                blocked_roads=blocked_roads,
                include_external=False,
            )
            trunk_distance += dist
            remaining.remove(h)
            progress = True

    return {
        "supplied_hubs": supplied,
        "unsupplied_hubs": remaining,
        "parent": parent,
        "trunk_paths": trunk_paths,
        "gateway": gateway,
        "trunk_distance": trunk_distance,
    }


def evaluate_logistics_network(
    logistics_hubs,
    blocked_roads=None,
    goods_multiplier=None,
    energy_choice=None,
    blocked_power_lines=None,
    disaster=False,
):
    """
    Evaluate one-day urban logistics.

    Districts are first assigned to their nearest supplied regional logistics hub.
    Each hub then serves districts in ascending distance order until its one-day
    last-mile driving-distance budget is exhausted.

    Service burden for one district:
        round-trip network distance × required trip count.

    This is deliberately simpler than a VRP, but it preserves the important
    teaching ideas: network distance, finite daily delivery capacity, hub
    placement, and feasibility before efficiency.
    """
    blocked_roads = blocked_roads or set()
    goods_multiplier = goods_multiplier or {
        d: 1.0 for d in districts
    }

    hub_power = {}
    operational_hubs = {}

    for h in logistics_hubs:
        if disaster and energy_choice is not None:
            status = logistics_hub_power_status(
                h,
                energy_choice,
                blocked_power_lines=blocked_power_lines,
            )
        else:
            status = {
                "operational": True,
                "power_coverage": 1.0,
                "reason": "平常時は電力供給可能",
            }

        hub_power[h] = status
        operational_hubs[h] = status["operational"]

    chain = build_logistics_supply_chain(
        logistics_hubs,
        blocked_roads=blocked_roads,
        operational_hubs=operational_hubs,
    )

    supplied_hubs = chain["supplied_hubs"]

    assignment = {
        d: {
            "hub": None,
            "distance": float("inf"),
            "trips": 0,
            "route_burden": float("inf"),
            "served": False,
            "reason": "利用可能な地域物流拠点なし",
        }
        for d in districts
    }

    # Candidate assignment to the closest supplied hub.
    candidates_by_hub = {h: [] for h in supplied_hubs}

    for d, p in districts.items():
        goods = p["goods"] * goods_multiplier[d]
        trips = logistics_trip_count(goods)

        options = []
        for h in supplied_hubs:
            dist = shortest_paths_from(
                h,
                blocked_roads=blocked_roads,
                include_external=False,
            )[d]

            if np.isfinite(dist):
                options.append((dist, h))

        if not options:
            continue

        dist, hub = min(options)
        burden = 2.0 * dist * trips

        assignment[d].update({
            "hub": hub,
            "distance": dist,
            "trips": trips,
            "route_burden": burden,
            "reason": "地域物流拠点へ割当",
        })
        candidates_by_hub[hub].append(
            (dist, d, burden, goods)
        )

    hub_usage = {}
    served_goods = 0.0
    total_goods = 0.0
    weighted_distance = 0.0
    served_districts = 0

    for d, p in districts.items():
        total_goods += p["goods"] * goods_multiplier[d]

    for h in supplied_hubs:
        used = 0.0
        served = []

        # Nearby districts first; this is the fixed behind-the-scenes dispatch rule.
        for dist, d, burden, goods in sorted(
            candidates_by_hub[h],
            key=lambda x: (x[0], x[1]),
        ):
            if used + burden <= LOGISTICS_DAILY_DISTANCE_CAPACITY + 1e-9:
                used += burden
                served.append(d)
                assignment[d]["served"] = True
                assignment[d]["reason"] = "1日配送距離制約内で配送可能"
                served_goods += goods
                weighted_distance += dist * goods
                served_districts += 1
            else:
                assignment[d]["reason"] = "1日配送距離上限を超えるため未配送"

        hub_usage[h] = {
            "used_distance": used,
            "capacity": LOGISTICS_DAILY_DISTANCE_CAPACITY,
            "served_districts": served,
        }

    coverage_pct = (
        100 * served_goods / total_goods
        if total_goods > 0 else 0.0
    )

    avg_lastmile_distance = (
        weighted_distance / served_goods
        if served_goods > 0 else float("inf")
    )

    unserved = [
        d for d, a in assignment.items()
        if not a["served"]
    ]

    return {
        "coverage_pct": coverage_pct,
        "avg_delivery_distance": avg_lastmile_distance,
        "assignment": assignment,
        "hub_usage": hub_usage,
        "hub_power": hub_power,
        "chain": chain,
        "served_districts": served_districts,
        "unserved_districts": unserved,
        "unserved_count": len(unserved),
        "trunk_distance": chain["trunk_distance"],
    }


def normal_logistics_metrics(logistics_hubs):
    result = evaluate_logistics_network(
        logistics_hubs,
        blocked_roads=set(),
        disaster=False,
    )

    # Preserve key names used elsewhere in the dashboard.
    return {
        **result,
        "local_service_ratio": result["coverage_pct"],
    }

# ============================================================
# 6. Disaster scenario
# ============================================================

def disaster_scenario():
    blocked_roads = {
        tuple(sorted(("B", "E"))),
        tuple(sorted(("D", "G"))),
        tuple(sorted(("G", "H"))),
        # One of the two external city-access links is also lost.
        tuple(sorted((EXTERNAL_NODE, "A"))),
    }

    blackout_districts = {"C", "E", "G", "H"}

    # Distribution-line outages for the electricity layer.
    blocked_power_lines = {
        tuple(sorted((GRID_NODE, "E"))),
    }

    goods_multiplier = {d: 1.0 for d in districts}
    goods_multiplier["E"] = 1.8
    goods_multiplier["G"] = 2.2
    goods_multiplier["H"] = 1.4

    return {
        "blocked_roads": blocked_roads,
        "blackout_districts": blackout_districts,
        "blocked_power_lines": blocked_power_lines,
        "goods_multiplier": goods_multiplier,
        "minimum_service_ratio": DISASTER_MIN_SERVICE_RATIO,
    }

def disaster_demand_profile_for_district(d, logistics_hubs):
    """
    災害時需要 =
      地区基礎需要 × 地区別倍率
      + 物流設備需要（物流拠点がある場合。災害時も低下させない）
    """
    p = districts[d]
    base = DEMAND_SHAPES[p["profile"]] * p["daily_energy"]
    disaster_base = base * DISASTER_BASE_DEMAND_MULTIPLIER[d]

    if d in logistics_hubs:
        return disaster_base + LOGISTICS_POWER_PROFILE

    return disaster_base


def disaster_energy_metrics(energy_choice, logistics_hubs):
    scenario = disaster_scenario()

    supported_critical_energy = 0.0
    required_critical_energy = 0.0
    supported_blackout_energy = 0.0
    required_blackout_energy = 0.0

    for d in scenario["blackout_districts"]:
        p = districts[d]

        disaster_demand = disaster_demand_profile_for_district(
            d,
            logistics_hubs,
        )

        minimum_service_demand = (
            disaster_demand
            * scenario["minimum_service_ratio"][d]
        )

        required_blackout_energy += minimum_service_demand.sum()

        tech = energy_choice[d]
        generation = TECHNOLOGIES[tech]["generation"]
        op = operate_local_energy(
            minimum_service_demand,
            generation,
        )

        supplied = (
            minimum_service_demand.sum()
            - op["grid"].sum()
        )
        supported_blackout_energy += supplied

        if p["critical"] is not None:
            required_critical_energy += minimum_service_demand.sum()
            supported_critical_energy += supplied

    critical_supply_pct = (
        100 * supported_critical_energy / required_critical_energy
        if required_critical_energy > 0 else 100.0
    )

    blackout_service_pct = (
        100 * supported_blackout_energy / required_blackout_energy
        if required_blackout_energy > 0 else 100.0
    )

    # Approximate autonomy: smallest supported-hours equivalent among blackout critical districts.
    autonomy_candidates = []

    for d in scenario["blackout_districts"]:
        if districts[d]["critical"] is None:
            continue

        demand = (
            disaster_demand_profile_for_district(
                d,
                logistics_hubs,
            )
            * scenario["minimum_service_ratio"][d]
        )

        generation = TECHNOLOGIES[
            energy_choice[d]
        ]["generation"]

        op = operate_local_energy(demand, generation)

        daily_supported = demand.sum() - op["grid"].sum()
        mean_critical_load = demand.mean()

        hours_equivalent = (
            daily_supported / mean_critical_load
            if mean_critical_load > 0 else 0.0
        )
        autonomy_candidates.append(hours_equivalent)

    autonomy_hours = (
        min(24.0, min(autonomy_candidates))
        if autonomy_candidates
        else 0.0
    )

    return {
        "critical_supply_pct": critical_supply_pct,
        "blackout_service_pct": blackout_service_pct,
        "autonomy_hours": autonomy_hours,
    }

def disaster_logistics_metrics(
    logistics_hubs,
    energy_choice,
):
    scenario = disaster_scenario()

    result = evaluate_logistics_network(
        logistics_hubs,
        blocked_roads=scenario["blocked_roads"],
        goods_multiplier=scenario["goods_multiplier"],
        energy_choice=energy_choice,
        blocked_power_lines=scenario["blocked_power_lines"],
        disaster=True,
    )

    return {
        **result,
        "goods_delivery_pct": result["coverage_pct"],
        "isolated_districts": result["unserved_districts"],
        "isolated_count": result["unserved_count"],
    }

# ============================================================
# 7. Sidebar controls
# ============================================================

with st.sidebar:
    st.header("条件設定")

    st.markdown("### ① 地域物流拠点")
    logistics_hubs = st.multiselect(
        "最大2地区を選択",
        options=LOGISTICS_ALLOWED,
        default=[],
        max_selections=2,
        format_func=lambda d: f"{d}：{districts[d]['name']}",
    )
    st.caption(
        "物流拠点を置くと、その地区に物流設備の24時間電力需要が追加されます。"
    )

    st.markdown("### ② 地域エネルギー拠点")

    energy_hub_districts = st.multiselect(
        "設置する地区（最大3か所）",
        options=list(districts.keys()),
        default=[],
        max_selections=3,
        format_func=lambda d: f"{d}：{districts[d]['name']}",
    )

    n_energy_selected = len(energy_hub_districts)
    remaining_energy_slots = 3 - n_energy_selected

    if n_energy_selected == 0:
        st.caption("0 / 3 か所選択中（あと3か所設置可能）")
    elif remaining_energy_slots > 0:
        st.caption(
            f"{n_energy_selected} / 3 か所選択中"
            f"（あと{remaining_energy_slots}か所設置可能）"
        )
    else:
        st.caption("3 / 3 か所選択中（設置可能数の上限）")

    # Default: no energy hub in any district.
    energy_choice = {
        d: "なし"
        for d in districts
    }

    # Configure the technology only for selected districts.
    for d in energy_hub_districts:
        available_tech = [
            tech for tech in ALLOWED_TECH[d]
            if tech != "なし"
        ]

        energy_choice[d] = st.selectbox(
            f"{d}：{districts[d]['name']} の電源",
            options=available_tech,
            index=0,
            key=f"energy_{d}",
        )

    with st.expander("電源特性を見る"):
        tech_rows = []
        for tech in ["PV型", "風力型", "安定電源型"]:
            t = TECHNOLOGIES[tech]
            tech_rows.append({
                "電源": t["label"],
                "1日発電量 [kWh]": round(float(t["generation"].sum()), 0),
                "CI [kg-CO₂/kWh]": t["ci"],
                "コスト [pt]": t["cost"],
            })
        st.dataframe(
            pd.DataFrame(tech_rows),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "※ CI・設備特性は意思決定の考え方を学ぶための講義用仮想値。"
        )

    with st.expander("詳細シナリオ設定"):
        mode = st.radio(
            "評価モード",
            ["平常時", "災害時"],
            index=0,
        )

        if mode == "災害時":
            st.caption(
                "一部道路の寸断・一部地区の停電・"
                "病院/避難所等の優先物資需要増加を想定。"
            )

# ============================================================
# 8. Feasibility / cost
# ============================================================

energy_hub_count = sum(
    1 for tech in energy_choice.values()
    if tech != "なし"
)

energy_cost = sum(
    TECHNOLOGIES[tech]["cost"]
    for tech in energy_choice.values()
)

logistics_cost = len(logistics_hubs) * LOGISTICS_HUB_COST
total_cost = energy_cost + logistics_cost

budget_feasible = total_cost <= TOTAL_BUDGET
hub_count_feasible = True  # UI guarantees a maximum of 3 energy-hub districts.

feasible = budget_feasible

# ============================================================
# 9. Calculate metrics
# ============================================================

# Cross-layer dependency:
# locating a logistics hub adds its operational electricity demand
# to the base district demand profile.
for d, p in districts.items():
    base = DEMAND_SHAPES[p["profile"]] * p["daily_energy"]
    if d in logistics_hubs:
        p["demand_profile"] = base + LOGISTICS_POWER_PROFILE
    else:
        p["demand_profile"] = base

normal_energy = normal_energy_metrics(energy_choice)
normal_logistics = normal_logistics_metrics(logistics_hubs)

disaster_energy = disaster_energy_metrics(energy_choice, logistics_hubs)
disaster_logistics = disaster_logistics_metrics(
    logistics_hubs,
    energy_choice,
)

minimum_service = min(
    disaster_energy["critical_supply_pct"],
    disaster_energy["blackout_service_pct"],
    disaster_logistics["goods_delivery_pct"],
)

# ============================================================
# 10. Dashboard: switchable infrastructure layers
# ============================================================

# Shared district selection across the electricity / logistics layers.
if "selected_district" not in st.session_state:
    st.session_state["selected_district"] = "D"


def update_selected_district_from_plot_event(event):
    """
    A single click on a district node updates the shared selected district.

    Returns True only when the selected district actually changed.
    Other traces (roads, hubs, power lines, logistics paths, etc.) are ignored.
    """
    try:
        points = event.selection.points
    except Exception:
        return False

    if not points:
        return False

    point = points[-1]

    try:
        customdata = point.get("customdata")
    except Exception:
        try:
            customdata = point["customdata"]
        except Exception:
            customdata = None

    if isinstance(customdata, (list, tuple)) and customdata:
        customdata = customdata[0]

    if (
        customdata in districts
        and customdata != st.session_state["selected_district"]
    ):
        st.session_state["selected_district"] = customdata
        return True

    return False


col_main, col_kpi = st.columns([3.4, 1.0], gap="large")

with col_main:

    layer = st.radio(
        "表示レイヤー",
        ["🚚 物流", "⚡ 電力"],
        horizontal=True,
        key="urban_layer",
    )

    selected_hour = 12

    scenario = disaster_scenario()

    # --------------------------------------------------------
    # 10a. Electricity layer
    # --------------------------------------------------------
    if layer == "⚡ 電力":

        st.markdown(
            f"### {'平常時' if mode == '平常時' else '災害時'}："
            "電力供給レイヤー"
        )

        blocked_power_lines = (
            scenario["blocked_power_lines"]
            if mode == "災害時"
            else set()
        )

        # The map initially uses noon; the actual selected hour is set
        # by the slider immediately below and Streamlit reruns the page.
        selected_hour = st.session_state.get("power_hour", 12)

        power_state = calculate_power_flows_for_hour(
            selected_hour,
            energy_choice,
            normal_energy,
            blocked_lines=blocked_power_lines,
        )

        fig = go.Figure()

        # Distribution feeder lines.
        first_line = True
        first_outage = True

        finite_flows = [
            v for v in power_state["line_flows"].values()
            if v is not None
        ]
        max_flow = max(finite_flows) if finite_flows else 1.0

        for a, b in POWER_LINES:
            edge = tuple(sorted((a, b)))
            xa, ya = power_node_xy(a)
            xb, yb = power_node_xy(b)

            if edge in blocked_power_lines:
                fig.add_trace(
                    go.Scatter(
                        x=[xa, xb],
                        y=[ya, yb],
                        mode="lines",
                        name="配電線断線",
                        legendgroup="power_outage",
                        showlegend=first_outage,
                        line=dict(
                            width=5,
                            dash="dash",
                            color="#d62728",
                        ),
                        hovertemplate="配電線断線<extra></extra>",
                    )
                )
                first_outage = False
            else:
                flow = power_state["line_flows"][(a, b)]
                width = 1.5 + 6.0 * (flow / max(max_flow, 1e-6))

                fig.add_trace(
                    go.Scatter(
                        x=[xa, xb],
                        y=[ya, yb],
                        mode="lines",
                        name="配電系統",
                        legendgroup="power_line",
                        showlegend=first_line,
                        line=dict(
                            width=width,
                            color="rgba(79,129,189,0.58)",
                        ),
                        hovertemplate=(
                            f"{a}–{b}<br>"
                            + (
                                f"上流からの供給：{flow:.1f} kWh/h"
                                if flow is not None
                                else "断線"
                            )
                            + "<extra></extra>"
                        ),
                    )
                )
                first_line = False

        # Grid node
        fig.add_trace(
            go.Scatter(
                x=[GRID_SUPPLY["x"]],
                y=[GRID_SUPPLY["y"]],
                mode="markers+text",
                text=["系統"],
                textposition="top center",
                name="系統電力",
                marker=dict(
                    size=26,
                    symbol="star",
                    color="#4f81bd",
                ),
                hovertext=[
                    "上位系統<br>"
                    "地域電源・蓄電池で不足する電力を配電系統へ供給"
                ],
                hoverinfo="text",
            )
        )

        # District nodes.
        xvals, yvals, labels, sizes, hovertexts = [], [], [], [], []

        for d, p in districts.items():
            demand_h = float(p["demand_profile"][selected_hour])
            op = normal_energy["district_ops"][d]
            local_h = float(op["local_supply"][selected_hour])
            gen_h = float(
                TECHNOLOGIES[energy_choice[d]]["generation"][selected_hour]
            )

            connected = d in power_state["reachable"]
            residual = power_state["local_net"][d]
            grid_h = residual if connected else 0.0

            xvals.append(p["x"])
            yvals.append(p["y"])
            labels.append(d)
            sizes.append(22 + demand_h * 1.2)

            hovertexts.append(
                f"<b>{d}：{p['name']}</b><br>"
                f"{selected_hour}:00需要：{demand_h:.1f} kWh/h<br>"
                f"地域発電：{gen_h:.1f} kWh/h<br>"
                f"地域供給（蓄電池含む）：{local_h:.1f} kWh/h<br>"
                f"上流系統から必要：{grid_h:.1f} kWh/h<br>"
                f"系統接続：{'接続' if connected else '切離'}<br>"
                f"電源：{energy_choice[d]}"
            )

        fig.add_trace(
            go.Scatter(
                x=xvals,
                y=yvals,
                mode="markers+text",
                text=labels,
                customdata=list(districts.keys()),
                textposition="middle center",
                name="地区需要",
                marker=dict(
                    size=sizes,
                    color="#f2f2f2",
                    line=dict(width=2, color="#666666"),
                ),
                hovertext=hovertexts,
                hoverinfo="text",
            )
        )

        # Energy hubs remain visible.
        tech_symbols = {
            "PV型": "diamond-open",
            "風力型": "triangle-up-open",
            "安定電源型": "hexagon-open",
        }
        tech_colors = {
            "PV型": "#f2b134",
            "風力型": "#5aa6d1",
            "安定電源型": "#7a6fb0",
        }

        for tech in ["PV型", "風力型", "安定電源型"]:
            hubs = [
                d for d, t in energy_choice.items()
                if t == tech
            ]
            if not hubs:
                continue

            fig.add_trace(
                go.Scatter(
                    x=[districts[d]["x"] for d in hubs],
                    y=[districts[d]["y"] for d in hubs],
                    mode="markers",
                    name=tech,
                    marker=dict(
                        size=38,
                        symbol=tech_symbols[tech],
                        color=tech_colors[tech],
                        line=dict(width=4),
                    ),
                    hovertext=[
                        (
                            f"{d}：{tech}<br>"
                            f"{selected_hour}:00発電："
                            f"{TECHNOLOGIES[tech]['generation'][selected_hour]:.1f} kWh/h"
                        )
                        for d in hubs
                    ],
                    hoverinfo="text",
                )
            )

        # Logistics hubs remain visible even on the electricity layer.
        if logistics_hubs:
            fig.add_trace(
                go.Scatter(
                    x=[districts[d]["x"] for d in logistics_hubs],
                    y=[districts[d]["y"] for d in logistics_hubs],
                    mode="markers",
                    name="物流拠点",
                    marker=dict(
                        size=25,
                        symbol="square-open",
                        color="#2ca02c",
                        line=dict(width=3),
                    ),
                    hovertext=[
                        f"{d}：地域物流拠点"
                        for d in logistics_hubs
                    ],
                    hoverinfo="text",
                )
            )

        fig.update_layout(
            template="plotly_white",
            height=500,
            xaxis=dict(
                visible=False,
                range=[-1.2, 6.8],
                fixedrange=True,
            ),
            yaxis=dict(
                visible=False,
                range=[-0.5, 3.8],
                scaleanchor="x",
                scaleratio=1,
                fixedrange=True,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            clickmode="event+select",
            margin=dict(l=10, r=10, t=35, b=10),
        )

        power_event = st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
            on_select="rerun",
            selection_mode="points",
            key="power_network_map",
        )

        if update_selected_district_from_plot_event(power_event):
            st.rerun()

        st.caption(
            "地区ノードをクリックすると、下の詳細がすぐ切り替わります。 "
            "電力レイヤー：青線＝配電系統（太いほど選択時刻の上流供給が大きい）／ "
            "赤破線＝災害時の配電線断線。地域電源・物流設備は常時表示。"
        )

        # Time slider intentionally placed below the network map.
        selected_hour = st.slider(
            "電力供給状態を見る時刻",
            min_value=0,
            max_value=23,
            value=st.session_state.get("power_hour", 12),
            step=1,
            format="%d:00",
            key="power_hour",
        )

        selected_district = st.session_state["selected_district"]

        st.markdown(
            f"#### 選択中：{selected_district}："
            f"{districts[selected_district]['name']}"
        )

        p = districts[selected_district]
        tech = energy_choice[selected_district]
        generation = TECHNOLOGIES[tech]["generation"]
        op = normal_energy["district_ops"][selected_district]

        fig_profile = go.Figure()

        base_demand_profile = (
            DEMAND_SHAPES[p["profile"]] * p["daily_energy"]
        )

        fig_profile.add_trace(
            go.Scatter(
                x=hours,
                y=base_demand_profile,
                mode="lines",
                name="地区基礎需要",
            )
        )

        if selected_district in logistics_hubs:
            fig_profile.add_trace(
                go.Scatter(
                    x=hours,
                    y=LOGISTICS_POWER_PROFILE,
                    mode="lines",
                    name="物流設備需要",
                )
            )

        fig_profile.add_trace(
            go.Scatter(
                x=hours,
                y=p["demand_profile"],
                mode="lines+markers",
                name="合計電力需要",
            )
        )

        if tech != "なし":
            fig_profile.add_trace(
                go.Scatter(
                    x=hours,
                    y=generation,
                    mode="lines",
                    name=f"{tech} 発電",
                )
            )

        fig_profile.add_trace(
            go.Bar(
                x=hours,
                y=op["grid"],
                name="系統から購入",
                opacity=0.35,
            )
        )

        fig_profile.add_vline(
            x=selected_hour,
            line_dash="dash",
            annotation_text=f"{selected_hour}:00",
            annotation_position="top",
        )

        fig_profile.update_layout(
            template="plotly_white",
            height=270,
            title=(
                f"{selected_district}：{p['name']} — "
                "24時間の需要・地域発電・系統購入"
            ),
            xaxis=dict(
                title="時刻",
                tickvals=list(range(0, 24, 2)),
            ),
            yaxis=dict(title="kWh / h"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
            ),
            margin=dict(l=40, r=15, t=65, b=35),
        )

        st.plotly_chart(
            fig_profile,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # --------------------------------------------------------
    # 10b. Logistics layer
    # --------------------------------------------------------
    else:

        st.markdown(
            f"### {'平常時' if mode == '平常時' else '災害時'}："
            "物流・道路レイヤー"
        )

        fig = go.Figure()

        blocked_roads = (
            scenario["blocked_roads"]
            if mode == "災害時"
            else set()
        )

        logistics_state = (
            normal_logistics
            if mode == "平常時"
            else disaster_logistics
        )

        first_road = True
        first_external = True
        first_blocked = True

        all_draw_links = (
            [(a, b, "city") for a, b in roads]
            + [(a, b, "external") for a, b in EXTERNAL_ROADS]
        )

        for a, b, link_type in all_draw_links:
            blocked = tuple(sorted((a, b))) in blocked_roads
            xa, ya = node_xy(a)
            xb, yb = node_xy(b)

            if blocked:
                name = "通行不能"
                legendgroup = "blocked"
                showlegend = first_blocked
                line = dict(
                    width=5,
                    dash="dash",
                    color="#d62728",
                )
                first_blocked = False

            elif link_type == "external":
                name = "都市外物流アクセス"
                legendgroup = "external"
                showlegend = first_external
                line = dict(
                    width=3,
                    color="#1f4e79",
                )
                first_external = False

            else:
                name = "都市内道路"
                legendgroup = "road"
                showlegend = first_road
                line = dict(
                    width=2,
                    color="#b0b0b0",
                )
                first_road = False

            fig.add_trace(
                go.Scatter(
                    x=[xa, xb],
                    y=[ya, yb],
                    mode="lines",
                    name=name,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    line=line,
                    hovertemplate=(
                        f"{a}–{b}<br>"
                        f"リンク距離：{euclidean(a,b):.2f}"
                        "<extra></extra>"
                    ),
                )
            )

        # External long-haul logistics center
        fig.add_trace(
            go.Scatter(
                x=[EXTERNAL_LOGISTICS["x"]],
                y=[EXTERNAL_LOGISTICS["y"]],
                mode="markers+text",
                text=["都市外物流"],
                textposition="bottom center",
                name="都市外物流センター",
                marker=dict(
                    size=24,
                    symbol="square",
                    color="#1f4e79",
                ),
                hovertext=[
                    "都市外物流センター<br>"
                    "長距離ロジスティクスの都市側ゲートウェイ<br>"
                    "A・Fから都市道路網へ接続"
                ],
                hoverinfo="text",
            )
        )

        # District nodes
        xvals, yvals, labels, sizes, hovertexts = [], [], [], [], []

        for d, p in districts.items():
            a = logistics_state["assignment"][d]

            xvals.append(p["x"])
            yvals.append(p["y"])
            labels.append(d)
            sizes.append(20 + p["goods"] / 7)

            hub_text = a["hub"] if a["hub"] is not None else "—"
            dist_text = (
                f"{a['distance']:.2f}"
                if np.isfinite(a["distance"])
                else "—"
            )

            hovertexts.append(
                f"<b>{d}：{p['name']}</b><br>"
                f"物資需要：{p['goods']} unit/day<br>"
                f"担当地域物流拠点：{hub_text}<br>"
                f"ラストワンマイル距離：{dist_text}<br>"
                f"必要往復距離："
                + (
                    f"{a['route_burden']:.2f}"
                    if np.isfinite(a["route_burden"])
                    else "—"
                )
                + "<br>"
                f"配送可否：{'可' if a['served'] else '不可'}<br>"
                f"{a['reason']}"
            )

        fig.add_trace(
            go.Scatter(
                x=xvals,
                y=yvals,
                mode="markers+text",
                text=labels,
                customdata=list(districts.keys()),
                textposition="middle center",
                name="地区",
                marker=dict(
                    size=sizes,
                    color="#f2f2f2",
                    line=dict(width=2, color="#666666"),
                ),
                hovertext=hovertexts,
                hoverinfo="text",
            )
        )

        # Energy equipment stays visible on both layers.
        tech_symbols = {
            "PV型": "diamond-open",
            "風力型": "triangle-up-open",
            "安定電源型": "hexagon-open",
        }
        tech_colors = {
            "PV型": "#f2b134",
            "風力型": "#5aa6d1",
            "安定電源型": "#7a6fb0",
        }

        for tech in ["PV型", "風力型", "安定電源型"]:
            hubs = [
                d for d, t in energy_choice.items()
                if t == tech
            ]

            if hubs:
                fig.add_trace(
                    go.Scatter(
                        x=[districts[d]["x"] for d in hubs],
                        y=[districts[d]["y"] for d in hubs],
                        mode="markers",
                        name=tech,
                        marker=dict(
                            size=32,
                            symbol=tech_symbols[tech],
                            color=tech_colors[tech],
                            line=dict(width=3),
                        ),
                        hovertext=[
                            f"{d}：{TECHNOLOGIES[tech]['label']}"
                            for d in hubs
                        ],
                        hoverinfo="text",
                    )
                )

        # Regional logistics hubs and operating status.
        if logistics_hubs:
            hub_hover = []
            for h in logistics_hubs:
                status = logistics_state["hub_power"].get(
                    h,
                    {
                        "operational": False,
                        "power_coverage": 0.0,
                        "reason": "未評価",
                    },
                )
                usage = logistics_state["hub_usage"].get(
                    h,
                    {
                        "used_distance": 0.0,
                        "capacity": LOGISTICS_DAILY_DISTANCE_CAPACITY,
                        "served_districts": [],
                    },
                )

                hub_hover.append(
                    f"{h}：地域物流拠点<br>"
                    f"稼働：{'可' if status['operational'] else '不可'}<br>"
                    f"電力状態：{status['reason']}<br>"
                    f"1日配送距離：{usage['used_distance']:.1f}"
                    f" / {LOGISTICS_DAILY_DISTANCE_CAPACITY:.1f}<br>"
                    f"担当地区："
                    + (
                        ", ".join(usage["served_districts"])
                        if usage["served_districts"]
                        else "なし"
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=[districts[d]["x"] for d in logistics_hubs],
                    y=[districts[d]["y"] for d in logistics_hubs],
                    mode="markers",
                    name="地域物流拠点",
                    marker=dict(
                        size=35,
                        symbol="square-open",
                        color="#2ca02c",
                        line=dict(width=4),
                    ),
                    hovertext=hub_hover,
                    hoverinfo="text",
                )
            )

        # Trunk transport: EXT -> gateway hub -> other regional hubs
        first_trunk = True
        for hub, path in logistics_state["chain"]["trunk_paths"].items():
            add_network_path(
                fig,
                path,
                name="幹線・拠点間輸送",
                legendgroup="trunk",
                showlegend=first_trunk,
                line=dict(
                    width=5,
                    color="rgba(31,78,121,0.70)",
                ),
            )
            first_trunk = False

        # Last-mile paths for actually served districts
        first_lastmile = True

        for d, a in logistics_state["assignment"].items():
            if not a["served"] or a["hub"] is None:
                continue
            if d == a["hub"]:
                continue

            path = shortest_path_nodes(
                a["hub"],
                d,
                blocked_roads=blocked_roads,
                include_external=False,
            )

            add_network_path(
                fig,
                path,
                name="ラストワンマイル配送",
                legendgroup="lastmile",
                showlegend=first_lastmile,
                line=dict(
                    width=4,
                    color="rgba(44,160,44,0.58)",
                ),
            )
            first_lastmile = False

        fig.update_layout(
            template="plotly_white",
            height=500,
            xaxis=dict(
                visible=False,
                range=[-1.2, 6.8],
                fixedrange=True,
            ),
            yaxis=dict(
                visible=False,
                range=[-0.5, 3.8],
                scaleanchor="x",
                scaleratio=1,
                fixedrange=True,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            clickmode="event+select",
            margin=dict(l=10, r=10, t=35, b=10),
        )

        logistics_event = st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
            on_select="rerun",
            selection_mode="points",
            key="logistics_network_map",
        )

        if update_selected_district_from_plot_event(logistics_event):
            st.rerun()

        st.caption(
            "地区ノードをクリックすると、下の物流サービス詳細がすぐ切り替わります。 "
            "物流レイヤー：グレー＝道路 ／ 濃青＝都市外・拠点間の幹線輸送 ／ "
            "緑＝実際に1日で配送可能なラストワンマイル ／ 赤破線＝通行不能。"
            f" 各地域物流拠点の1日配送距離上限は "
            f"{LOGISTICS_DAILY_DISTANCE_CAPACITY:.0f} distance-unit。"
        )

        selected_district = st.session_state["selected_district"]

        st.markdown(
            f"#### 選択中：{selected_district}："
            f"{districts[selected_district]['name']}"
        )

        p = districts[selected_district]
        a = logistics_state["assignment"][selected_district]

        if a["served"]:
            st.success(
                f"{selected_district}地区は地域物流拠点 {a['hub']} が担当。"
                f" 片道ネットワーク距離 {a['distance']:.2f}、"
                f"1日の配送負担 {a['route_burden']:.2f}。"
            )
        else:
            st.error(
                f"{selected_district}地区は1日配送制約の中では未配送です。"
                f" 理由：{a['reason']}"
            )

        if logistics_hubs:
            with st.expander("物流拠点ごとの1日配送状況"):
                rows = []
                for h in logistics_hubs:
                    usage = logistics_state["hub_usage"].get(
                        h,
                        {
                            "used_distance": 0.0,
                            "served_districts": [],
                        },
                    )
                    status = logistics_state["hub_power"].get(
                        h,
                        {
                            "operational": False,
                            "reason": "未供給",
                        },
                    )

                    rows.append({
                        "物流拠点": h,
                        "稼働": "可" if status["operational"] else "不可",
                        "電力状態": status["reason"],
                        "1日配送距離": round(usage["used_distance"], 2),
                        "上限": LOGISTICS_DAILY_DISTANCE_CAPACITY,
                        "担当地区": ", ".join(usage["served_districts"]),
                    })

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

# ============================================================
# 11. KPI panel
# ============================================================

with col_kpi:

    st.markdown("### 予算・制約")

    st.metric(
        "使用予算",
        f"{total_cost} / {TOTAL_BUDGET} pt"
    )

    st.metric(
        "エネルギー拠点",
        f"{energy_hub_count} / 3"
    )

    st.metric(
        "物流拠点",
        f"{len(logistics_hubs)} / 2"
    )

    if not feasible:
        msgs = []
        if not budget_feasible:
            msgs.append("予算超過")
        st.error("実行不可能：" + " / ".join(msgs))
    else:
        st.success("配置制約を満たしています")

    st.divider()

    if mode == "平常時":
        st.markdown("### 平常時の評価")

        st.metric(
            "CO₂削減量",
            f"{normal_energy['co2_reduction']:.0f} kg/day",
        )
        st.caption(
            f"地域電源由来CO₂：{normal_energy['local_co2']:.0f} kg/day"
        )

        st.metric(
            "系統購入削減量",
            f"{normal_energy['grid_reduction']:.0f} kWh/day",
        )

        st.metric(
            "地域低炭素電源供給率",
            f"{normal_energy['local_supply_ratio']:.0f} %",
        )

        st.metric(
            "1日配送可能物資率",
            f"{normal_logistics['coverage_pct']:.0f} %",
        )

        avg_lm = normal_logistics["avg_delivery_distance"]
        st.metric(
            "平均ラストワンマイル距離",
            f"{avg_lm:.2f}" if np.isfinite(avg_lm) else "—",
        )

        st.metric(
            "幹線・拠点間輸送距離",
            f"{normal_logistics['trunk_distance']:.2f}",
        )

        st.caption(
            "平常時：全地区は系統電力と都市外物流でサービス可能。"
            "地域拠点は低炭素化・効率化のための追加施策です。"
        )

    else:
        st.markdown("### 災害時の評価")

        st.metric(
            "重要施設電力維持率",
            f"{disaster_energy['critical_supply_pct']:.0f} %",
        )

        st.metric(
            "停電地区最低負荷供給率",
            f"{disaster_energy['blackout_service_pct']:.0f} %",
        )

        st.metric(
            "優先物資到達率",
            f"{disaster_logistics['goods_delivery_pct']:.0f} %",
        )

        st.metric(
            "孤立地区数",
            f"{disaster_logistics['isolated_count']} 地区",
        )

        st.metric(
            "最低サービス維持率",
            f"{minimum_service:.0f} %",
        )

        st.metric(
            "重要施設 自立継続時間",
            f"{disaster_energy['autonomy_hours']:.1f} h",
        )

        st.caption(
            "災害時：CO₂や平常時配送効率より、"
            "重要サービス・到達性・継続性を優先評価します。"
        )

        st.divider()
        st.markdown("#### 補助指標")
        st.caption(
            f"平常時CO₂削減："
            f"{normal_energy['co2_reduction']:.0f} kg/day"
        )
        st.caption(
            f"平常時平均ラストワンマイル距離："
            + (
                f"{normal_logistics['avg_delivery_distance']:.2f}"
                if np.isfinite(normal_logistics["avg_delivery_distance"])
                else "—"
            )
        )

# ============================================================
# 12. Details
# ============================================================

with st.expander("8地区の条件と配置制約を見る"):
    rows = []

    for d, p in districts.items():
        rows.append({
            "地区": d,
            "名称": p["name"],
            "人口": p["population"],
            "基礎電力需要 [kWh/day]": p["daily_energy"],
            "災害時基礎需要倍率": DISASTER_BASE_DEMAND_MULTIPLIER[d],
            "災害時最低サービス率": DISASTER_MIN_SERVICE_RATIO[d],
            "物流設備需要 [kWh/day]": (
                round(float(LOGISTICS_POWER_PROFILE.sum()), 1)
                if d in logistics_hubs else 0.0
            ),
            "平常時合計需要 [kWh/day]": round(float(p["demand_profile"].sum()), 1),
            "災害時合計需要 [kWh/day]": round(
                float(
                    disaster_demand_profile_for_district(
                        d,
                        logistics_hubs,
                    ).sum()
                ),
                1,
            ),
            "物資需要 [unit/day]": p["goods"],
            "重要施設": p["critical"] if p["critical"] else "—",
            "選択電源": energy_choice[d],
            "選択可能電源": " / ".join(ALLOWED_TECH[d]),
            "物流拠点可否": "可" if d in LOGISTICS_ALLOWED else "不可",
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("モデルの基本ルールを見る"):
    st.markdown(
        """
- **平常時の電力**：全地区が系統電力に接続されており、地域エネルギー拠点がなくても需要は満たされる。
- **電力レイヤー**：A〜Hは配電系統トポロジーで接続され、上位系統から下流へ電力が供給される。選択した時刻について、配電線の太さはその線を通る上流供給量を表す。
- **物流→電力の依存関係**：地域物流拠点を設置すると、冷蔵・情報通信・荷役等を表す24時間の物流設備需要がその地区の電力需要に加算される。
- **地域エネルギー拠点**：選択した電源の24時間発電プロファイルを持ち、設置地区の需要を優先して供給する。
- **地域電源のCO₂**：PV型・風力型・安定電源型（コージェネ相当）はそれぞれ異なる講義用CIを持ち、実際に地域供給した電力量に応じてCO₂排出量へ反映する。
- **蓄電池**：各エネルギー拠点に付随し、余剰発電を充電・不足時に放電する。運用はDigital Twin内の固定ルールで自動決定する。
- **不足電力**：平常時は系統から購入する。CO₂排出量は実際の系統購入電力量と時間別CIから計算する。
- **都市外物流センター**：長距離ロジスティクスの拠点とし、A・Fを都市への入口として地域物流ネットワークへ物資を投入する。
- **地域物流拠点**：都市外物流センターまたは別の地域物流拠点から物資を受け取り、他拠点への中継輸送と周辺地区へのラストワンマイル集配を担う。
- **物流リンク**：都市道路リンクには距離を持たせ、物流の配送距離は表示されている道路ネットワーク上の最短距離から計算する。
- **1日配送制約**：各地域物流拠点には1日に走行可能な配送距離の上限があり、近い地区から順に担当する固定ルールで配送可否を判定する。
- **平常時の物流KPI**：まず1日の配送制約の中でどれだけの物資を配送できるかを評価し、その上で平均ラストワンマイル距離や幹線・拠点間輸送距離を効率性指標として見る。
- **物流レイヤー**：濃青で都市外・地域物流拠点間の幹線輸送、緑で実際に担当する地区へのラストワンマイル配送を表示する。
- **災害時の地区基礎需要**：地区の用途に応じて需要が変化する。住宅・商業・学校等では低下する一方、病院地区は低下せず、避難所地区では増加する。
- **物流設備需要**：物流拠点の冷蔵・情報通信・荷役等の需要は災害時も低下しない。
- **最低サービス需要**：実際に発生する災害時需要とは別に、地区ごとに最低限維持したいサービス率を設定してレジリエンスKPIを評価する。
- **災害時**：一部道路・外部アクセスに加えて一部配電線も断線する。電力レイヤーでは系統から切り離される地区、物流レイヤーでは物資到達が困難になる地区をそれぞれ確認する。
- **予算制約**：電源タイプ・物流拠点ごとの講義用コストを合計し、100 points以内とする。
        """
    )

# ============================================================
# 13. Mission
# ============================================================

if mode == "平常時":
    st.info(
        "MISSION：予算100 points以内で、最大3か所の地域エネルギー拠点と"
        "最大2か所の物流拠点を配置してください。"
        "物流では1日配送可能性と配送距離、電力ではCO₂と系統依存を確認し、"
        "両レイヤーを切り替えて都市全体として良い配置を探してください。"
    )
else:
    isolated = disaster_logistics["isolated_districts"]
    isolated_text = (
        "　孤立地区：" + ", ".join(isolated)
        if isolated else ""
    )

    st.warning(
        "DISASTER MISSION：平常時に考えた配置は、道路寸断・系統停電下でも"
        "重要サービスを維持できるでしょうか？"
        + isolated_text
    )
