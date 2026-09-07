# Mini Digital Twin Teaching

Interactive Streamlit applications for learning digital twins and decision support.

This repository contains small, interactive digital twin examples designed for use in lectures and exercises.
Each application provides a simplified virtual system in which users can explore how different conditions and decisions affect system behavior and performance.

## Apps

### Energy

#### `energy/01_basic_pv_battery/`
**Mini Energy Digital Twin**

A simple residential energy system with electricity demand, photovoltaic (PV) generation, and battery energy storage.

Users can explore how system configuration and operation affect:
- Grid electricity consumption
- PV utilization
- Battery operation
- CO₂ emissions

#### `energy/02_basic_ev_charge/`
**Mini EV Energy Digital Twin**

A household EV charging example for exploring time-dependent energy decisions.

Users can investigate:
- EV charging schedules
- Charging feasibility under vehicle-use constraints
- Electricity cost
- CO₂ emissions
- Effects of forecast uncertainty

### Mobility

#### `mobility/01_bus_electrification/`
**Mini Mobility Digital Twin**

A bus electrification planning example with multiple vehicles and a shared charging resource.

Users can explore:
- Which buses to electrify
- Vehicle operation and charging opportunities
- Shared charger constraints
- Charger capacity
- Grid carbon-intensity scenarios
- CO₂ reduction and operational feasibility

### Urban

#### `urban/01_urban_planning/`
**Mini Urban & Resilience Digital Twin**

A simplified urban system integrating logistics and electricity infrastructure.

Users can explore:
- Placement of regional logistics hubs
- Placement and selection of local energy resources
- Urban logistics and last-mile delivery
- Electricity demand and distribution-network operation
- Interdependencies between logistics and electricity
- Infrastructure cost and feasibility
- Changes in system performance under disaster scenarios

The application provides separate **logistics** and **electricity** layers, allowing the same urban system to be examined from different infrastructure perspectives.

## Purpose

These applications are intended to support learning about:

- Digital twins as simplified representations of real-world systems
- System states, inputs, constraints, and outputs
- Decision-making through "what-if" simulation
- Operational feasibility
- Trade-offs among multiple performance indicators
- Interactions between multiple components and infrastructure layers
- Decision-making under changing conditions and uncertainty

The models are intentionally simplified for educational purposes and are not intended to reproduce actual systems in full detail.
