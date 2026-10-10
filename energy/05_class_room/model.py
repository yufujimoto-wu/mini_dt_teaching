"""Teaching model: one well-mixed room, one-minute heat balance; no live observations."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Conditions:
    initial_temp: float = 31.0
    outdoor_temp: float = 33.0
    people: int = 40
    capacity: float = 20.0  # thermal kW
    capacitance: float = 2.5  # effective room + furnishings kWh/K
    conductance: float = 0.6  # envelope + fixed ventilation kW/K
    solar: float = 1.5  # kW
    cop: float = 3.5
    threshold: float = 28.0
    emission_factor: float = 0.4  # kg-CO2/kWh, illustrative constant


def simulate(c: Conditions, lead: int, setpoint: float):
    """07:30–11:00; class 09:00–11:00. Samples are interval starts."""
    if not (0 <= lead <= 90 and c.cop > 0 and c.capacitance > 0):
        raise ValueError('Invalid simulation parameters')
    temp = c.initial_temp
    rows = []
    energy = 0.0
    for minute in range(-90, 121):
        occupied = 0 <= minute < 120
        on = -lead <= minute < 120
        gain = c.conductance * (c.outdoor_temp - temp) + c.solar + (0.075 * c.people if occupied else 0)
        # Ideal modulating controller with 10-minute response, bounded cooling capacity.
        cooling = np.clip(gain + c.capacitance * (temp - setpoint) / (10/60), 0, c.capacity) if on else 0.0
        power = float(cooling / c.cop)
        rows.append((minute, temp, power, energy, gain, float(cooling)))
        if minute < 120:
            energy += power / 60
            temp += (gain - cooling) / c.capacitance / 60
    return pd.DataFrame(rows, columns=['minute', 'temperature', 'power', 'energy', 'heat_gain', 'cooling'])


def metrics(df, c):
    teaching = df[(df.minute >= 0) & (df.minute < 120)]
    # Count one-minute intervals using midpoint temperature.
    mids = (df.temperature.to_numpy()[:-1] + df.temperature.to_numpy()[1:]) / 2
    during = (df.minute.to_numpy()[:-1] >= 0)
    return dict(start=float(teaching.iloc[0].temperature),
                hot=int(np.sum((mids > c.threshold) & during)),
                energy=float(df.iloc[-1].energy),
                co2=float(df.iloc[-1].energy * c.emission_factor))
