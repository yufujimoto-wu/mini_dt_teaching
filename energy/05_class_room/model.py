"""One-node sensible-heat teaching model. All profiles are explicit scenarios, not measured data."""
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
