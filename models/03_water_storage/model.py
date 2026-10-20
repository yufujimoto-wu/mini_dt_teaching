"""Two linear reservoirs: SI internally, lecture-5 equations."""
import math
from bisect import bisect_right


def rain_at(t, peak, onset, width):
    """Triangular effective rainfall [mm/h]; t/onset/width in hours."""
    z = (t-onset)/width
    return peak * max(0.0, 1-abs(2*z-1)) if 0 <= z <= 1 else 0.0


def step(s, v, u, tc, tr, dt):
    ec, er = math.exp(-dt/tc), math.exp(-dt/tr)
    delta = 1/tr-1/tc
    kernel = er*dt if abs(delta*dt)<1e-9 else er*math.expm1(delta*dt)/delta
    s1 = s*ec + u*tc*(-math.expm1(-dt/tc))
    v1 = v*er + u*tr*(-math.expm1(-dt/tr)) + (s/tc-u)*kernel
    return s1, v1


def simulate(area, river_area, tc_h, tr_h, h0, s0, v0, peak, onset, width, duration):
    tc, tr = tc_h*3600, tr_h*3600
    # Include rainfall breakpoints; midpoint forcing then integrates its volume exactly.
    end = duration*3600
    nodes = {0.0, end}
    nodes.update(float(i*10) for i in range(1,math.ceil(end/10)))
    nodes.update(t*3600 for t in [onset,onset+width/2,onset+width] if 0<t<duration)
    times = sorted(nodes)
    result = {k:[] for k in ['t','s','v','h','rain','qin','qout','input','output','balance']}
    s,v,cin,cout = s0,v0,0.,0.
    for i,t in enumerate(times):
        if i:
            dt = t-times[i-1]
            u = area*rain_at((t+times[i-1])/7200,peak,onset,width)/3_600_000
            sn,vn = step(s,v,u,tc,tr,dt)
            cin += u*dt
            cout += s+v+u*dt-sn-vn
            s,v = sn,vn
        values = [t/3600,s,v,h0+v/river_area,rain_at(t/3600,peak,onset,width),s/tc,v/tr,cin,cout,s+v-(s0+v0+cin-cout)]
        for key,val in zip(result,values):result[key].append(val)
    return result


def at(data,t):
    i = min(max(0,bisect_right(data['t'],t)-1),len(data['t'])-2)
    alpha = (t-data['t'][i])/(data['t'][i+1]-data['t'][i])
    return {key: vals[i]+alpha*(vals[i+1]-vals[i]) for key,vals in data.items()}
