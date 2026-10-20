"""Independent bounded uniform draws; each draw defines one complete trajectory."""
import math
import numpy as np

PARAMS=['peak','onset','width','tc','tr','s0','v0']


def samples(base, spreads, n, seed):
    # Drawing rows keeps the first trajectories unchanged when n is increased.
    z=np.random.default_rng(seed).uniform(-1,1,(n,7))
    result={}
    for j,key in enumerate(PARAMS):
        if key=='onset':result[key]=base[key]+spreads[key]*z[:,j]
        else:result[key]=base[key]*(1+spreads[key]*z[:,j])
    return result


def simulate_ensemble(base,spreads,n,seed,step_s=30):
    draws=samples(base,spreads,n,seed)
    end=base['duration']*3600
    times=np.linspace(0,end,math.ceil(end/step_s)+1)
    dt=float(times[1]-times[0])
    tc,tr=draws['tc']*3600,draws['tr']*3600
    ec,er=np.exp(-dt/tc),np.exp(-dt/tr)
    delta=1/tr-1/tc
    kernel=np.empty(n)
    near=np.abs(delta*dt)<1e-9
    kernel[near]=er[near]*dt
    kernel[~near]=er[~near]*np.expm1(delta[~near]*dt)/delta[~near]
    s=draws['s0']*base['area']*1000
    v=draws['v0']*10000
    initial=s+v
    cumulative_input=np.zeros(n)
    cumulative_output=np.zeros(n)
    ss=np.empty((n,len(times)))
    hh=np.empty_like(ss)
    ss[:,0],hh[:,0]=s,base['h0']+v/(base['river_area']*10000)
    for i in range(1,len(times)):
        tmid=(times[i]+times[i-1])/7200
        z=(tmid-draws['onset'])/draws['width']
        rain=draws['peak']*np.maximum(0,1-np.abs(2*z-1))
        u=base['area']*rain/3.6
        sn=s*ec+u*tc*(-np.expm1(-dt/tc))
        vn=v*er+u*tr*(-np.expm1(-dt/tr))+(s/tc-u)*kernel
        cumulative_input+=u*dt
        cumulative_output+=s+v+u*dt-sn-vn
        s,v=sn,vn
        ss[:,i],hh[:,i]=s,base['h0']+v/(base['river_area']*10000)
    return dict(t=times/3600,h=hh,s=ss,draws=draws,
                max_balance=float(np.max(np.abs(s+v-initial-cumulative_input+cumulative_output))))


def wilson(k,n):
    z=1.959963984540054
    p=k/n
    denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return max(0.,center-half),min(1.,center+half)


def values_at(matrix,times,t):
    idx=int(np.searchsorted(times,t,side='right')-1)
    idx=max(0,min(idx,len(times)-1))
    if idx==len(times)-1:return matrix[:,idx],idx
    alpha=(t-times[idx])/(times[idx+1]-times[idx])
    return matrix[:,idx]*(1-alpha)+matrix[:,idx+1]*alpha,idx


def event_counts(h,times,threshold,horizon):
    current,idx=values_at(h,times,horizon)
    maxima=np.maximum(h[:,:idx+1].max(axis=1),current)
    return int(np.count_nonzero(maxima>threshold)),maxima
