"""Fixed uniform marginal samples; change only the pairing of rainfall peaks."""
import numpy as np
from model import simulate


def peak_pairs(n,seed,center,spread,target):
    rng=np.random.default_rng(seed)
    z1,z2=rng.standard_normal((n,2)).T
    levels=center*(1+spread*(2*(np.arange(n)+.5)/n-1))
    rank1=np.argsort(np.argsort(z1))
    first=levels[rank1]
    if spread==0 or center==0:
        return first,first.copy(),None
    standardized=(rank1-(n-1)/2)/np.sqrt((n*n-1)/12)
    def pair(r):
        score=r*z1+np.sqrt(max(0.,1-r*r))*z2
        ranks=np.argsort(np.argsort(score))
        corr=float(np.mean(standardized*(ranks-(n-1)/2)/np.sqrt((n*n-1)/12)))
        return levels[ranks],corr
    if abs(target)==1:
        second,corr=pair(target)
        return first,second,corr
    # Finite-sample calibration by ranks; the empirical marginals never change.
    lo,hi=-1.,1.
    best=None
    for _ in range(40):
        mid=(lo+hi)/2
        second,corr=pair(mid)
        if best is None or abs(corr-target)<abs(best[1]-target):best=(second,corr)
        if corr<target:lo=mid
        else:hi=mid
    return first,best[0],best[1]


def simulate_two(base,n,seed,rho):
    p1,p2,corr=peak_pairs(n,seed,base['peak'],base['spread'],rho)
    profiles=[]
    for area,tc in [(base['area1'],base['tc1']),(base['area2'],base['tc2'])]:
        args=dict(area=area*1e6,river_area=base['river_area']*1e4,tc_h=tc,tr_h=base['tr'],h0=0.,s0=base['s0']*area*1000,v0=base['v0']*10000/2,onset=base['onset'],width=base['width'],duration=base['duration'])
        zero=simulate(peak=0.,**args)
        unit=simulate(peak=1.,**args)
        profiles.append((zero,{k:np.asarray(unit[k])-np.asarray(zero[k]) for k in zero if k!='t'}))
    times=np.asarray(profiles[0][0]['t'])
    (a0,a1),(b0,b1)=profiles
    h=base['h0']+np.asarray(a0['h'])[None,:]+np.asarray(b0['h'])[None,:]+p1[:,None]*a1['h']+p2[:,None]*b1['h']
    return dict(t=times,h=h,p1=p1,p2=p2,corr=corr,profiles=profiles)


def selected_data(result,index,base):
    data={'t':result['t']}
    derived=[]
    for i,(zero,unit) in enumerate(result['profiles']):
        peak=result['p1' if i==0 else 'p2'][index]
        derived.append({k:np.asarray(zero[k])+peak*unit[k] for k in unit})
    a,b=derived
    data.update(s1=a['s'],s2=b['s'],qin1=a['qin'],qin2=b['qin'],rain1=a['rain'],rain2=b['rain'])
    for key in ['v','qout','input','output']:data[key]=a[key]+b[key]
    data['h']=result['h'][index]
    return data


def at(data,t):
    return {k:float(np.interp(t,data['t'],v)) for k,v in data.items()}


def probability(h,times,threshold,horizon):
    i=min(len(times)-1,max(0,int(np.searchsorted(times,horizon,side='right')-1)))
    if i<len(times)-1:
        alpha=(horizon-times[i])/(times[i+1]-times[i])
        current=h[:,i]*(1-alpha)+h[:,i+1]*alpha
    else:current=h[:,i]
    maxima=np.maximum(h[:,:i+1].max(axis=1),current)
    return float(np.mean(maxima>threshold)),maxima
