"""How much does transmit power weigh in the energy objective? (report/05_han_che.md; audit of 2026-11-13)

Propulsion (Zeng et al., no wind) and the radio draw at a few speeds and power levels, with the energy objective
-(P - P_min) / (P_max - P_min) of _init_energy_bounds. The radio term spans the gap between power 0.05 and 1.

Run from the repo root: py report/scripts/energy_terms.py
"""
import io,contextlib,numpy as np,sys
sys.path.insert(0,'.')
with contextlib.redirect_stdout(io.StringIO()):
    from environment import NetworkSlicingEnv
    env=NetworkSlicingEnv('config/environment/ma3.yaml')
P=lambda v,vz=0,air=None: env._propulsion_power(v,vz,air)
vs=np.arange(0,35,0.5)
p=[P(v,0,v) for v in vs]
print('no-wind power: hover',P(0,0,0),' 17 m/s',P(17,0,17),' argmin', vs[np.argmin(p)], min(p))
print('bounds min/max total', env.min_total_power, env.max_total_power)
rad=lambda f: env._radio_power(f*6.3)
for name,v,f in [('hover p=1',0,1),('hover p=.05',0,.05),('6 m/s p=1',6,1),('11.6 m/s p=.4',11.6,.4),('17 m/s p=.4',17,.4),('17 m/s p=.05',17,.05)]:
    tot=P(v,0,v)+rad(f)
    print(name, round(tot,1), 'energy obj', round(-(tot-env.min_total_power)/(env.max_total_power-env.min_total_power),4))
print('radio range W', rad(0.05), rad(1))
