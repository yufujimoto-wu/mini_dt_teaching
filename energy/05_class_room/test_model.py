import unittest
from dataclasses import replace
import numpy as np
from model import *
class ModelTest(unittest.TestCase):
 def test_balance(self):
  for u in INSULATION.values():
   c=Conditions(envelope=u);d=simulate(c,120,24.5,.6)
   self.assertAlmostEqual(d.energy.iloc[-1],d.power.iloc[:-1].sum()/60)
   self.assertAlmostEqual(d.co2.iloc[-1],(d.power*d.ci).iloc[:-1].sum()/60)
   np.testing.assert_allclose(np.diff(d.temperature),(d.heat_gain-d.cooling).iloc[:-1]/c.capacitance/60,atol=1e-12)
   self.assertLessEqual(d.loc[d.minute>=0,'cooling'].max(),c.capacity*.6)
 def test_reference_latest_feasible_start(self):
  for u in INSULATION.values():
   c=Conditions(envelope=u);d,l=baseline(c)
   self.assertTrue(metrics(d,c)['feasible'])
   self.assertFalse(metrics(simulate(c,l-1,c.upper-.2,1),c)['feasible'])
 def test_carbon_reversal(self):
  for a,z,sign in [(.2,.6,-1),(.4,.4,1),(.6,.2,1)]:
   c=Conditions(ci_before=a,ci_during=z);b,_=baseline(c);p=simulate(c,120,24.5,.6)
   self.assertTrue(metrics(p,c)['feasible'])
   self.assertGreater(metrics(p,c)['energy'],metrics(b,c)['energy'])
   self.assertGreater(sign*(metrics(p,c)['co2']-metrics(b,c)['co2']),0)
 def test_infeasible_not_credited(self):
  c=Conditions(envelope=.9,people=80,outdoor_temp=38)
  self.assertFalse(metrics(simulate(c,0,24.5,.2),c)['feasible'])
if __name__=='__main__': unittest.main()
