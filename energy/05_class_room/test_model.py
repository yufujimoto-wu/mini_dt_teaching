import unittest
import numpy as np
from model import Conditions, simulate, metrics

class ModelTest(unittest.TestCase):
    def test_energy_and_heat_balance(self):
        for people in [0,80]:
            for lead in [0,45,90]:
                c=Conditions(people=people)
                d=simulate(c,lead,26)
                self.assertTrue((d.power>=0).all())
                self.assertTrue((d.power<=c.capacity/c.cop+1e-9).all())
                self.assertAlmostEqual(d.energy.iloc[-1],d.power.iloc[:-1].sum()/60)
                np.testing.assert_allclose(np.diff(d.temperature),
                    (d.heat_gain-d.cooling).iloc[:-1]/c.capacitance/60,atol=1e-12)

    def test_precooling_effect(self):
        c=Conditions()
        baseline=metrics(simulate(c,0,26),c)
        early=metrics(simulate(c,45,26),c)
        self.assertLess(early['start'],baseline['start'])
        self.assertLess(early['hot'],baseline['hot'])
        self.assertGreater(early['energy'],baseline['energy'])

    def test_identical_conditions(self):
        c=Conditions()
        np.testing.assert_array_equal(simulate(c,0,26),simulate(c,0,26))

if __name__=='__main__':
    unittest.main()
