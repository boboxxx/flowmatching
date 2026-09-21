"""Small, model-free checks of the manuscript's accounting identities."""
import importlib.util
import math
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('spring_story',Path(__file__).resolve().parents[1]/'scripts/build_spring_story.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SpringAccountingTest(unittest.TestCase):
    def test_scale_cost_once(self):
        cost=lambda a:4096*(1+a)+8192+2*(1+a)
        self.assertEqual(cost(0),12290)
        self.assertEqual(cost(1),16388)
        self.assertEqual(cost(1)-cost(0),4098)

    def test_expected_cost_difference(self):
        pa,pf=.7397,.726506
        for eta in (.25,.5,1,2,4):
            c=lambda p:4096*(1+p)+math.ceil(4096/eta)+math.ceil(1/eta)*(1+p)
            self.assertAlmostEqual(c(pa)-c(pf),(4096+math.ceil(1/eta))*(pa-pf),places=9)
            self.assertLess((c(pa)-c(pf))/c(pa),(pa-pf)/(1+pa))

    def test_decision_partition(self):
        events={(0,0):.25875,(1,0):.01474,(0,1):.001545,(1,1):.724965}
        self.assertAlmostEqual(sum(events.values()),1)
        pa=sum(a*p for (a,_),p in events.items())
        pf=sum(f*p for (_,f),p in events.items())
        self.assertAlmostEqual(pa-pf,events[1,0]-events[0,1])

    def test_interval_uses_training_run_count(self):
        r=module.estimate([1,2,3])
        self.assertEqual(r['n'],3)
        self.assertEqual(r['mean'],2)
        self.assertAlmostEqual(r['ci_high']-2,4.3026527299/math.sqrt(3))


if __name__=='__main__':unittest.main()
