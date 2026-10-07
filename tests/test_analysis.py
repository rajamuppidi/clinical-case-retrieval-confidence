import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze import curve,COVERAGES,ece

class AnalysisChecks(unittest.TestCase):
    def test_known_selection_curve(self):
        score=np.array([.1,.9,.4,.7]);loss=np.array([1.,0.,1.,0.]);ids=np.array(['a','b','c','d'])
        expected=[]
        for c in COVERAGES:
            ordered=sorted(range(4),key=lambda i:(-score[i],ids[i]))
            expected.append(sum(loss[i] for i in ordered[:int(np.ceil(c*4))])/int(np.ceil(c*4)))
        np.testing.assert_allclose(curve(score,loss,ids),expected)
    def test_calibration_edges(self):
        self.assertEqual(ece(np.array([0,1]),np.array([0.,1.])),0.)
        self.assertAlmostEqual(ece(np.array([0,1]),np.array([1.,0.])),1.)

if __name__=='__main__':unittest.main()
