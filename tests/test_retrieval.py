"""Hand-computed checks protect ranking, metric and grouping correctness."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from retrieval import metrics,top_indices,partition,tokens,view_tokens

class RetrievalChecks(unittest.TestCase):
    def test_ties_at_cutoff(self):
        self.assertEqual(top_indices(np.array([2.,3.,2.,2.,1.]),3).tolist(),[1,0,2])
    def test_known_metrics(self):
        m=metrics(['x','a','b','y','z','t','u','v','w','r'],{'a':1,'b':1,'missing':1})
        self.assertAlmostEqual(m['mrr1000'],.5)
        self.assertAlmostEqual(m['recall1000'],2/3)
        self.assertAlmostEqual(m['precision10'],.2)
        self.assertAlmostEqual(m['ndcg10'],(1/np.log2(3)+.5)/(1+1/np.log2(3)+.5))
    def test_group_independence(self):
        self.assertEqual(partition('12345-1'),partition('12345-99'))
    def test_negation_retained(self):
        self.assertIn('no',tokens('No fever or cough.'))
    def test_head_tail_query_view(self):
        terms=[str(i) for i in range(200)]
        self.assertEqual(view_tokens(terms,'headtail128'),terms[:64]+terms[-64:])
        self.assertEqual(view_tokens(terms[:100],'headtail128'),terms[:100])
    def test_high_idf_view_keeps_rare_terms_in_original_order(self):
        terms=['common']*130+['rare1','rare2']
        vocab={'common':0,'rare1':1,'rare2':2}
        selected=view_tokens(terms,'highidf128',np.array([1.,5.,4.]),vocab)
        self.assertEqual(len(selected),128)
        self.assertEqual(selected[-2:],['rare1','rare2'])

if __name__=='__main__':unittest.main()
