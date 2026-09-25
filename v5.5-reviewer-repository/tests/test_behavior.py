import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproduce import behavior as b


class BehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence, cls.manifest = b.load_evidence(b.ROOT / 'data/behavior')
        cls.result = b.analyze(cls.evidence)

    def test_published_main_tables(self):
        expected = {
            ('qwen', 'original'):(95.67,0.00,93.73,0.00,0.00,0.00,0.00),
            ('qwen', 'answer_prefill'):(98.13,98.16,95.73,95.73,2.42,1.67,3.18),
            ('mistral', 'original'):(93.73,0.00,83.02,0.00,0.00,0.00,0.00),
            ('mistral', 'answer_prefill'):(95.09,95.18,87.11,87.20,7.98,6.73,9.27)
        }
        for key,want in expected.items():
            r=self.result['primary'][key[0]][key[1]]
            a,c=(r['methods'][o+'/learned'] for o in ('f_star','m3'))
            d=r['intended_minus_alternative_global']
            got=(a['candidate_mean'],a['global_mean'],c['candidate_mean'],c['global_mean'],d['difference'],*d['ci95'])
            self.assertEqual(tuple(round(100*x,2) for x in got),want)
        table3={('qwen','f_star'):((98.33,98.16,-.18,-.42,.07),(100,100,0,0,0)),('qwen','m3'):((.40,95.73,95.33,94.49,96.11),(37.50,97.22,59.72,58.19,61.04)),('mistral','f_star'):((94.24,95.18,.93,.27,1.64),(100,99.65,-.35,-.87,0)),('mistral','m3'):((1.31,87.20,85.89,84.42,87.27),(37.50,90.45,52.95,50.28,55.59))}
        for (model,o),want in table3.items():
            actual=[self.result['primary'][model]['answer_prefill']['learned_minus_pca'][o],self.result['consequences'][model]['checking'][o]['learned_minus_comparators_all_views']['pca']]
            for r,w in zip(actual,want):
                self.assertEqual(tuple(round(100*x,2) for x in (r['right_mean'],r['left_mean'],r['difference'],*r['ci95'])),w)

    def test_published_checking_and_recoding(self):
        expected={('qwen','f_star'):([960,960,960],[120,120,120],[120,120,120]),('qwen','m3'):([894,949,957],[106,117,119],[106,117,120]),('mistral','f_star'):([960,960,950],[120,120,117],[120,120,117]),('mistral','m3'):([846,847,912],[92,94,105],[97,97,109])}
        for (m,o),(answers,joint,report) in expected.items():
            r=self.result['consequences'][m]['checking'][o]
            self.assertEqual([x['correct_answers'] for x in r['methods']['learned']['per_fit']],answers)
            self.assertEqual([x['joint_correct'] for x in r['methods']['learned']['per_fit']],joint)
            self.assertEqual([x['view_correct'][-1] for x in r['methods']['learned']['per_fit']],report)
            if o=='m3':
                for x in r['methods']['recoded_pca']['per_fit']:
                    self.assertEqual((x['view_correct'][0],x['joint_correct']),(120,0))
                self.assertEqual(r['common_opportunity_cases'],96)
        self.assertEqual([x['all_other_consistent'] for x in self.result['consequences']['qwen']['checking']['m3']['methods']['learned']['consistency']],[119,119,119])
        self.assertEqual([x['all_other_consistent'] for x in self.result['consequences']['mistral']['checking']['m3']['methods']['learned']['consistency']],[110,112,110])
        self.assertEqual(self.result['consequences']['qwen']['discovery']['m3']['account_discrimination']['belief_edit']['per_fit'][0]['net_event_hits'],82)
        for m in ('qwen','mistral'):
            for split in ('discovery','checking'):
                for cells in self.result['consequences'][m][split]['natural_controls'].values():
                    self.assertTrue(all(x['global_correct']==120 for x in cells.values()))

    def test_opportunities_use_only_clean_predictions(self):
        x=copy.deepcopy(self.evidence['main'])
        reference=b.opportunity_results(x)
        for v in x['models'].values():
            for rows in v['primary'].values():
                for row in rows:
                    for s in row[2:]:s[1]=None
        altered=b.opportunity_results(x)
        for key,pop in reference['panels'].items():
            for population,row in pop.items():
                self.assertEqual(row['selected_pair_uids'],altered['panels'][key][population]['selected_pair_uids'])
        self.assertEqual(reference['panels']['answer_prefill/global']['shared']['n_pairs'],1069)
        self.assertEqual(reference['panels']['original/global']['shared']['n_pairs'],0)

    def test_supporting_rows_retain_failures(self):
        s=self.result['supporting']
        for objective,target in [('f_star',(.972,.954)),('m0_legacy',(.813,.774)),('m1',(.914,.841)),('m3',(.951,.892))]:
            actual=tuple(round(np.mean([s['breadth'][f'{objective}_ts{seed}'][split]['own_target_accuracy'] for seed in b.SEEDS]),3) for split in ('iid','shifted'))
            self.assertEqual(actual,target)
        for model,expected in [('qwen7_das',(196,193)),('mistral7_das',(200,200))]:
            self.assertEqual(tuple(s['rank'][model]['cells'][f'{r}/all/all']['source_transfer_correct'] for r in (16,64)),expected)
        self.assertFalse(s['belief']['qwen_three_agent']['historical_natural_gate_pass'])
        self.assertFalse(s['belief']['mistral_two_agent_selective']['historical_natural_gate_pass'])
        two=s['belief']['mistral_two_agent_selective']
        self.assertEqual(two['cells']['all/belief_counterfactual/other_agent']['global_correct'],1)
        self.assertEqual(two['joint']['all/belief_counterfactual']['correct'],1)
        self.assertEqual(s['route']['pilot']['rows'],1080)
        self.assertEqual(s['route']['development']['rows'],252)
        self.assertFalse(s['route']['pilot']['historical_natural_gate_pass'])
        self.assertFalse(s['route']['development']['historical_natural_gate_pass'])
        med=self.result['final_answer_mediation']
        self.assertEqual(med['passed_candidates'],0)
        for key,n in [('restoration_half',20),('transplant_half',20),('restoration_beats_controls',16),('transplant_beats_controls',14)]:
            self.assertEqual(sum(not x['criterion_flags'][key] for x in med['candidates'].values()),n)
        self.assertAlmostEqual(max(x['restoration_fraction'] for x in med['candidates'].values()),.053514886668977205)

    def test_invalid_outputs_and_degenerate_intervals(self):
        self.assertIsNone(b.PERM.get(None))
        scores=[['box',None,1.0]]
        r=b.cell_result(scores,['box'])
        self.assertEqual((r['candidate_correct'],r['global_correct'],r['invalid_global']),(1,0,1))
        p=b.paired([[1,1,1],[1,1,1]],[[1,1,1],[1,1,1]])
        self.assertEqual(p['ci95'],[0.,0.])
        with self.assertRaises(ValueError):b.compare({'ci95':[0.,1.]},{'ci95':[1e-15,1.]})

    def test_integrity_and_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'SOURCE_PROVENANCE.json').write_text(json.dumps(self.manifest))
            (p/'main.json.gz').write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'Compressed evidence differs'):
                b.load_evidence(p)
        invalid=copy.deepcopy(self.evidence['main'])
        invalid['models']['qwen']['primary']['original'].pop()
        with self.assertRaisesRegex(ValueError,'Primary row count'):
            b.validate_main(invalid)

    def test_default_full_reference_and_write_once(self):
        expected=json.loads((b.ROOT/'expected/behavior.json').read_text())
        b.compare(self.result,expected)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            receipt=b.run(out)
            self.assertEqual(receipt['status'],'PASS')
            self.assertTrue(receipt['expected_reference_compared'])
            self.assertIn('PCA (%)',(out/'behavior_tables.md').read_text())
            with self.assertRaisesRegex(ValueError,'new directory'):b.run(out)


if __name__=='__main__':
    unittest.main()
