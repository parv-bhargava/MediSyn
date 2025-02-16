import unittest
import os
import json
import sys
src_eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'eval'))
if src_eval_path not in sys.path:
    sys.path.insert(0, src_eval_path)
from evaluate import TreatmentEvaluator


class TestTreatmentEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create test data before all tests"""
        cls.test_dir = "test_data"
        os.makedirs(cls.test_dir, exist_ok=True)

        # Create sample reference file
        cls.ref_content = {
            "treatment_plan": "The patient should take aspirin daily. Monitor blood pressure."
        }
        with open(os.path.join(cls.test_dir, "case-study-1.json"), 'w') as f:
            json.dump(cls.ref_content, f)

        # Create sample output file
        cls.gen_content = {
            "generated_text": "The patient should take aspirin daily. Monitor blood pressure.",
            "source_file": "case-study-1.json"
        }
        with open(os.path.join(cls.test_dir, "treatment-plan-case-study-1.json"), 'w') as f:
            json.dump(cls.gen_content, f)

    def setUp(self):
        """Initialize evaluator for each test"""
        self.evaluator = TreatmentEvaluator()

    # Existing tests
    def test_load_data(self):
        """Test loading of generated and reference data. It validates correct loading data from JSON files."""
        gen, ref = self.evaluator.load_data(
            os.path.join(self.test_dir, "treatment-plan-case-study-1.json"),
            os.path.join(self.test_dir, "case-study-1.json")
        )
        self.assertEqual(gen, self.gen_content['generated_text'])
        self.assertEqual(ref, self.ref_content['treatment_plan'])

    def test_calculate_perfect_match(self):
        """Test BLEU score and ROUGE scores for identical texts"""
        # Test BLEU
        score = self.evaluator.calculate_bleu(
            "Perfect match text for testing.",
            "Perfect match text for testing."
        )
        self.assertAlmostEqual(score, 1.0, places=2)

        # Test ROUGE
        rouge = self.evaluator.calculate_rouge(
            "Perfect match text for testing.",
            "Perfect match text for testing."
        )
        self.assertAlmostEqual(rouge['rouge1'].fmeasure, 1.0, places=2)
        self.assertAlmostEqual(rouge['rouge2'].fmeasure, 1.0, places=2)
        self.assertAlmostEqual(rouge['rougeL'].fmeasure, 1.0, places=2)

    def test_zero_match(self):
        """Test BLEU score and ROUGE scores for completely different texts"""
        # Test BLEU
        bleu_score = self.evaluator.calculate_bleu(
            "Apple banana orange",
            "Cat dog mouse"
        )
        self.assertAlmostEqual(bleu_score, 0.0, places=2)

        # Test ROUGE
        rouge = self.evaluator.calculate_rouge(
            "Apple banana orange",
            "Cat dog mouse"
        )
        self.assertAlmostEqual(rouge['rouge1'].fmeasure, 0.0, places=2)
        self.assertAlmostEqual(rouge['rouge2'].fmeasure, 0., places=2)
        self.assertAlmostEqual(rouge['rougeL'].fmeasure, 0.0, places=2)

    def test_calculate_partial_match(self):
        """Test BLEU score and ROUGE scores for partially matching texts"""

        # Test BLEU
        score = self.evaluator.calculate_bleu(
            "The patient should take medication.",
            "The patient should take aspirin daily."
        )
        self.assertGreater(score, 0.2)
        self.assertLess(score, 0.8)

        # Test ROUGE
        rouge = self.evaluator.calculate_rouge(
            "The patient should take medication.",
            "The patient should take aspirin daily."
        )
        self.assertGreater(rouge['rouge1'].fmeasure, 0.2)
        self.assertLess(rouge['rouge1'].fmeasure, 0.8)

    def test_evaluate_directory(self):
        """Test evaluation of multiple files in directory. This tests the full evaluation workflow."""
        results = self.evaluator.evaluate(
            output_dir=self.test_dir,
            reference_dir=self.test_dir
        )
        self.assertEqual(len(results), 1) # Tests exactly 1 result is returned since only 1 file exists
        self.assertIn("case-study-1.json", results[0]["case_study"]) # Generated plan is for the matching case study
        self.assertAlmostEqual(results[0]["bleu"], 1.0, places=2)

    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        for fname in os.listdir(cls.test_dir):
            os.remove(os.path.join(cls.test_dir, fname))
        os.rmdir(cls.test_dir)


if __name__ == '__main__':
    unittest.main()