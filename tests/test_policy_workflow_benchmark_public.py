import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "policy_workflow_benchmark_public.py"
)
SPEC = importlib.util.spec_from_file_location("policy_workflow_benchmark", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class PolicyWorkflowBenchmarkTests(unittest.TestCase):
    def test_repo_benchmark_distribution_and_direct_count(self):
        cases = benchmark.generate_repo_benchmark()
        self.assertEqual(len(cases), 500)
        counts = Counter(case["drift_family"] for case in cases)
        self.assertEqual(set(counts.keys()), set(benchmark.DRIFT_FAMILIES))
        self.assertTrue(all(value == 100 for value in counts.values()))
        self.assertGreaterEqual(
            sum(1 for case in cases if case["source_type"] == "repo_direct"),
            34,
        )

    def test_external_transfer_slice_distribution(self):
        cases = benchmark.generate_external_transfer_slice()
        self.assertEqual(len(cases), 30)
        counts = Counter(case["drift_family"] for case in cases)
        self.assertTrue(all(value == 6 for value in counts.values()))
        self.assertTrue(
            all(case["source_type"] == "external_benchmark_inspired" for case in cases)
        )

    def test_observable_view_hides_oracle(self):
        case = benchmark.generate_repo_benchmark()[0]
        view = benchmark.case_observable_view(case)
        self.assertNotIn("hidden_oracle", view)
        serialized = str(view)
        self.assertNotIn("gold_preferred_action", serialized)
        self.assertNotIn("gold_legal_actions", serialized)

    def test_projector_blocks_illegal_side_effect_when_audit_missing(self):
        case = next(
            candidate
            for candidate in benchmark.generate_repo_benchmark()
            if candidate["observable_context"]["drift_signals"].get("audit_missing")
        )
        action = benchmark.project_action(
            case,
            ["send_sms_marketing", "handoff_human_scripted", "abstain"],
            {"audit_guard", "script_guard"},
        )
        self.assertIn(action, {"handoff_human_scripted", "abstain"})
        self.assertNotIn(action, benchmark.SIDE_EFFECT_ACTIONS)

    def test_pgca_has_lower_repo_violation_than_static_and_react(self):
        repo_cases = benchmark.generate_repo_benchmark()
        external_cases = benchmark.generate_external_transfer_slice()
        summary = benchmark.evaluate_benchmarks(repo_cases, external_cases)
        repo = summary["repo_native"]
        self.assertLess(
            repo["overall"]["pgca"]["policy_violation_rate"],
            repo["overall"]["static_prompt"]["policy_violation_rate"],
        )
        self.assertLess(
            repo["overall"]["pgca"]["policy_violation_rate"],
            repo["overall"]["react_tool"]["policy_violation_rate"],
        )
        self.assertGreater(
            repo["guard_selector"]["micro_f1"],
            0.65,
        )
        self.assertGreater(
            repo["pairwise_tests"]["pgca_vs_static_prompt"]["mcnemar_policy_violation"]["discordant"],
            0,
        )

    def test_summary_exposes_ablation_and_guard_diagnostics(self):
        repo_cases = benchmark.generate_repo_benchmark()
        external_cases = benchmark.generate_external_transfer_slice()
        summary = benchmark.evaluate_benchmarks(repo_cases, external_cases)
        repo = summary["repo_native"]

        self.assertIn("guard_diagnostics", repo)
        self.assertIn("ablations", repo)
        self.assertIn("pgca_full_guard", repo["ablations"]["overall"])
        self.assertIn("pgca_raw_selector", repo["ablations"]["overall"])
        self.assertIn("pgca_no_projector", repo["ablations"]["overall"])
        self.assertIn("pgca_minus_join_guard", repo["ablations"]["overall"])
        self.assertIn("bootstrap_delta", repo)
        self.assertIn("policy_violation_rate_ci95", repo["overall"]["pgca"])
        self.assertIn("leakage_diagnostics", repo["guard_diagnostics"])
        self.assertIn("stress_tests", repo)
        self.assertIn("dataset_profile", repo)
        self.assertIn("label_provenance", repo)

    def test_dataset_profile_reports_independent_blueprints_and_perturbation_diversity(self):
        repo_cases = benchmark.generate_repo_benchmark()
        external_cases = benchmark.generate_external_transfer_slice()
        summary = benchmark.evaluate_benchmarks(repo_cases, external_cases)
        profile = summary["repo_native"]["dataset_profile"]

        self.assertEqual(profile["blueprint_count"], 20)
        self.assertEqual(profile["avg_variants_per_blueprint"], 25.0)
        self.assertGreaterEqual(profile["avg_structured_field_diff_from_anchor"], 4.0)
        self.assertEqual(profile["mean_distinct_signal_signatures_per_blueprint"], 25.0)

    def test_label_provenance_is_oracle_derived_and_calibrator_is_rule_based(self):
        repo_cases = benchmark.generate_repo_benchmark()
        external_cases = benchmark.generate_external_transfer_slice()
        summary = benchmark.evaluate_benchmarks(repo_cases, external_cases)
        provenance = summary["repo_native"]["label_provenance"]

        self.assertEqual(provenance["guard_label_source"], "oracle_derived_required_guards")
        self.assertEqual(provenance["selector_type"], "trained_tree_based_multi_label_classifier")
        self.assertEqual(provenance["calibrator_type"], "rule_based_safety_prior")

    def test_group_cv_generalization_gap_is_reported(self):
        repo_cases = benchmark.generate_repo_benchmark()
        diagnostics = benchmark.compute_guard_diagnostics(repo_cases)
        self.assertIn("group_cv", diagnostics)
        self.assertIn("random_cv", diagnostics)
        self.assertIn("leakage_diagnostics", diagnostics)
        self.assertLessEqual(
            diagnostics["group_cv"]["validation_micro_f1"],
            diagnostics["random_cv"]["validation_micro_f1"],
        )
        self.assertGreater(
            diagnostics["group_cv"]["validation_micro_f1"],
            0.55,
        )
        self.assertGreater(
            diagnostics["leakage_diagnostics"]["random_cv_blueprint_leakage_rate"],
            0.9,
        )
        self.assertEqual(
            diagnostics["leakage_diagnostics"]["group_cv_blueprint_leakage_rate"],
            0.0,
        )

    def test_calibrator_improves_over_raw_selector(self):
        repo_cases = benchmark.generate_repo_benchmark()
        external_cases = benchmark.generate_external_transfer_slice()
        summary = benchmark.evaluate_benchmarks(repo_cases, external_cases)
        ablations = summary["repo_native"]["ablations"]["overall"]
        self.assertGreater(
            summary["repo_native"]["overall"]["pgca"]["compliant_action_accuracy"],
            ablations["pgca_raw_selector"]["compliant_action_accuracy"],
        )
        self.assertLessEqual(
            summary["repo_native"]["overall"]["pgca"]["policy_violation_rate"],
            ablations["pgca_raw_selector"]["policy_violation_rate"],
        )

    def test_proposal_noise_stress_keeps_pgca_safer_than_dynamic(self):
        repo_cases = benchmark.generate_repo_benchmark()
        external_cases = benchmark.generate_external_transfer_slice()
        summary = benchmark.evaluate_benchmarks(repo_cases, external_cases)
        stress = summary["repo_native"]["stress_tests"]["proposal_noise_medium"]["overall"]
        self.assertLess(
            stress["pgca"]["policy_violation_rate"],
            stress["dynamic_prompt"]["policy_violation_rate"],
        )
        self.assertLess(
            stress["pgca"]["policy_violation_rate"],
            stress["react_tool"]["policy_violation_rate"],
        )


if __name__ == "__main__":
    unittest.main()
