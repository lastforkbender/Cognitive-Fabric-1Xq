from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations
from math import isclose
from threading import Barrier, Event, Thread
from unittest import TestCase, main, mock
import random

import cognitive_fabric_reference as cf


EVALUATION_NS = 123_456_789_000
COMMIT_NS = 123_456_999_000


def candidate_model(
    candidate_id: str,
    *,
    dimension: int = 1,
    bias: float = 0.0,
) -> cf.SelectorCandidateModel:
    return cf.SelectorCandidateModel(
        candidate_id=candidate_id,
        spline=cf.zero_spline(dimension, bias=bias),
    )


def minimal_snapshot(
    *,
    priors: tuple[tuple[str, float], ...] = (("a", 0.2), ("b", 0.8)),
) -> cf.CognitiveFabricSnapshot:
    candidates = tuple(
        cf.CandidateModel(candidate_id, prior)
        for candidate_id, prior in priors
    )
    selector = cf.SelectorRegion(
        selector_id="s",
        transform_chain=("identity",),
        center=(0.0,),
        precision=cf.default_precision(1),
        candidate_models=tuple(
            candidate_model(candidate.candidate_id)
            for candidate in candidates
        ),
        reliability=0.7,
    )
    return cf.CognitiveFabricSnapshot(
        generation=1,
        transforms=(
            cf.OrthogonalTransform(
                "identity",
                cf.identity_matrix(1),
                cost=0.0,
            ),
        ),
        candidates=candidates,
        selectors=(selector,),
        edges=(),
        fallback_precision=cf.default_precision(1),
        created_ns=100,
    )


class TransformContractTests(TestCase):
    def test_rejects_false_orthogonal_claim(self) -> None:
        with self.assertRaises(ValueError):
            cf.OrthogonalTransform(
                "not_orthogonal",
                ((2.0, 0.0), (0.0, 1.0)),
            )

    def test_affine_condition_contract(self) -> None:
        with self.assertRaises(ValueError):
            cf.AffineTransform(
                "ill_conditioned",
                ((1e-11, 0.0), (0.0, 1.0)),
                (0.0, 0.0),
                max_condition=1e8,
            )

    def test_complete_chain_round_trip(self) -> None:
        fabric = cf.build_example_fabric()
        transforms = fabric.snapshot.transform_map()
        trace = cf.apply_transform_chain(
            (0.25, -0.75, 1.5),
            ("rotate_01", "offset_scale", "rotate_12"),
            transforms,
        )
        self.assertLess(trace.chain_reconstruction_error, 1e-12)
        self.assertEqual(
            trace.reconstructed_initial_vector,
            tuple(
                transforms["rotate_01"].inverse(
                    transforms["offset_scale"].inverse(
                        transforms["rotate_12"].inverse(trace.final_vector)
                    )
                )
            ),
        )

    def test_constructor_detaches_mutable_matrix(self) -> None:
        source = [[1.0, 0.0], [0.0, 1.0]]
        transform = cf.OrthogonalTransform("identity", source)
        source[0][0] = 9.0
        self.assertEqual(transform.forward((2.0, 3.0)), (2.0, 3.0))


class PrecisionContractTests(TestCase):
    def test_certified_spectral_bounds_and_condition(self) -> None:
        rng = random.Random(7)
        for dimension in range(1, 5):
            for _ in range(20):
                raw = tuple(
                    tuple(
                        rng.uniform(-12.0, 12.0) if col <= row else 0.0
                        for col in range(dimension)
                    )
                    for row in range(dimension)
                )
                precision = cf.PrecisionModel(
                    raw,
                    min_eigenvalue=0.01,
                    max_eigenvalue=5.0,
                    max_condition=50.0,
                )
                eigenvalues = cf.symmetric_eigenvalues(precision.matrix())
                self.assertGreaterEqual(
                    eigenvalues[0],
                    precision.spectral_floor - 1e-9,
                )
                self.assertLessEqual(eigenvalues[-1], 5.0 + 1e-9)
                self.assertLessEqual(
                    eigenvalues[-1] / eigenvalues[0],
                    50.0 + 1e-7,
                )

    def test_jacobi_diagnostic_sees_nonprincipal_eigenvector(self) -> None:
        eigenvalues = cf.symmetric_eigenvalues(
            ((2.0, -1.0), (-1.0, 2.0))
        )
        self.assertTrue(isclose(eigenvalues[0], 1.0, abs_tol=1e-12))
        self.assertTrue(isclose(eigenvalues[1], 3.0, abs_tol=1e-12))

    def test_broadening_never_sharpens(self) -> None:
        precision = cf.PrecisionModel(
            ((-10.0, 0.0), (0.3, 1.2)),
            min_eigenvalue=1e-4,
            max_eigenvalue=10.0,
            max_condition=1e4,
        )
        difference = cf.matrix_subtract(
            precision.matrix(),
            precision.broaden(0.75).matrix(),
        )
        self.assertGreaterEqual(
            cf.symmetric_eigenvalues(difference)[0],
            -1e-10,
        )

    def test_extreme_finite_factor_is_capped_without_overflow(self) -> None:
        precision = cf.PrecisionModel(
            ((1e308,),),
            min_eigenvalue=1e-4,
            max_eigenvalue=1e3,
            max_condition=1e6,
        )
        matrix = precision.matrix()
        self.assertTrue(isclose(matrix[0][0], 1e3, rel_tol=1e-12))


class SplineContractTests(TestCase):
    def test_partition_of_unity_on_closed_domain(self) -> None:
        spline = cf.zero_spline(1)
        axis = spline.axes[0]
        for index in range(121):
            value = -3.0 + 6.0 * index / 120.0
            basis = axis.evaluate(value)
            self.assertTrue(all(component >= -1e-12 for component in basis))
            self.assertTrue(isclose(sum(basis), 1.0, abs_tol=1e-10))
        self.assertEqual(axis.evaluate(3.0), (0.0,) * 6 + (1.0,))

    def test_partition_domain_is_derived_from_degree_and_basis_count(self) -> None:
        axis = cf.SplineAxis(knots=(0.0, 1.0, 2.0, 3.0), degree=1)
        self.assertEqual((axis.domain_min, axis.domain_max), (1.0, 2.0))
        for value in (1.0, 1.25, 1.5, 1.75, 2.0):
            self.assertTrue(isclose(sum(axis.evaluate(value)), 1.0, abs_tol=1e-12))
        self.assertTrue(isclose(sum(axis.evaluate(0.5)), 0.5, abs_tol=1e-12))

    def test_degenerate_active_domain_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cf.SplineAxis(knots=(0.0, 0.0, 0.0), degree=1)


class SnapshotContractTests(TestCase):
    def setUp(self) -> None:
        self.snapshot = cf.build_example_fabric().snapshot

    def test_registry_permutations_have_one_canonical_identity(self) -> None:
        selectors = tuple(
            replace(
                selector,
                candidate_models=tuple(reversed(selector.candidate_models)),
            )
            for selector in reversed(self.snapshot.selectors)
        )
        permuted = replace(
            self.snapshot,
            transforms=tuple(reversed(self.snapshot.transforms)),
            candidates=tuple(reversed(self.snapshot.candidates)),
            selectors=selectors,
            edges=tuple(reversed(self.snapshot.edges)),
        )
        self.assertEqual(permuted, self.snapshot)
        self.assertEqual(
            cf.snapshot_fingerprint(permuted),
            cf.snapshot_fingerprint(self.snapshot),
        )

    def test_duplicate_and_missing_references_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(
                self.snapshot,
                candidates=self.snapshot.candidates
                + (self.snapshot.candidates[0],),
            )

        incomplete = replace(
            self.snapshot.selectors[0],
            candidate_models=self.snapshot.selectors[0].candidate_models[:-1],
        )
        with self.assertRaises(ValueError):
            replace(
                self.snapshot,
                selectors=(incomplete,) + self.snapshot.selectors[1:],
            )

        unknown_transform = replace(
            self.snapshot.selectors[0],
            transform_chain=("missing",),
        )
        with self.assertRaises(ValueError):
            replace(
                self.snapshot,
                selectors=(unknown_transform,) + self.snapshot.selectors[1:],
            )

    def test_nonfinite_input_is_rejected(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                cf.evaluate_snapshot(
                    self.snapshot,
                    (value, 0.0, 0.0),
                    evaluation_ns=EVALUATION_NS,
                )


class EvaluationContractTests(TestCase):
    def setUp(self) -> None:
        self.fabric = cf.build_example_fabric()
        self.snapshot = self.fabric.snapshot

    def resolve(self, vector: cf.Vector = (0.25, 1.25, 0.1)) -> cf.DecisionDecomposition:
        return cf.evaluate_snapshot(
            self.snapshot,
            vector,
            exploration_seed=42,
            evaluation_ns=EVALUATION_NS,
        )

    def test_replay_is_functional_for_explicit_context(self) -> None:
        first = self.resolve()
        second = self.resolve()
        self.assertEqual(first, second)
        self.assertEqual(first.evaluated_ns, EVALUATION_NS)
        self.assertEqual(first.exploration_seed, 42)

    def test_decision_invariants_and_vote_mass(self) -> None:
        decision = self.resolve()
        cf.assert_decision_invariants(decision)
        self.assertTrue(
            isclose(
                sum(item.posterior for item in decision.candidate_posteriors),
                1.0,
                abs_tol=1e-12,
            )
        )
        for vote in decision.selector_votes:
            self.assertTrue(
                isclose(
                    sum(score.local_posterior for score in vote.candidate_scores),
                    1.0,
                    abs_tol=1e-12,
                )
            )
            for score in vote.candidate_scores:
                self.assertTrue(
                    isclose(
                        score.support
                        + score.opposition
                        + score.uncertainty,
                        1.0,
                        abs_tol=1e-12,
                    )
                )

    def test_coalition_is_exact_bounded_argmax(self) -> None:
        decision = self.resolve((1.2, -0.4, 0.8))
        votes = tuple(
            sorted(decision.selector_votes, key=lambda vote: vote.selector_id)
        )
        feasible = tuple(
            cf.score_coalition(selected, self.snapshot.edges)
            for size in range(1, min(4, len(votes)) + 1)
            for selected in combinations(votes, size)
        )
        expected = min(
            feasible,
            key=lambda trace: (-trace.score, trace.selector_ids),
        )
        self.assertEqual(decision.coalition, expected)

    def test_disabled_candidate_is_consistently_excluded(self) -> None:
        candidates = tuple(
            replace(candidate, enabled=False)
            if candidate.candidate_id == "direct_path"
            else candidate
            for candidate in self.snapshot.candidates
        )
        snapshot = replace(self.snapshot, candidates=candidates)
        decision = cf.evaluate_snapshot(
            snapshot,
            (0.0, 0.0, 0.0),
            exploration_seed=1,
            evaluation_ns=EVALUATION_NS,
        )
        ids = {
            item.candidate_id for item in decision.candidate_posteriors
        }
        self.assertNotIn("direct_path", ids)
        self.assertTrue(
            isclose(
                sum(item.posterior for item in decision.candidate_posteriors),
                1.0,
                abs_tol=1e-12,
            )
        )

    def test_prior_tempered_pool_is_neutral_when_local_reports_equal_prior(self) -> None:
        snapshot = minimal_snapshot()
        decision = cf.evaluate_snapshot(
            snapshot,
            (0.0,),
            exploration_seed=0,
            evaluation_ns=EVALUATION_NS,
        )
        posterior = {
            item.candidate_id: item.posterior
            for item in decision.candidate_posteriors
        }
        self.assertTrue(isclose(posterior["a"], 0.2, abs_tol=1e-12))
        self.assertTrue(isclose(posterior["b"], 0.8, abs_tol=1e-12))

    def test_pool_is_neutral_to_stabilized_prior_at_probability_floor(self) -> None:
        snapshot = minimal_snapshot(priors=(("a", 0.5), ("b", 2.5e-13)))
        decision = cf.evaluate_snapshot(
            snapshot,
            (0.0,),
            exploration_seed=0,
            evaluation_ns=EVALUATION_NS,
        )
        prior_total = sum(candidate.prior for candidate in snapshot.candidates)
        raw_prior = {
            candidate.candidate_id: candidate.prior / prior_total
            for candidate in snapshot.candidates
        }
        votes = tuple(
            replace(
                vote,
                candidate_scores=tuple(
                    replace(
                        score,
                        local_posterior=raw_prior[score.candidate_id],
                    )
                    for score in vote.candidate_scores
                ),
            )
            for vote in decision.selector_votes
        )
        pooled = cf.aggregate_coalition(
            decision.coalition,
            votes,
            snapshot.candidate_map(),
        )
        expected = cf.stabilize_distribution(raw_prior)
        for item in pooled:
            self.assertTrue(
                isclose(
                    item.posterior,
                    expected[item.candidate_id],
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            )

    def test_structural_proposals_are_advisory_and_noninterfering(self) -> None:
        before = cf.snapshot_fingerprint(self.snapshot)
        decision = self.resolve((100.0, 100.0, 100.0))
        after = cf.snapshot_fingerprint(self.snapshot)
        self.assertEqual(before, after)
        self.assertTrue(decision.structural_proposals)
        self.assertTrue(
            all(proposal.advisory_only for proposal in decision.structural_proposals)
        )


class LearningContractTests(TestCase):
    def test_non_boolean_success_is_rejected_before_queueing(self) -> None:
        fabric = cf.build_example_fabric()
        decision = fabric.resolve(
            (0.0, 0.0, 0.0),
            exploration_seed=0,
            evaluation_ns=EVALUATION_NS,
        )
        with self.assertRaises(TypeError):
            fabric.observe(
                decision,
                reward=1.0,
                succeeded=1,  # type: ignore[arg-type]
                latency_ns=10,
            )

    def test_two_outcomes_accumulate_two_spline_updates(self) -> None:
        fabric = cf.build_example_fabric()
        before = fabric.snapshot
        decision = fabric.resolve(
            (0.35, 0.45, -0.2),
            exploration_seed=8,
            evaluation_ns=EVALUATION_NS,
        )
        vote = decision.selector_votes[0]
        selected_score = next(
            score
            for score in vote.candidate_scores
            if score.candidate_id == decision.selected_candidate_id
        )
        old_selector = before.selector_map()[vote.selector_id]
        old_model = old_selector.candidate_map()[decision.selected_candidate_id]

        for _ in range(2):
            fabric.observe(
                decision,
                reward=1.0,
                succeeded=True,
                latency_ns=10,
            )
        after = fabric.publish_generation(commit_ns=COMMIT_NS)
        new_model = (
            after.selector_map()[vote.selector_id]
            .candidate_map()[decision.selected_candidate_id]
        )

        error = 1.0 - selected_score.spline.probability
        locality = vote.activation.learning_activation
        dimension_trace = selected_score.spline.dimensions[0]
        basis_index = next(
            index
            for index, basis in enumerate(dimension_trace.basis_values)
            if basis > 0.0
        )
        expected = (
            old_model.spline.coefficients[0][basis_index]
            + 2.0
            * 0.05
            * locality
            * error
            * dimension_trace.basis_values[basis_index]
        )
        self.assertTrue(
            isclose(
                new_model.spline.coefficients[0][basis_index],
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        self.assertEqual(
            new_model.statistics.observations
            - old_model.statistics.observations,
            2,
        )

    def test_precision_learning_is_not_a_noop(self) -> None:
        fabric = cf.build_example_fabric()
        before = fabric.snapshot
        decision = fabric.resolve(
            (0.55, -0.35, 0.75),
            exploration_seed=3,
            evaluation_ns=EVALUATION_NS,
        )
        fabric.observe(
            decision,
            reward=0.8,
            succeeded=True,
            latency_ns=50,
        )
        after = fabric.publish_generation(commit_ns=COMMIT_NS)
        changed = any(
            before.selector_map()[vote.selector_id].precision.matrix()
            != after.selector_map()[vote.selector_id].precision.matrix()
            for vote in decision.selector_votes
        )
        self.assertTrue(changed)

    def test_probation_is_reachable(self) -> None:
        health = cf.SelectorHealth()
        for _ in range(8):
            health = health.observed(
                0.0,
                0.0,
                0.0,
                dormant_after=2,
                probation_after=3,
            )
        self.assertEqual(health.lifecycle, cf.SelectorLifecycle.PROBATION)
        self.assertGreaterEqual(health.recovery_attempts, 3)

    def test_generation_transition_preserves_old_snapshot(self) -> None:
        fabric = cf.build_example_fabric()
        old = fabric.snapshot
        old_fingerprint = cf.snapshot_fingerprint(old)
        decision = fabric.resolve(
            (0.2, 0.3, 0.4),
            exploration_seed=2,
            evaluation_ns=EVALUATION_NS,
        )
        fabric.observe(
            decision,
            reward=0.7,
            succeeded=True,
            latency_ns=100,
        )
        new = fabric.publish_generation(commit_ns=COMMIT_NS)
        self.assertEqual(new.generation, old.generation + 1)
        self.assertEqual(cf.snapshot_fingerprint(old), old_fingerprint)
        self.assertNotEqual(cf.snapshot_fingerprint(new), old_fingerprint)


class PublicationContractTests(TestCase):
    def test_concurrent_publishers_apply_each_queued_outcome_once(self) -> None:
        fabric = cf.build_example_fabric()
        before = fabric.snapshot
        decisions = (
            fabric.resolve(
                (-0.8, -0.7, -0.5),
                exploration_seed=1,
                evaluation_ns=EVALUATION_NS,
            ),
            fabric.resolve(
                (0.9, 0.2, 0.7),
                exploration_seed=2,
                evaluation_ns=EVALUATION_NS,
            ),
        )
        for decision in decisions:
            fabric.observe(
                decision,
                reward=0.9,
                succeeded=True,
                latency_ns=10,
            )

        barrier = Barrier(3)
        results: list[cf.CognitiveFabricSnapshot] = []

        def publisher() -> None:
            barrier.wait()
            results.append(
                fabric.publish_generation(commit_ns=COMMIT_NS)
            )

        threads = (Thread(target=publisher), Thread(target=publisher))
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()

        after = fabric.snapshot
        self.assertEqual(after.generation, before.generation + 1)
        self.assertTrue(all(result is after for result in results))
        expected = Counter(
            (vote.selector_id, decision.selected_candidate_id)
            for decision in decisions
            for vote in decision.selector_votes
        )
        for (selector_id, candidate_id), count in expected.items():
            old_stats = (
                before.selector_map()[selector_id]
                .candidate_map()[candidate_id]
                .statistics
            )
            new_stats = (
                after.selector_map()[selector_id]
                .candidate_map()[candidate_id]
                .statistics
            )
            self.assertEqual(
                new_stats.observations - old_stats.observations,
                count,
            )
        self.assertEqual(fabric.rejected_outcomes, 0)

    def test_publish_exception_retains_evidence(self) -> None:
        fabric = cf.build_example_fabric()
        decision = fabric.resolve(
            (0.1, 0.2, 0.3),
            exploration_seed=9,
            evaluation_ns=EVALUATION_NS,
        )
        fabric.observe(
            decision,
            reward=0.75,
            succeeded=True,
            latency_ns=12,
        )
        with mock.patch.object(
            cf,
            "apply_outcomes",
            side_effect=RuntimeError("injected"),
        ):
            with self.assertRaises(RuntimeError):
                fabric.publish_generation(commit_ns=COMMIT_NS)

        successor = fabric.publish_generation(commit_ns=COMMIT_NS)
        self.assertEqual(successor.generation, decision.generation + 1)

    def test_stale_and_wrong_fingerprint_outcomes_are_accounted(self) -> None:
        fabric = cf.build_example_fabric()
        old_decision = fabric.resolve(
            (0.1, 0.2, 0.3),
            exploration_seed=4,
            evaluation_ns=EVALUATION_NS,
        )
        fabric.observe(
            old_decision,
            reward=0.8,
            succeeded=True,
            latency_ns=3,
        )
        current = fabric.publish_generation(commit_ns=COMMIT_NS)

        fabric.observe(
            old_decision,
            reward=0.8,
            succeeded=True,
            latency_ns=3,
        )
        unchanged = fabric.publish_generation(commit_ns=COMMIT_NS + 1)
        self.assertIs(unchanged, current)
        self.assertEqual(fabric.rejected_outcomes, 1)

        foreign_snapshot = replace(
            current,
            created_ns=current.created_ns + 1,
        )
        foreign = cf.evaluate_snapshot(
            foreign_snapshot,
            (0.1, 0.2, 0.3),
            exploration_seed=4,
            evaluation_ns=EVALUATION_NS,
        )
        fabric.observe(
            foreign,
            reward=0.8,
            succeeded=True,
            latency_ns=3,
        )
        self.assertIs(
            fabric.publish_generation(commit_ns=COMMIT_NS + 2),
            current,
        )
        self.assertEqual(fabric.rejected_outcomes, 2)

    def test_bounded_queue_reports_drops(self) -> None:
        base = cf.build_example_fabric().snapshot
        fabric = cf.CognitiveFabric(base, outcome_capacity=1)
        first = fabric.resolve(
            (0.0, 0.0, 0.0),
            exploration_seed=1,
            evaluation_ns=EVALUATION_NS,
        )
        second = fabric.resolve(
            (1.0, 1.0, 1.0),
            exploration_seed=2,
            evaluation_ns=EVALUATION_NS,
        )
        for decision in (first, second):
            fabric.observe(
                decision,
                reward=0.5,
                succeeded=True,
                latency_ns=1,
            )
        self.assertEqual(fabric.dropped_outcomes, 1)

    def test_inflight_resolve_remains_bound_to_captured_snapshot(self) -> None:
        fabric = cf.build_example_fabric()
        old_snapshot = fabric.snapshot
        training_decision = fabric.resolve(
            (0.0, 0.0, 0.0),
            exploration_seed=1,
            evaluation_ns=EVALUATION_NS,
        )
        fabric.observe(
            training_decision,
            reward=0.8,
            succeeded=True,
            latency_ns=1,
        )

        captured = Event()
        continue_evaluation = Event()
        original = cf.evaluate_snapshot
        results: list[cf.DecisionDecomposition] = []

        def blocked(snapshot, input_vector, **kwargs):
            captured.set()
            continue_evaluation.wait(timeout=5)
            return original(snapshot, input_vector, **kwargs)

        with mock.patch.object(cf, "evaluate_snapshot", side_effect=blocked):
            thread = Thread(
                target=lambda: results.append(
                    fabric.resolve(
                        (0.4, 0.5, 0.6),
                        exploration_seed=7,
                        evaluation_ns=EVALUATION_NS,
                    )
                )
            )
            thread.start()
            self.assertTrue(captured.wait(timeout=5))
            new_snapshot = fabric.publish_generation(commit_ns=COMMIT_NS)
            continue_evaluation.set()
            thread.join(timeout=5)

        self.assertEqual(len(results), 1)
        decision = results[0]
        self.assertEqual(decision.generation, old_snapshot.generation)
        self.assertEqual(
            decision.snapshot_fingerprint,
            cf.snapshot_fingerprint(old_snapshot),
        )
        self.assertNotEqual(
            decision.snapshot_fingerprint,
            cf.snapshot_fingerprint(new_snapshot),
        )
        cf.assert_decision_invariants(decision)


if __name__ == "__main__":
    main(verbosity=2)
