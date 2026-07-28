# Cognitive Fabric 1Xq, Revision 1.1

This package turns the original Cognitive Fabric prototype into a mathematically
specified architecture and a checked reference realization.

## Contents

- `output/pdf/Cognitive_Fabric_1Xq.pdf` — the research manuscript.
- `Cognitive_Fabric_1Xq.tex` — complete LaTeX source.
- `formal/CognitiveFabricPublication.tla` — executable publication transition
  system and safety predicates.
- `formal/CognitiveFabricPublicationProof.tla` — TLAPS proof of the temporal
  safety theorem.
- `formal/CognitiveFabricPublication.cfg` — finite TLC cross-check.
- `formal/VERIFICATION_RESULTS.txt` — tool versions, commands, and recorded
  results.
- `cognitive_fabric_reference.py` — standard-library Python realization of
  formal model `CF-1XQ-1.1`.
- `test_cognitive_fabric_reference.py` — executable conformance suite.
- `CONFORMANCE_MANIFEST.txt` — release identity, verification record, and
  SHA-256 digests.

The original uploaded prototype is preserved separately.

## Formal core

The manuscript defines selectors, selector votes, bounded coalitions,
operational posteriors, posterior alignment, structural proposals, immutable
snapshots, decisions, outcomes, and publication transitions. It proves:

- coordinate-transform composition reversibility;
- spectral floor, ceiling, and condition-number bounds for selector precision;
- local and coalition simplex invariants and vote-mass conservation;
- exact optimality of bounded coalition selection under the stated budget;
- the tempered log-pool identity, stabilized prior neutrality at the
  probability floor, and exact neutrality when the floor is inactive;
- deterministic replay for explicit logical time and seed;
- generation monotonicity, direct foreign-evidence exclusion, and update
  safety;
- per-resolution snapshot consistency and a restricted-history implementation
  linearization-point argument;
- in-process rollback for queued outcomes after synchronous successor
  construction failure; and
- structural-proposal non-interference.

The publication module defines typing, bounded capacity, queue-ID uniqueness,
disjoint and exhaustive occurrence accounting, exact count correspondence,
snapshot/generation agreement, publisher phase facts, resolution metadata
consistency, exact per-commit classification, and foreign-application
exclusion. TLAPS proves:

```text
ParameterAssumptions |- SafetySpec => []Safety
```

The clean run discharged all 747 obligations and completed Isabelle
reconstruction with no errors. TLC independently explored 920,549 generated
states, 402,582 distinct states, and depth 38 for two resolvers, capacity two,
and four occurrence IDs, with no error. A deliberately false invariant was
rejected at a reachable `PublishCommit`.

Revision 1.1 leaves the `CF-PUB-1.0` specification, proof, and configuration
byte-for-byte unchanged, so that formal verification record is carried forward
without representing it as a new TLAPS execution.

Algebraic theorems are over exact reals. The Python code uses explicit
finite-precision tolerances, immutable values, a canonical SHA-256 snapshot
identity, and stated concurrency assumptions. The conformance suite is
executable evidence. It is not the TLAPS proof and does not establish a
statement-level refinement from Python to TLA+.

## Revision 1.1 corrections

- Prior and local reports now pass through the same normalized stabilization
  operator. The corrected theorem distinguishes stabilized neutrality from
  exact interior neutrality.
- Every spline axis derives the active domain
  `[knots[degree], knots[basis_count]]`; partition of unity is claimed only on
  that domain, and degenerate domains are rejected.
- Precision construction normalizes before forming the PSD product and uses a
  logarithmic cap comparison, preventing overflow for extreme finite factors.
- Outcome success is explicitly Boolean in both the mathematical state and the
  executable constructor.
- Coalition optimality is explicitly combinatorial exactness over realized
  objective values; exact-real ordering is not attributed to IEEE-754 ties.

## Claim boundary

- The TLA+ theorem covers the publication abstraction, not the selector,
  coalition, posterior, or structural-proposal algebra.
- `ResolutionSnapshotConsistency` proves metadata coherence; the Python
  no-mixed-fields argument additionally relies on immutable captured objects.
- Foreign outcomes are never directly applied, but bounded-capacity occupancy
  can still displace local evidence.
- The model consumes one local observation per resolver completion. The Python
  API permits repeated observations, so unrestricted trace refinement is not
  claimed.
- Each model instance has a positive finite `MaxOutcomeId`; the symbolic proof
  covers every admissible bound, not an unbounded implementation trace without
  a lifting argument.
- The linearization account is restricted to the listed in-process calls under
  the two assumed locks. Crash consistency, durable exactly-once delivery,
  scheduler fairness, and a generic atomic-object linearizability proof are
  outside scope.

## Reproduce

Requires Python 3.11 or newer. The implementation has no third-party runtime
dependencies.

```sh
python3 -m unittest -v test_cognitive_fabric_reference.py
python3 cognitive_fabric_reference.py
```

With TLAPS rolling 1.6.0-pre build `3ab43c7`:

```sh
tlapm --strict --cleanfp -C formal/CognitiveFabricPublicationProof.tla
```

With TLA Tools `2026.03.02.213938`, run TLC from `formal/`:

```sh
java -cp "$TLA2TOOLS" tlc2.TLC -workers 1 \
  -config CognitiveFabricPublication.cfg CognitiveFabricPublication.tla
```

To rebuild the paper with a TeX installation containing TikZ:

```sh
mkdir -p output/pdf
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=output/pdf Cognitive_Fabric_1Xq.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=output/pdf Cognitive_Fabric_1Xq.tex
```

## Research status

The package is suitable as an arXiv architecture-and-proofs preprint draft.
The manuscript deliberately avoids claiming that the operational posterior is
calibrated Bayesian inference, that the information-gain heuristic is mutual
information, that learning is reversible, or that advisory structural
proposals self-modify the registry.

For an archival formal-methods submission, the next strongest evidence would be
a statement-level Python-to-TLA+ refinement against an atomic sequential
reference history, a mechanized algebraic core, and experiments covering
calibration, regret, abstention, and selector growth.
