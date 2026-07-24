"""
cognitive_fabric_reference_mp_sqtp1XQ.py

COGNITIVE FABRIC PHASE 1.3 OPERATING PROFILE

Resolve Action
- Resolve actions: 144
- Sustained throughput: 360.36 actions/s
- Mean latency: 2.775 ms
- P95 latency: 2.991 ms
- P99 latency: 4.179 ms
- Deterministic replay agreement: 16/16

Cognitive Appeal
- Appeal actions: 24
- Appeal throughput: 410.39 appeals/s
- Mean appeal latency: 2.437 ms
- P95 appeal latency: 2.560 ms
- P99 appeal latency: 3.178 ms
- Winner revisions: 0
- Revision rate: 0.00%

Evidence and Publication
- Occurrences created: 48
- Queued: 0
- Applied: 32
- Dropped: 16
- Rejected: 0
- Queue saturation: 16
- Publication attempts: 2
- Successful generations: 1
- Empty publications: 1
- Accounting invariant: PASS

Structural Transactions
- Transactions attempted: 7
- Committed: 2
- Transactions rejected: 5
- Commit rate: 28.57%
- Transaction throughput: 247.73 tx/s
- Mean transaction latency: 4.037 ms
- P95 transaction latency: 16.135 ms
- P99 transaction latency: 20.938 ms
- Stale-parent rejections: 1
- Precondition rejections: 1
- Parameter-schema rejections: 1
- Successor-validation rejections: 1
- Owner-only promotion violations: 1

Realized Cognitive Gain
- Estimated-gain samples: 2
- Realized-gain samples: 1
- Mean estimated gain: 0.025000
- Mean realized gain: 0.000100
- Minimum realized gain: 0.000100
- Maximum realized gain: 0.000100
- Mean prediction error: 0.029900

Rollback
- Rollbacks attempted: 2
- Rollbacks completed: 1
- Certificate rejections: 1
- Mean rollback latency: 0.517 ms
- P95 rollback latency: 0.976 ms
- Generation-monotonicity violations: 0
- Parent-retention failures: 0

History Discipline
- Completed invocations: 218
- Call events: 218
- Linearization events: 218
- Return events: 218
- Raise events: 0
- History-discipline violations: 0

Multiprocessing Isolation
- Worker jobs admitted: 2
- Worker jobs completed: 2
- Worker failures: 0
- Lease expirations: 0
- Cancellations: 0
- Stale results rejected: 1
- Valid candidates promoted: 1
- Non-owner promotion attempts: 1
- Worker-process isolation violations: 0

Verification
- Total tests: 19
- Passed: 19
- Failed: 0
- Stress duration: 1.280 s
- External storage/API calls: 0


STAQTAPP-1Xq v2.2.5 BETA / TELEMETRY FEEDBACK PROFILE

*Current version has a embedded ml-ai for integrity
 and recovery intravenously of segmentation options
 with state of the art recovery surfacing/polling

Release Identification
- Product: Staqtapp-1Xq
- Package version: 2.2.5
- Engine generation: 1Xq 2.2.5H
- Storage family: Staqtapp-1Xq
- API boundary: staqtapp.public_api
- Installed artifact: staqtapp-1xq==2.2.5
- Release identity verified: PASS

Storage Fire-Up
- Storage directory created: YES
- Configuration initialized: YES
- VFS created: YES
- VFS opened: YES
- Startup latency: 5.132 ms
- Shutdown completed: YES
- External storage/API calls: 21

Cognitive-Fabric Evidence Storage
- Events submitted: 6
- Decisions stored: 1
- Outcomes stored: 1
- Dispositions stored: 1
- Transactions stored: 1
- Rollbacks stored: 1
- Records rejected: 0
- Duplicate/idempotent submissions: 1
- Records read back: 11
- Payload agreement: PASS

Staqtapp-1Xq Telemetry
- Storage writes: 6
- Storage reads: 11
- Transactions attempted: 2
- Transactions committed: 2
- Transaction failures: 0
- Durability barriers: 1
- Bytes written: 2650
- Mean write latency: 9.289 ms
- P95 write latency: 14.785 ms
- P99 write latency: 15.273 ms
- Mean read latency: 1.404 ms
- P95 read latency: 5.361 ms
- P99 read latency: 6.411 ms
- Telemetry samples received: 17
- Telemetry samples dropped: 0

Integrity and Recovery
- Integrity checks: 1
- Integrity failures: 0
- Revisions observed: 3
- Recovery checks: 1
- Recovery failures: 0
- Durable restart agreement: PASS

Isolation
- Cognitive Fabric locks held during API calls: 0
- Cognitive Fabric result changes caused by storage: 0
- Storage failures propagated into resolution: 0
- Outbox records pending: 0
- Outbox retries: 0
- Isolation invariant: PASS

Verification
- Staqtapp integration tests: 10
- Passed: 10
- Failed: 0
- Test duration: 0.154 s
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from hashlib import sha256
from itertools import combinations, count
import argparse
import json
import multiprocessing as mp
import os
from math import cos, exp, fsum, isfinite, log, sin, sqrt
from threading import Lock, get_ident
from statistics import quantiles
from time import monotonic_ns, perf_counter_ns
from typing import Any, Callable, Deque, Iterable, Mapping, Protocol, Sequence

#import staqtapp1xq as qx1


Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]
CandidateId = str
SelectorId = str
TransformId = str

FORMAL_MODEL_ID = "CF-1XQ-1.0"
NUMERICAL_TOLERANCE = 1e-10


# ============================================================
# Scalar and linear-algebra utilities
# ============================================================

def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_probability(value: float, epsilon: float = 1e-12) -> float:
    if not isfinite(value):
        raise ValueError(f"Non-finite probability component: {value!r}")
    return clamp(value, epsilon, 1.0)


def sigmoid(value: float) -> float:
    value = clamp(value, -60.0, 60.0)
    return 1.0 / (1.0 + exp(-value))


def softplus(value: float) -> float:
    if value > 30.0:
        return value
    if value < -30.0:
        return exp(value)
    return log(1.0 + exp(value))


def inverse_softplus(value: float) -> float:
    if not isfinite(value) or value <= 0.0:
        raise ValueError("Inverse softplus requires a finite positive value.")
    if value > 30.0:
        return value
    return log(exp(value) - 1.0)


def require_finite(values: Iterable[float], *, label: str) -> None:
    if any(not isfinite(value) for value in values):
        raise ValueError(f"{label} must contain only finite values.")


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vector dimensions do not match.")
    return fsum(a * b for a, b in zip(left, right))


def squared_norm(vector: Sequence[float]) -> float:
    return dot(vector, vector)


def squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vector dimensions do not match.")
    return sum((a - b) ** 2 for a, b in zip(left, right))


def matvec(matrix: Matrix, vector: Sequence[float]) -> Vector:
    if not matrix:
        raise ValueError("Matrix cannot be empty.")
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("Matrix/vector dimensions do not match.")
    return tuple(dot(row, vector) for row in matrix)


def matmul(left: Matrix, right: Matrix) -> Matrix:
    if not left or not right:
        raise ValueError("Matrices cannot be empty.")
    right_t = transpose(right)
    if len(left[0]) != len(right):
        raise ValueError("Matrix dimensions do not align.")
    return tuple(tuple(dot(row, col) for col in right_t) for row in left)


def transpose(matrix: Matrix) -> Matrix:
    if not matrix:
        raise ValueError("Matrix cannot be empty.")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("Matrix rows must have equal length.")
    return tuple(tuple(matrix[r][c] for r in range(len(matrix))) for c in range(width))


def identity_matrix(dimension: int) -> Matrix:
    if dimension <= 0:
        raise ValueError("Dimension must be positive.")
    return tuple(
        tuple(1.0 if row == col else 0.0 for col in range(dimension))
        for row in range(dimension)
    )


def matrix_add(left: Matrix, right: Matrix) -> Matrix:
    if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right)):
        raise ValueError("Matrix dimensions do not match.")
    return tuple(tuple(a + b for a, b in zip(lrow, rrow)) for lrow, rrow in zip(left, right))


def matrix_subtract(left: Matrix, right: Matrix) -> Matrix:
    if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right)):
        raise ValueError("Matrix dimensions do not match.")
    return tuple(tuple(a - b for a, b in zip(lrow, rrow)) for lrow, rrow in zip(left, right))


def matrix_scale(matrix: Matrix, scalar: float) -> Matrix:
    return tuple(tuple(value * scalar for value in row) for row in matrix)


def matrix_infinity_norm(matrix: Matrix) -> float:
    if not matrix:
        raise ValueError("Matrix cannot be empty.")
    return max(sum(abs(value) for value in row) for row in matrix)


def outer(vector: Sequence[float]) -> Matrix:
    return tuple(tuple(a * b for b in vector) for a in vector)


def quadratic_form(vector: Sequence[float], matrix: Matrix) -> float:
    return dot(vector, matvec(matrix, vector))


def determinant(matrix: Matrix) -> float:
    """Gaussian-elimination determinant for small dense matrices."""
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Determinant requires a non-empty square matrix.")

    work = [list(row) for row in matrix]
    det = 1.0

    for pivot in range(n):
        pivot_row = max(range(pivot, n), key=lambda r: abs(work[r][pivot]))
        pivot_value = work[pivot_row][pivot]

        if abs(pivot_value) < 1e-15:
            return 0.0

        if pivot_row != pivot:
            work[pivot], work[pivot_row] = work[pivot_row], work[pivot]
            det *= -1.0

        det *= work[pivot][pivot]
        inv_pivot = 1.0 / work[pivot][pivot]

        for row in range(pivot + 1, n):
            factor = work[row][pivot] * inv_pivot
            for col in range(pivot + 1, n):
                work[row][col] -= factor * work[pivot][col]

    return det


def inverse(matrix: Matrix) -> Matrix:
    """Gauss-Jordan inverse for small dense matrices."""
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Inverse requires a non-empty square matrix.")

    identity = identity_matrix(n)
    work = [list(matrix[row_index]) + list(identity[row_index]) for row_index in range(n)]

    for pivot in range(n):
        pivot_row = max(range(pivot, n), key=lambda r: abs(work[r][pivot]))
        if abs(work[pivot_row][pivot]) < 1e-15:
            raise ValueError("Matrix is singular.")

        if pivot_row != pivot:
            work[pivot], work[pivot_row] = work[pivot_row], work[pivot]

        divisor = work[pivot][pivot]
        work[pivot] = [value / divisor for value in work[pivot]]

        for row in range(n):
            if row == pivot:
                continue
            factor = work[row][pivot]
            work[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(work[row], work[pivot])
            ]

    return tuple(tuple(row[n:]) for row in work)


def power_iteration(matrix: Matrix, iterations: int = 40) -> float:
    n = len(matrix)
    vector = tuple(1.0 / sqrt(n) for _ in range(n))
    eigenvalue = 0.0

    for _ in range(iterations):
        transformed = matvec(matrix, vector)
        norm = sqrt(max(squared_norm(transformed), 1e-30))
        vector = tuple(value / norm for value in transformed)
        eigenvalue = dot(vector, matvec(matrix, vector))

    return eigenvalue


def symmetric_eigenvalues(
    matrix: Matrix,
    *,
    tolerance: float = 1e-12,
    maximum_rotations: int | None = None,
) -> Vector:
    """Jacobi eigenvalues for the small real-symmetric matrices used here."""
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Eigenvalue decomposition requires a square matrix.")
    require_finite((value for row in matrix for value in row), label="Matrix")
    work = [list(row) for row in matrix]
    for row in range(n):
        for col in range(row):
            if abs(work[row][col] - work[col][row]) > NUMERICAL_TOLERANCE:
                raise ValueError("Jacobi eigenvalues require a symmetric matrix.")
            average = 0.5 * (work[row][col] + work[col][row])
            work[row][col] = average
            work[col][row] = average

    rotations = maximum_rotations or max(16, 100 * n * n)
    for _ in range(rotations):
        pivot_row, pivot_col, largest = 0, 0, 0.0
        for row in range(n):
            for col in range(row + 1, n):
                magnitude = abs(work[row][col])
                if magnitude > largest:
                    pivot_row, pivot_col, largest = row, col, magnitude
        if largest <= tolerance:
            break

        p, q = pivot_row, pivot_col
        app, aqq, apq = work[p][p], work[q][q], work[p][q]
        tau = (aqq - app) / (2.0 * apq)
        tangent = (
            1.0 / (tau + sqrt(1.0 + tau * tau))
            if tau >= 0.0
            else -1.0 / (-tau + sqrt(1.0 + tau * tau))
        )
        cosine = 1.0 / sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine

        for index in range(n):
            if index in (p, q):
                continue
            aip = work[index][p]
            aiq = work[index][q]
            work[index][p] = cosine * aip - sine * aiq
            work[p][index] = work[index][p]
            work[index][q] = sine * aip + cosine * aiq
            work[q][index] = work[index][q]

        work[p][p] = app - tangent * apq
        work[q][q] = aqq + tangent * apq
        work[p][q] = 0.0
        work[q][p] = 0.0

    return tuple(sorted(work[index][index] for index in range(n)))


def approximate_eigen_bounds(matrix: Matrix) -> tuple[float, float]:
    eigenvalues = symmetric_eigenvalues(matrix)
    return eigenvalues[0], eigenvalues[-1]


def condition_number(matrix: Matrix) -> float:
    minimum, maximum = approximate_eigen_bounds(matrix)
    if minimum <= 0.0:
        return float("inf")
    return maximum / minimum


def symmetric_spectral_upper_bound(matrix: Matrix) -> float:
    """
    Return a rigorous induced-infinity-norm upper bound on the spectral radius.

    For a real symmetric matrix, every eigenvalue has magnitude no greater
    than the maximum absolute row sum.  Unlike power iteration, this bound is
    suitable for enforcing a hard precision cap.
    """
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Spectral bound requires a non-empty square matrix.")
    return max(sum(abs(value) for value in row) for row in matrix)


def cholesky(matrix: Matrix, *, jitter: float = 1e-14) -> Matrix:
    """Lower Cholesky factor for a symmetric positive-definite matrix."""
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Cholesky factorization requires a square matrix.")
    require_finite((value for row in matrix for value in row), label="Matrix")

    result = [[0.0 for _ in range(n)] for _ in range(n)]
    for row in range(n):
        for col in range(row + 1):
            if abs(matrix[row][col] - matrix[col][row]) > NUMERICAL_TOLERANCE:
                raise ValueError("Cholesky matrix must be symmetric.")
            subtotal = sum(result[row][k] * result[col][k] for k in range(col))
            if row == col:
                diagonal = matrix[row][row] - subtotal
                if diagonal <= 0.0:
                    if diagonal < -jitter:
                        raise ValueError("Matrix is not positive definite.")
                    diagonal = jitter
                result[row][col] = sqrt(diagonal)
            else:
                if result[col][col] <= 0.0:
                    raise ValueError("Matrix is not positive definite.")
                result[row][col] = (matrix[row][col] - subtotal) / result[col][col]
    return tuple(tuple(row) for row in result)


def softmax(logits: Mapping[str, float]) -> dict[str, float]:
    if not logits:
        return {}
    maximum = max(logits.values())
    exponentials = {key: exp(value - maximum) for key, value in logits.items()}
    denominator = fsum(exponentials.values())
    if denominator <= 0.0:
        raise RuntimeError("Invalid softmax denominator.")
    return {key: value / denominator for key, value in exponentials.items()}


def normalized_entropy(probabilities: Iterable[float]) -> float:
    values = [safe_probability(value) for value in probabilities if value > 0.0]
    if len(values) <= 1:
        return 0.0
    entropy = -fsum(value * log(value) for value in values)
    return entropy / log(len(values))


def seeded_jitter(seed: int, label: str) -> float:
    payload = f"{seed}:{label}".encode("utf-8")
    integer = int.from_bytes(sha256(payload).digest()[:8], "big")
    return integer / float((1 << 64) - 1)


# ============================================================
# Reversible transformations
# ============================================================

class ReversibleTransform(Protocol):
    transform_id: TransformId

    def forward(self, vector: Vector) -> Vector: ...
    def inverse(self, vector: Vector) -> Vector: ...
    def complexity_cost(self) -> float: ...


@dataclass(frozen=True, slots=True)
class OrthogonalTransform:
    transform_id: TransformId
    matrix: Matrix
    cost: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matrix",
            tuple(tuple(float(value) for value in row) for row in self.matrix),
        )
        size = len(self.matrix)
        if not self.transform_id:
            raise ValueError("Transform id cannot be empty.")
        if size == 0 or any(len(row) != size for row in self.matrix):
            raise ValueError("Orthogonal transform matrix must be square.")
        require_finite(
            (value for row in self.matrix for value in row),
            label="Orthogonal transform matrix",
        )
        if not isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("Transform complexity cost must be finite and non-negative.")
        gram = matmul(transpose(self.matrix), self.matrix)
        identity = identity_matrix(size)
        maximum_error = max(
            abs(value - expected)
            for row, expected_row in zip(gram, identity)
            for value, expected in zip(row, expected_row)
        )
        if maximum_error > NUMERICAL_TOLERANCE:
            raise ValueError(
                "Orthogonal transform requires M^T M = I "
                f"within {NUMERICAL_TOLERANCE:g}; error={maximum_error:.3e}."
            )

    def forward(self, vector: Vector) -> Vector:
        return matvec(self.matrix, vector)

    def inverse(self, vector: Vector) -> Vector:
        return matvec(transpose(self.matrix), vector)

    def complexity_cost(self) -> float:
        return self.cost


@dataclass(frozen=True, slots=True)
class AffineTransform:
    transform_id: TransformId
    matrix: Matrix
    offset: Vector
    cost: float = 1.5
    max_condition: float = 1e10

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matrix",
            tuple(tuple(float(value) for value in row) for row in self.matrix),
        )
        object.__setattr__(self, "offset", tuple(float(value) for value in self.offset))
        size = len(self.matrix)
        if not self.transform_id:
            raise ValueError("Transform id cannot be empty.")
        if size == 0 or any(len(row) != size for row in self.matrix):
            raise ValueError("Affine matrix must be square.")
        if len(self.offset) != size:
            raise ValueError("Affine offset dimension mismatch.")
        require_finite(
            (value for row in self.matrix for value in row),
            label="Affine transform matrix",
        )
        require_finite(self.offset, label="Affine transform offset")
        if not isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("Transform complexity cost must be finite and non-negative.")
        if not isfinite(self.max_condition) or self.max_condition < 1.0:
            raise ValueError("Affine maximum condition must be finite and at least one.")
        if abs(determinant(self.matrix)) < 1e-12:
            raise ValueError("Affine transform must be invertible.")
        condition_bound = (
            matrix_infinity_norm(self.matrix)
            * matrix_infinity_norm(inverse(self.matrix))
        )
        if condition_bound > self.max_condition:
            raise ValueError(
                "Affine transform is too ill-conditioned for the numerical contract."
            )

    def forward(self, vector: Vector) -> Vector:
        projected = matvec(self.matrix, vector)
        return tuple(value + offset for value, offset in zip(projected, self.offset))

    def inverse(self, vector: Vector) -> Vector:
        shifted = tuple(value - offset for value, offset in zip(vector, self.offset))
        return matvec(inverse(self.matrix), shifted)

    def complexity_cost(self) -> float:
        return self.cost


@dataclass(frozen=True, slots=True)
class TransformStepTrace:
    transform_id: TransformId
    input_vector: Vector
    output_vector: Vector
    reconstructed_vector: Vector
    reconstruction_error: float
    complexity_cost: float


@dataclass(frozen=True, slots=True)
class TransformChainTrace:
    transform_ids: tuple[TransformId, ...]
    initial_vector: Vector
    final_vector: Vector
    steps: tuple[TransformStepTrace, ...]
    total_complexity_cost: float
    reconstructed_initial_vector: Vector
    chain_reconstruction_error: float


def apply_transform_chain(
    vector: Vector,
    transform_ids: Sequence[TransformId],
    transforms: Mapping[TransformId, ReversibleTransform],
) -> TransformChainTrace:
    require_finite(vector, label="Transform input")
    ordered_transform_ids = tuple(transform_ids)
    current = vector
    traces: list[TransformStepTrace] = []
    total_cost = 0.0

    for transform_id in ordered_transform_ids:
        transform = transforms[transform_id]
        output = transform.forward(current)
        reconstructed = transform.inverse(output)
        require_finite(output, label=f"Transform {transform_id!r} output")
        require_finite(
            reconstructed,
            label=f"Transform {transform_id!r} reconstruction",
        )
        error = sqrt(squared_distance(current, reconstructed))
        cost = transform.complexity_cost()
        traces.append(
            TransformStepTrace(
                transform_id=transform_id,
                input_vector=current,
                output_vector=output,
                reconstructed_vector=reconstructed,
                reconstruction_error=error,
                complexity_cost=cost,
            )
        )
        current = output
        total_cost += cost

    reconstructed_chain = current
    for transform_id in reversed(ordered_transform_ids):
        reconstructed_chain = transforms[transform_id].inverse(reconstructed_chain)
    chain_error = sqrt(squared_distance(vector, reconstructed_chain))

    return TransformChainTrace(
        transform_ids=ordered_transform_ids,
        initial_vector=vector,
        final_vector=current,
        steps=tuple(traces),
        total_complexity_cost=total_cost,
        reconstructed_initial_vector=reconstructed_chain,
        chain_reconstruction_error=chain_error,
    )


# ============================================================
# Precision matrices and selector health
# ============================================================

class SelectorLifecycle(str, Enum):
    HEALTHY = "healthy"
    DORMANT = "dormant"
    RECOVERY = "recovery"
    PROBATION = "probation"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class PrecisionDiagnostics:
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    determinant: float
    log_determinant: float
    healthy: bool


@dataclass(frozen=True, slots=True)
class PrecisionModel:
    """
    Precision is constructed as L L^T + floor * I.

    The lower-triangular factor is stored directly. Its diagonal is passed
    through softplus, guaranteeing strictly positive diagonal entries.
    """

    raw_lower: Matrix
    min_eigenvalue: float = 1e-4
    max_eigenvalue: float = 1e3
    max_condition: float = 1e6
    factor_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_lower",
            tuple(tuple(float(value) for value in row) for row in self.raw_lower),
        )
        n = len(self.raw_lower)
        if n == 0 or any(len(row) != n for row in self.raw_lower):
            raise ValueError("Precision factor must be square.")
        require_finite(
            (value for row in self.raw_lower for value in row),
            label="Precision factor",
        )
        require_finite(
            (
                self.min_eigenvalue,
                self.max_eigenvalue,
                self.max_condition,
                self.factor_scale,
            ),
            label="Precision bounds",
        )
        if self.min_eigenvalue <= 0.0:
            raise ValueError("Minimum eigenvalue must be positive.")
        if self.max_eigenvalue <= self.min_eigenvalue:
            raise ValueError("Maximum eigenvalue must exceed the minimum eigenvalue.")
        if self.max_condition < 1.0:
            raise ValueError("Maximum condition number must be at least one.")
        if self.factor_scale <= 0.0:
            raise ValueError("Precision factor scale must be positive.")

    @property
    def dimension(self) -> int:
        return len(self.raw_lower)

    @property
    def spectral_floor(self) -> float:
        return max(
            self.min_eigenvalue,
            self.max_eigenvalue / self.max_condition,
        )

    def factor(self) -> Matrix:
        scale = sqrt(self.factor_scale)
        rows: list[Vector] = []
        for row_index, row in enumerate(self.raw_lower):
            built = []
            for col_index, value in enumerate(row):
                if col_index > row_index:
                    built.append(0.0)
                elif col_index == row_index:
                    built.append(scale * (softplus(value) + 1e-6))
                else:
                    built.append(scale * value)
            rows.append(tuple(built))
        return tuple(rows)

    def matrix(self) -> Matrix:
        factor = self.factor()
        positive_semidefinite = matmul(factor, transpose(factor))
        spectral_floor = self.spectral_floor
        available = self.max_eigenvalue - spectral_floor
        upper_bound = symmetric_spectral_upper_bound(positive_semidefinite)
        if upper_bound > available:
            positive_semidefinite = matrix_scale(
                positive_semidefinite,
                available / upper_bound,
            )
        floor = matrix_scale(identity_matrix(self.dimension), spectral_floor)
        return matrix_add(positive_semidefinite, floor)

    @classmethod
    def from_base_matrix(
        cls,
        base_matrix: Matrix,
        *,
        template: "PrecisionModel",
    ) -> "PrecisionModel":
        """
        Encode a positive-definite matrix B as the factor term L L^T.

        The configured eigenvalue floor is added later by ``matrix()``.
        """
        factor = cholesky(base_matrix)
        raw_rows: list[Vector] = []
        for row_index, row in enumerate(factor):
            built: list[float] = []
            for col_index, value in enumerate(row):
                if col_index > row_index:
                    built.append(0.0)
                elif col_index == row_index:
                    built.append(inverse_softplus(max(value - 1e-6, 1e-14)))
                else:
                    built.append(value)
            raw_rows.append(tuple(built))
        return cls(
            raw_lower=tuple(raw_rows),
            min_eigenvalue=template.min_eigenvalue,
            max_eigenvalue=template.max_eigenvalue,
            max_condition=template.max_condition,
            factor_scale=1.0,
        )

    def diagnostics(self) -> PrecisionDiagnostics:
        precision = self.matrix()
        minimum, maximum = approximate_eigen_bounds(precision)
        factor = cholesky(precision)
        log_det = 2.0 * fsum(
            log(factor[index][index])
            for index in range(self.dimension)
        )
        det = exp(log_det) if log_det < 709.0 else float("inf")
        cond = float("inf") if minimum <= 0.0 else maximum / minimum

        healthy = (
            isfinite(minimum)
            and isfinite(maximum)
            and isfinite(cond)
            and isfinite(log_det)
            and minimum >= self.min_eigenvalue * 0.95
            and maximum <= self.max_eigenvalue * 1.05
            and cond <= self.max_condition
        )

        return PrecisionDiagnostics(
            minimum_eigenvalue=minimum,
            maximum_eigenvalue=maximum,
            condition_number=cond,
            determinant=det,
            log_determinant=log_det,
            healthy=healthy,
        )

    def broaden(self, factor: float = 0.95) -> "PrecisionModel":
        if not 0.0 < factor <= 1.0:
            raise ValueError("Broadening factor must be in (0, 1].")
        return replace(self, factor_scale=self.factor_scale * factor)

    def shrink_toward(
        self,
        fallback: "PrecisionModel",
        amount: float,
    ) -> "PrecisionModel":
        if self.dimension != fallback.dimension:
            raise ValueError("Fallback precision dimension mismatch.")
        amount = clamp(amount, 0.0, 1.0)
        mixed = tuple(
            tuple((1.0 - amount) * a + amount * b for a, b in zip(row_a, row_b))
            for row_a, row_b in zip(self.raw_lower, fallback.raw_lower)
        )
        factor_scale = (
            (1.0 - amount) * self.factor_scale
            + amount * fallback.factor_scale
        )
        return replace(
            self,
            raw_lower=mixed,
            factor_scale=factor_scale,
            min_eigenvalue=(
                (1.0 - amount) * self.min_eigenvalue
                + amount * fallback.min_eigenvalue
            ),
            max_eigenvalue=(
                (1.0 - amount) * self.max_eigenvalue
                + amount * fallback.max_eigenvalue
            ),
            max_condition=(
                (1.0 - amount) * self.max_condition
                + amount * fallback.max_condition
            ),
        )


@dataclass(frozen=True, slots=True)
class SelectorHealth:
    activation_mass: float = 0.0
    learning_mass: float = 0.0
    update_mass: float = 0.0
    inactive_steps: int = 0
    stalled_steps: int = 0
    observations: int = 0
    lifecycle: SelectorLifecycle = SelectorLifecycle.HEALTHY
    recovery_attempts: int = 0

    def observed(
        self,
        routing_activation: float,
        learning_activation: float,
        update_norm: float,
        *,
        decay: float = 0.98,
        activation_threshold: float = 1e-3,
        update_threshold: float = 1e-8,
        dormant_after: int = 64,
        probation_after: int = 4,
    ) -> "SelectorHealth":
        activation_mass = decay * self.activation_mass + (1.0 - decay) * routing_activation
        learning_mass = decay * self.learning_mass + (1.0 - decay) * learning_activation
        update_mass = decay * self.update_mass + (1.0 - decay) * update_norm

        inactive_steps = self.inactive_steps + 1 if routing_activation < activation_threshold else 0
        stalled_steps = self.stalled_steps + 1 if update_norm < update_threshold else 0

        lifecycle = self.lifecycle
        recovery_attempts = self.recovery_attempts

        if lifecycle != SelectorLifecycle.RETIRED:
            if inactive_steps >= dormant_after and lifecycle == SelectorLifecycle.HEALTHY:
                lifecycle = SelectorLifecycle.DORMANT
            elif lifecycle == SelectorLifecycle.DORMANT:
                lifecycle = SelectorLifecycle.RECOVERY
                recovery_attempts += 1
            elif lifecycle == SelectorLifecycle.RECOVERY:
                if routing_activation >= activation_threshold:
                    lifecycle = SelectorLifecycle.HEALTHY
                    recovery_attempts = 0
                else:
                    recovery_attempts += 1
                    if recovery_attempts >= probation_after:
                        lifecycle = SelectorLifecycle.PROBATION

        return SelectorHealth(
            activation_mass=activation_mass,
            learning_mass=learning_mass,
            update_mass=update_mass,
            inactive_steps=inactive_steps,
            stalled_steps=stalled_steps,
            observations=self.observations + 1,
            lifecycle=lifecycle,
            recovery_attempts=recovery_attempts,
        )


@dataclass(frozen=True, slots=True)
class ActivationDecomposition:
    selector_id: SelectorId
    delta: Vector
    mahalanobis_squared: float
    log_activation: float
    routing_activation: float
    learning_activation: float
    exploration_probability: float
    precision: Matrix
    precision_diagnostics: PrecisionDiagnostics
    health_before: SelectorHealth


# ============================================================
# B-spline scoring
# ============================================================

def bspline_basis(index: int, degree: int, value: float, knots: Sequence[float]) -> float:
    if degree < 0:
        raise ValueError("Degree cannot be negative.")
    if index + degree + 1 >= len(knots):
        raise IndexError("Insufficient knots.")

    if degree == 0:
        left = knots[index]
        right = knots[index + 1]
        final_interval = value == knots[-1] and index + 1 == len(knots) - 1
        return 1.0 if (left <= value < right) or final_interval else 0.0

    left_den = knots[index + degree] - knots[index]
    right_den = knots[index + degree + 1] - knots[index + 1]

    left_term = 0.0
    right_term = 0.0

    if left_den != 0.0:
        left_term = (
            (value - knots[index]) / left_den
            * bspline_basis(index, degree - 1, value, knots)
        )
    if right_den != 0.0:
        right_term = (
            (knots[index + degree + 1] - value) / right_den
            * bspline_basis(index + 1, degree - 1, value, knots)
        )

    return left_term + right_term


@dataclass(frozen=True, slots=True)
class SplineAxis:
    knots: tuple[float, ...]
    degree: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "knots", tuple(float(value) for value in self.knots))
        if self.degree < 0:
            raise ValueError("Spline degree cannot be negative.")
        if len(self.knots) < self.degree + 2:
            raise ValueError("Spline axis has insufficient knots.")
        require_finite(self.knots, label="Spline knots")
        if any(left > right for left, right in zip(self.knots, self.knots[1:])):
            raise ValueError("Spline knots must be non-decreasing.")
        if self.basis_count <= 0:
            raise ValueError("Spline axis requires at least one basis function.")

    @property
    def basis_count(self) -> int:
        return len(self.knots) - self.degree - 1

    def evaluate(self, value: float) -> Vector:
        if not isfinite(value):
            raise ValueError("Spline input must be finite.")
        if value == self.knots[-1]:
            return tuple(
                1.0 if index == self.basis_count - 1 else 0.0
                for index in range(self.basis_count)
            )
        return tuple(
            bspline_basis(index, self.degree, value, self.knots)
            for index in range(self.basis_count)
        )


@dataclass(frozen=True, slots=True)
class SplineDimensionTrace:
    dimension: int
    value: float
    basis_values: Vector
    coefficients: Vector
    contributions: Vector
    subtotal: float


@dataclass(frozen=True, slots=True)
class SplineScoreTrace:
    bias: float
    dimensions: tuple[SplineDimensionTrace, ...]
    raw_score: float
    probability: float


@dataclass(frozen=True, slots=True)
class SplineModel:
    axes: tuple[SplineAxis, ...]
    coefficients: tuple[Vector, ...]
    bias: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(
            self,
            "coefficients",
            tuple(
                tuple(float(value) for value in coefficient_row)
                for coefficient_row in self.coefficients
            ),
        )
        if len(self.axes) != len(self.coefficients):
            raise ValueError("Every spline axis requires coefficients.")
        for axis, coefficients in zip(self.axes, self.coefficients):
            if axis.basis_count != len(coefficients):
                raise ValueError("Spline basis/coefficient mismatch.")
            require_finite(coefficients, label="Spline coefficients")
        if not isfinite(self.bias):
            raise ValueError("Spline bias must be finite.")

    def evaluate(self, vector: Vector) -> SplineScoreTrace:
        if len(vector) != len(self.axes):
            raise ValueError("Spline input dimension mismatch.")

        raw = self.bias
        dimensions: list[SplineDimensionTrace] = []

        for dimension, (value, axis, coefficients) in enumerate(
            zip(vector, self.axes, self.coefficients)
        ):
            basis_values = axis.evaluate(value)
            contributions = tuple(
                basis * coefficient
                for basis, coefficient in zip(basis_values, coefficients)
            )
            subtotal = sum(contributions)
            raw += subtotal
            dimensions.append(
                SplineDimensionTrace(
                    dimension=dimension,
                    value=value,
                    basis_values=basis_values,
                    coefficients=coefficients,
                    contributions=contributions,
                    subtotal=subtotal,
                )
            )

        return SplineScoreTrace(
            bias=self.bias,
            dimensions=tuple(dimensions),
            raw_score=raw,
            probability=sigmoid(raw),
        )

    def learned(
        self,
        trace: SplineScoreTrace,
        error: float,
        learning_rate: float,
        locality: float,
    ) -> "SplineModel":
        updated: list[Vector] = []

        if len(trace.dimensions) != len(self.coefficients):
            raise ValueError("Spline learning trace dimension mismatch.")

        for current_coefficients, dimension in zip(
            self.coefficients,
            trace.dimensions,
        ):
            updated.append(
                tuple(
                    coefficient
                    + learning_rate * locality * error * basis
                    for coefficient, basis in zip(
                        current_coefficients,
                        dimension.basis_values,
                    )
                )
            )

        return replace(
            self,
            coefficients=tuple(updated),
            bias=self.bias + learning_rate * locality * error,
        )


# ============================================================
# Candidates, selectors, connectivity, and votes
# ============================================================

@dataclass(frozen=True, slots=True)
class CandidateStatistics:
    observations: int = 0
    successes: int = 0
    reward_mean: float = 0.5
    latency_mean_ns: float = 0.0
    last_observed_ns: int = 0

    def __post_init__(self) -> None:
        if self.observations < 0 or self.successes < 0:
            raise ValueError("Candidate counts cannot be negative.")
        if self.successes > self.observations:
            raise ValueError("Successes cannot exceed observations.")
        if not 0.0 <= self.reward_mean <= 1.0:
            raise ValueError("Reward mean must lie in [0, 1].")
        if not isfinite(self.latency_mean_ns) or self.latency_mean_ns < 0.0:
            raise ValueError("Latency mean must be finite and non-negative.")
        if self.last_observed_ns < 0:
            raise ValueError("Last-observed time cannot be negative.")

    @property
    def empirical_success(self) -> float:
        return (self.successes + 1.0) / (self.observations + 2.0)


@dataclass(frozen=True, slots=True)
class CandidateModel:
    candidate_id: CandidateId
    prior: float
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("Candidate id cannot be empty.")
        if not isfinite(self.prior) or not 0.0 < self.prior < 1.0:
            raise ValueError("Candidate prior propensity must be in (0, 1).")


@dataclass(frozen=True, slots=True)
class SelectorCandidateModel:
    candidate_id: CandidateId
    spline: SplineModel
    statistics: CandidateStatistics = CandidateStatistics()

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("Selector candidate id cannot be empty.")


@dataclass(frozen=True, slots=True)
class SelectorRegion:
    selector_id: SelectorId
    transform_chain: tuple[TransformId, ...]
    center: Vector
    precision: PrecisionModel
    candidate_models: tuple[SelectorCandidateModel, ...]
    reliability: float = 0.5
    novelty_tolerance: float = 0.5
    exploration_floor: float = 0.03
    routing_floor: float = 1e-12
    learning_floor: float = 1e-4
    health: SelectorHealth = SelectorHealth()
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform_chain", tuple(self.transform_chain))
        object.__setattr__(self, "center", tuple(float(value) for value in self.center))
        object.__setattr__(
            self,
            "candidate_models",
            tuple(
                sorted(
                    self.candidate_models,
                    key=lambda model: model.candidate_id,
                )
            ),
        )
        if not self.selector_id:
            raise ValueError("Selector id cannot be empty.")
        require_finite(self.center, label="Selector center")
        if len(self.center) != self.precision.dimension:
            raise ValueError("Selector center/precision dimension mismatch.")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("Selector reliability must lie in [0, 1].")
        if not 0.0 <= self.novelty_tolerance <= 1.0:
            raise ValueError("Selector novelty tolerance must lie in [0, 1].")
        if not 0.0 <= self.exploration_floor <= 1.0:
            raise ValueError("Exploration floor must lie in [0, 1].")
        if not 0.0 < self.routing_floor <= 1.0:
            raise ValueError("Routing floor must lie in (0, 1].")
        if not self.routing_floor <= self.learning_floor <= 1.0:
            raise ValueError("Learning floor must lie in [routing_floor, 1].")
        candidate_ids = [item.candidate_id for item in self.candidate_models]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Selector candidate ids must be unique.")
        for model in self.candidate_models:
            if len(model.spline.axes) != len(self.center):
                raise ValueError("Selector spline dimension mismatch.")

    def candidate_map(self) -> dict[CandidateId, SelectorCandidateModel]:
        return {item.candidate_id: item for item in self.candidate_models}


@dataclass(frozen=True, slots=True)
class SelectorEdge:
    source_id: SelectorId
    target_id: SelectorId
    compatibility: float = 0.5
    information_gain: float = 0.5
    redundancy: float = 0.0
    conflict_rate: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.source_id or not self.target_id:
            raise ValueError("Edge endpoints cannot be empty.")
        if self.source_id == self.target_id:
            raise ValueError("Self-edges are not part of the formal realization.")
        values = (
            self.compatibility,
            self.information_gain,
            self.redundancy,
            self.conflict_rate,
        )
        require_finite(values, label="Edge statistics")
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Edge statistics must lie in [0, 1].")

    def weight(self) -> float:
        if not self.enabled:
            return 0.0
        return sigmoid(
            1.2 * self.compatibility
            + 1.0 * self.information_gain
            - 1.1 * self.redundancy
            - 1.2 * self.conflict_rate
        )


@dataclass(frozen=True, slots=True)
class ProbabilityComponents:
    prior: float
    selector_activation: float
    spline_fit: float
    historical_success: float
    reward_quality: float
    freshness: float
    reliability: float
    uncertainty_penalty: float

    def as_mapping(self) -> Mapping[str, float]:
        return {
            "prior": self.prior,
            "selector_activation": self.selector_activation,
            "spline_fit": self.spline_fit,
            "historical_success": self.historical_success,
            "reward_quality": self.reward_quality,
            "freshness": self.freshness,
            "reliability": self.reliability,
            "uncertainty_penalty": self.uncertainty_penalty,
        }

    def log_score(self) -> float:
        return sum(log(safe_probability(value)) for value in self.as_mapping().values())

    def raw_product(self) -> float:
        product = 1.0
        for value in self.as_mapping().values():
            product *= safe_probability(value)
        return product


@dataclass(frozen=True, slots=True)
class CandidateScoreDecomposition:
    selector_id: SelectorId
    candidate_id: CandidateId
    spline: SplineScoreTrace
    components: ProbabilityComponents
    raw_product: float
    log_score: float
    local_posterior: float
    support: float
    opposition: float
    uncertainty: float


@dataclass(frozen=True, slots=True)
class SelectorVote:
    selector_id: SelectorId
    activation: ActivationDecomposition
    transform_trace: TransformChainTrace
    candidate_scores: tuple[CandidateScoreDecomposition, ...]
    novelty: float
    expected_information_gain: float
    evaluation_reason: str


@dataclass(frozen=True, slots=True)
class ConnectivityTrace:
    source_id: SelectorId
    target_id: SelectorId
    compatibility: float
    information_gain: float
    redundancy: float
    conflict_rate: float
    effective_weight: float


@dataclass(frozen=True, slots=True)
class CoalitionTrace:
    selector_ids: tuple[SelectorId, ...]
    coverage: float
    diversity: float
    historical_utility: float
    redundancy: float
    conflict_cost: float
    complexity_cost: float
    score: float
    connectivity: tuple[ConnectivityTrace, ...]


@dataclass(frozen=True, slots=True)
class CandidateCoalitionDecomposition:
    candidate_id: CandidateId
    prior_log_probability: float
    selector_log_evidence: tuple[tuple[SelectorId, float], ...]
    support_total: float
    opposition_total: float
    uncertainty_total: float
    posterior: float


@dataclass(frozen=True, slots=True)
class PosteriorAlignment:
    winning_posterior: float
    selector_agreement: float
    independent_support: float
    redundancy_penalty: float
    dissent_mass: float
    uncertainty_mass: float
    novelty_mass: float
    alignment_index: float


class StructuralOperation(str, Enum):
    BROADEN_SELECTOR = "broaden_selector"
    CREATE_SPECIALIST_SELECTOR = "create_specialist_selector"
    DISCONNECT_REDUNDANT_SELECTORS = "disconnect_redundant_selectors"
    SHADOW_ALTERNATIVE_COALITION = "shadow_alternative_coalition"


@dataclass(frozen=True, slots=True)
class StructuralProposal:
    operation: StructuralOperation
    selector_ids: tuple[SelectorId, ...]
    evidence: tuple[tuple[str, float], ...]
    preconditions: tuple[str, ...]
    rationale: str
    estimated_gain: float
    complexity_cost: float
    advisory_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "selector_ids", tuple(self.selector_ids))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "preconditions", tuple(self.preconditions))
        if not self.selector_ids:
            raise ValueError("Structural proposal requires at least one selector.")
        if not self.advisory_only:
            raise ValueError(
                "This realization emits proposals; it does not apply structural transitions."
            )
        if (
            not isfinite(self.estimated_gain)
            or not isfinite(self.complexity_cost)
            or not 0.0 <= self.estimated_gain <= 1.0
            or self.complexity_cost < 0.0
        ):
            raise ValueError(
                "Proposal gain must lie in [0, 1] and cost must be finite/non-negative."
            )
        require_finite(
            (value for _, value in self.evidence),
            label="Structural proposal evidence",
        )


@dataclass(frozen=True, slots=True)
class DecisionDecomposition:
    generation: int
    input_vector: Vector
    selector_votes: tuple[SelectorVote, ...]
    coalition: CoalitionTrace
    candidate_posteriors: tuple[CandidateCoalitionDecomposition, ...]
    selected_candidate_id: CandidateId
    posterior_alignment: PosteriorAlignment
    novelty_score: float
    structural_proposals: tuple[StructuralProposal, ...]
    formal_model_id: str
    snapshot_fingerprint: str
    evaluated_ns: int
    exploration_seed: int

    def candidate(self, candidate_id: CandidateId) -> CandidateCoalitionDecomposition:
        for item in self.candidate_posteriors:
            if item.candidate_id == candidate_id:
                return item
        raise KeyError(candidate_id)


# ============================================================
# Immutable snapshot and evaluation
# ============================================================

Constraint = Callable[[Vector, CandidateModel], bool]


@dataclass(frozen=True, slots=True)
class CognitiveFabricSnapshot:
    generation: int
    transforms: tuple[ReversibleTransform, ...]
    candidates: tuple[CandidateModel, ...]
    selectors: tuple[SelectorRegion, ...]
    edges: tuple[SelectorEdge, ...]
    fallback_precision: PrecisionModel
    created_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transforms",
            tuple(sorted(self.transforms, key=lambda item: item.transform_id)),
        )
        object.__setattr__(
            self,
            "candidates",
            tuple(sorted(self.candidates, key=lambda item: item.candidate_id)),
        )
        object.__setattr__(
            self,
            "selectors",
            tuple(sorted(self.selectors, key=lambda item: item.selector_id)),
        )
        object.__setattr__(
            self,
            "edges",
            tuple(
                sorted(
                    self.edges,
                    key=lambda item: (item.source_id, item.target_id),
                )
            ),
        )
        validate_snapshot(self)

    def transform_map(self) -> dict[TransformId, ReversibleTransform]:
        return {item.transform_id: item for item in self.transforms}

    def candidate_map(self) -> dict[CandidateId, CandidateModel]:
        return {item.candidate_id: item for item in self.candidates}

    def selector_map(self) -> dict[SelectorId, SelectorRegion]:
        return {item.selector_id: item for item in self.selectors}


def _canonical_value(value: object) -> object:
    if is_dataclass(value):
        return {
            "__type__": type(value).__name__,
            **{
                item.name: _canonical_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Snapshot identity cannot encode non-finite floats.")
        return {"__float_hex__": value.hex()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported value in formal snapshot: {type(value).__name__}.")


def snapshot_fingerprint(snapshot: CognitiveFabricSnapshot) -> str:
    payload = {
        "formal_model_id": FORMAL_MODEL_ID,
        "snapshot": _canonical_value(snapshot),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def validate_snapshot(snapshot: CognitiveFabricSnapshot) -> None:
    if snapshot.generation < 0:
        raise ValueError("Snapshot generation cannot be negative.")
    if snapshot.created_ns < 0:
        raise ValueError("Snapshot creation time cannot be negative.")
    if not snapshot.candidates:
        raise ValueError("Snapshot requires at least one candidate.")
    if not any(candidate.enabled for candidate in snapshot.candidates):
        raise ValueError("Snapshot requires at least one enabled candidate.")
    if not snapshot.selectors:
        raise ValueError("Snapshot requires at least one selector.")
    if not any(
        selector.enabled and selector.health.lifecycle != SelectorLifecycle.RETIRED
        for selector in snapshot.selectors
    ):
        raise ValueError("Snapshot requires at least one routable selector.")

    if any(
        not isinstance(transform, (OrthogonalTransform, AffineTransform))
        for transform in snapshot.transforms
    ):
        raise TypeError(
            "The formal reference realization accepts only certified built-in transforms."
        )

    transform_ids = [transform.transform_id for transform in snapshot.transforms]
    candidate_ids = [candidate.candidate_id for candidate in snapshot.candidates]
    selector_ids = [selector.selector_id for selector in snapshot.selectors]
    if len(transform_ids) != len(set(transform_ids)):
        raise ValueError("Transform ids must be unique.")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate ids must be unique.")
    if len(selector_ids) != len(set(selector_ids)):
        raise ValueError("Selector ids must be unique.")

    dimension = snapshot.fallback_precision.dimension
    transform_id_set = set(transform_ids)
    candidate_id_set = set(candidate_ids)
    selector_id_set = set(selector_ids)

    for transform in snapshot.transforms:
        if len(transform.matrix) != dimension:
            raise ValueError("Transform/fabric dimension mismatch.")

    for selector in snapshot.selectors:
        if len(selector.center) != dimension:
            raise ValueError("Selector/fabric dimension mismatch.")
        missing_transforms = set(selector.transform_chain) - transform_id_set
        if missing_transforms:
            raise ValueError(
                f"Selector {selector.selector_id!r} references unknown transforms: "
                f"{sorted(missing_transforms)!r}."
            )
        local_candidate_ids = {
            model.candidate_id for model in selector.candidate_models
        }
        if local_candidate_ids != candidate_id_set:
            raise ValueError(
                f"Selector {selector.selector_id!r} must model every candidate exactly once."
            )

    edge_keys: set[tuple[SelectorId, SelectorId]] = set()
    for edge in snapshot.edges:
        key = (edge.source_id, edge.target_id)
        if key in edge_keys:
            raise ValueError(f"Duplicate selector edge: {key!r}.")
        edge_keys.add(key)
        if edge.source_id not in selector_id_set or edge.target_id not in selector_id_set:
            raise ValueError(f"Edge {key!r} references an unknown selector.")


def freshness(statistics: CandidateStatistics, now_ns: int, half_life_ns: int = 60_000_000_000) -> float:
    if statistics.last_observed_ns <= 0:
        return 0.5
    age = max(0, now_ns - statistics.last_observed_ns)
    return 0.5 + 0.5 * exp(-age / max(1.0, float(half_life_ns)))


def uncertainty_penalty(statistics: CandidateStatistics) -> float:
    return 0.25 + 0.75 * statistics.observations / (statistics.observations + 16.0)


def selector_activation(
    selector: SelectorRegion,
    transformed: Vector,
) -> ActivationDecomposition:
    if len(transformed) != len(selector.center):
        raise ValueError(f"Selector {selector.selector_id!r} dimension mismatch.")

    delta = tuple(value - center for value, center in zip(transformed, selector.center))
    precision = selector.precision.matrix()
    mahalanobis = max(0.0, quadratic_form(delta, precision))
    log_activation = -0.5 * mahalanobis

    routing = max(
        selector.routing_floor,
        exp(max(log_activation, log(selector.routing_floor))),
    )
    learning = max(selector.learning_floor, routing)

    inactivity_bonus = clamp(selector.health.inactive_steps / 128.0, 0.0, 0.35)
    exploration = clamp(
        selector.exploration_floor
        + inactivity_bonus
        + 0.15 * selector.novelty_tolerance,
        0.0,
        1.0,
    )

    return ActivationDecomposition(
        selector_id=selector.selector_id,
        delta=delta,
        mahalanobis_squared=mahalanobis,
        log_activation=log_activation,
        routing_activation=routing,
        learning_activation=learning,
        exploration_probability=exploration,
        precision=precision,
        precision_diagnostics=selector.precision.diagnostics(),
        health_before=selector.health,
    )


def evaluate_selector(
    selector: SelectorRegion,
    vector: Vector,
    transforms: Mapping[TransformId, ReversibleTransform],
    candidates: Mapping[CandidateId, CandidateModel],
    *,
    now_ns: int,
    evaluation_reason: str,
) -> SelectorVote:
    transform_trace = apply_transform_chain(vector, selector.transform_chain, transforms)
    activation = selector_activation(selector, transform_trace.final_vector)

    scores: list[tuple[SelectorCandidateModel, CandidateScoreDecomposition]] = []
    logits: dict[CandidateId, float] = {}

    for model in selector.candidate_models:
        candidate = candidates.get(model.candidate_id)
        if candidate is None or not candidate.enabled:
            continue

        spline_trace = model.spline.evaluate(transform_trace.final_vector)
        stats = model.statistics

        components = ProbabilityComponents(
            prior=candidate.prior,
            selector_activation=activation.routing_activation,
            spline_fit=spline_trace.probability,
            historical_success=stats.empirical_success,
            reward_quality=clamp(stats.reward_mean, 0.0, 1.0),
            freshness=freshness(stats, now_ns),
            reliability=clamp(selector.reliability, 1e-6, 1.0),
            uncertainty_penalty=uncertainty_penalty(stats),
        )

        log_score = components.log_score()
        logits[candidate.candidate_id] = log_score

        scores.append(
            (
                model,
                CandidateScoreDecomposition(
                    selector_id=selector.selector_id,
                    candidate_id=candidate.candidate_id,
                    spline=spline_trace,
                    components=components,
                    raw_product=components.raw_product(),
                    log_score=log_score,
                    local_posterior=0.0,
                    support=0.0,
                    opposition=0.0,
                    uncertainty=0.0,
                ),
            )
        )

    local_posteriors = softmax(logits)
    completed: list[CandidateScoreDecomposition] = []

    for _, score in scores:
        posterior = local_posteriors[score.candidate_id]
        uncertainty = clamp(
            0.5 * (1.0 - activation.routing_activation)
            + 0.5 * normalized_entropy(local_posteriors.values()),
            0.0,
            1.0,
        )
        support = posterior * (1.0 - uncertainty)
        opposition = (1.0 - posterior) * (1.0 - uncertainty)

        completed.append(
            replace(
                score,
                local_posterior=posterior,
                support=support,
                opposition=opposition,
                uncertainty=uncertainty,
            )
        )

    novelty = clamp(
        (1.0 - activation.routing_activation)
        * selector.novelty_tolerance,
        0.0,
        1.0,
    )

    return SelectorVote(
        selector_id=selector.selector_id,
        activation=activation,
        transform_trace=transform_trace,
        candidate_scores=tuple(completed),
        novelty=novelty,
        expected_information_gain=clamp(
            normalized_entropy(local_posteriors.values()) * selector.reliability,
            0.0,
            1.0,
        ),
        evaluation_reason=evaluation_reason,
    )


def edge_trace(edge: SelectorEdge) -> ConnectivityTrace:
    return ConnectivityTrace(
        source_id=edge.source_id,
        target_id=edge.target_id,
        compatibility=edge.compatibility,
        information_gain=edge.information_gain,
        redundancy=edge.redundancy,
        conflict_rate=edge.conflict_rate,
        effective_weight=edge.weight(),
    )


def vote_distance(left: SelectorVote, right: SelectorVote) -> float:
    left_map = {item.candidate_id: item.local_posterior for item in left.candidate_scores}
    right_map = {item.candidate_id: item.local_posterior for item in right.candidate_scores}
    keys = sorted(set(left_map) | set(right_map))
    return 0.5 * fsum(
        abs(left_map.get(key, 0.0) - right_map.get(key, 0.0))
        for key in keys
    )


def score_coalition(
    selected_votes: Sequence[SelectorVote],
    edges: Sequence[SelectorEdge],
) -> CoalitionTrace:
    selected = tuple(sorted(selected_votes, key=lambda vote: vote.selector_id))
    if not selected:
        raise ValueError("Cannot score an empty coalition.")
    selected_ids = {vote.selector_id for vote in selected}
    relevant_edges = tuple(
        edge_trace(edge)
        for edge in sorted(
            edges,
            key=lambda item: (item.source_id, item.target_id),
        )
        if edge.enabled
        and edge.source_id in selected_ids
        and edge.target_id in selected_ids
    )

    uncovered = 1.0
    for vote in selected:
        uncovered *= 1.0 - clamp(
            vote.activation.routing_activation,
            0.0,
            1.0,
        )
    coverage = 1.0 - uncovered

    distances = [
        vote_distance(selected[left], selected[right])
        for left in range(len(selected))
        for right in range(left + 1, len(selected))
    ]
    diversity = fsum(distances) / len(distances) if distances else 0.0
    historical_utility = fsum(
        clamp(
            vote.expected_information_gain
            + 0.5 * vote.activation.routing_activation,
            0.0,
            1.0,
        )
        for vote in selected
    ) / len(selected)
    redundancy = (
        fsum(
            trace.redundancy * trace.effective_weight
            for trace in relevant_edges
        )
        / max(1, len(relevant_edges))
    )
    conflict_cost = (
        fsum(
            trace.conflict_rate * trace.effective_weight
            for trace in relevant_edges
        )
        / max(1, len(relevant_edges))
    )
    complexity_cost = fsum(
        vote.transform_trace.total_complexity_cost
        for vote in selected
    )
    score = (
        1.4 * coverage
        + 0.8 * diversity
        + 1.0 * historical_utility
        - 0.9 * redundancy
        - 0.8 * conflict_cost
        - 0.03 * complexity_cost
    )
    return CoalitionTrace(
        selector_ids=tuple(vote.selector_id for vote in selected),
        coverage=coverage,
        diversity=diversity,
        historical_utility=historical_utility,
        redundancy=redundancy,
        conflict_cost=conflict_cost,
        complexity_cost=complexity_cost,
        score=score,
        connectivity=relevant_edges,
    )


def build_coalition(
    votes: Sequence[SelectorVote],
    edges: Sequence[SelectorEdge],
    *,
    maximum_size: int = 4,
) -> CoalitionTrace:
    if not votes:
        raise RuntimeError("No selector votes available.")
    if maximum_size < 1:
        raise ValueError("Maximum coalition size must be positive.")

    ordered_votes = tuple(sorted(votes, key=lambda vote: vote.selector_id))
    candidates_to_compare = tuple(
        score_coalition(tuple(selected), edges)
        for size in range(1, min(maximum_size, len(ordered_votes)) + 1)
        for selected in combinations(ordered_votes, size)
    )
    return min(
        candidates_to_compare,
        key=lambda trace: (-trace.score, trace.selector_ids),
    )


def aggregate_coalition(
    coalition: CoalitionTrace,
    votes: Sequence[SelectorVote],
    candidates: Mapping[CandidateId, CandidateModel],
) -> tuple[CandidateCoalitionDecomposition, ...]:
    vote_map = {vote.selector_id: vote for vote in votes}
    selected_votes = [vote_map[selector_id] for selector_id in coalition.selector_ids]

    logits: dict[CandidateId, float] = {}
    traces: dict[CandidateId, tuple[float, list[tuple[SelectorId, float]], float, float, float]] = {}
    enabled_candidates = {
        candidate_id: candidate
        for candidate_id, candidate in candidates.items()
        if candidate.enabled
    }
    prior_total = fsum(
        candidate.prior for candidate in enabled_candidates.values()
    )
    if prior_total <= 0.0:
        raise RuntimeError("Enabled candidate prior mass must be positive.")
    prior_distribution = {
        candidate_id: candidate.prior / prior_total
        for candidate_id, candidate in enabled_candidates.items()
    }
    raw_selector_weights = {
        vote.selector_id: (
            clamp(vote.activation.routing_activation, 0.0, 1.0)
            * clamp(
                max(
                    (
                        score.components.reliability
                        for score in vote.candidate_scores
                    ),
                    default=0.0,
                ),
                0.0,
                1.0,
            )
        )
        for vote in selected_votes
    }
    weight_normalizer = max(1.0, fsum(raw_selector_weights.values()))

    for candidate_id in sorted(enabled_candidates):
        prior_log_probability = log(
            safe_probability(prior_distribution[candidate_id])
        )
        selector_terms: list[tuple[SelectorId, float]] = []
        support_total = 0.0
        opposition_total = 0.0
        uncertainty_total = 0.0

        for vote in selected_votes:
            score = next(item for item in vote.candidate_scores if item.candidate_id == candidate_id)
            p = clamp(score.local_posterior, 1e-12, 1.0 - 1e-12)
            reliability = score.components.reliability
            activation = vote.activation.routing_activation
            weight = reliability * activation / weight_normalizer
            term = weight * (
                log(p) - prior_log_probability
            )
            selector_terms.append((vote.selector_id, term))
            support_total += score.support
            opposition_total += score.opposition
            uncertainty_total += score.uncertainty

        total_logit = prior_log_probability + fsum(
            term for _, term in selector_terms
        )
        logits[candidate_id] = total_logit
        traces[candidate_id] = (
            prior_log_probability,
            selector_terms,
            support_total,
            opposition_total,
            uncertainty_total,
        )

    posteriors = softmax(logits)

    return tuple(
        CandidateCoalitionDecomposition(
            candidate_id=candidate_id,
            prior_log_probability=traces[candidate_id][0],
            selector_log_evidence=tuple(traces[candidate_id][1]),
            support_total=traces[candidate_id][2],
            opposition_total=traces[candidate_id][3],
            uncertainty_total=traces[candidate_id][4],
            posterior=posteriors[candidate_id],
        )
        for candidate_id in sorted(posteriors)
    )


def posterior_alignment(
    coalition: CoalitionTrace,
    votes: Sequence[SelectorVote],
    candidate_posteriors: Sequence[CandidateCoalitionDecomposition],
) -> PosteriorAlignment:
    vote_map = {vote.selector_id: vote for vote in votes}
    selected_votes = [vote_map[selector_id] for selector_id in coalition.selector_ids]
    winner = min(
        candidate_posteriors,
        key=lambda item: (-item.posterior, item.candidate_id),
    )

    winner_supports: list[tuple[float, float]] = []
    dissent = 0.0
    uncertainty = 0.0
    novelty = 0.0
    total_weight = 0.0

    for vote in selected_votes:
        score = next(item for item in vote.candidate_scores if item.candidate_id == winner.candidate_id)
        weight = (
            clamp(vote.activation.routing_activation, 0.0, 1.0)
            * clamp(score.components.reliability, 0.0, 1.0)
        )
        winner_supports.append((score.local_posterior, weight))
        dissent += weight * score.opposition
        uncertainty += weight * score.uncertainty
        novelty += weight * vote.novelty
        total_weight += weight

    denominator = max(total_weight, 1e-12)
    mean = sum(value * weight for value, weight in winner_supports) / denominator
    variance = (
        sum(
            weight * (value - mean) ** 2
            for value, weight in winner_supports
        )
        / denominator
    )
    selector_agreement = clamp(
        mean * (1.0 - sqrt(max(0.0, variance))),
        0.0,
        1.0,
    )

    independent_support = clamp(
        mean * (1.0 - coalition.redundancy),
        0.0,
        1.0,
    )

    dissent_mass = dissent / denominator
    uncertainty_mass = uncertainty / denominator
    novelty_mass = novelty / denominator

    alignment = clamp(
        0.35 * winner.posterior
        + 0.30 * selector_agreement
        + 0.25 * independent_support
        - 0.15 * coalition.redundancy
        - 0.15 * dissent_mass
        - 0.10 * uncertainty_mass
        - 0.10 * novelty_mass,
        0.0,
        1.0,
    )

    return PosteriorAlignment(
        winning_posterior=winner.posterior,
        selector_agreement=selector_agreement,
        independent_support=independent_support,
        redundancy_penalty=coalition.redundancy,
        dissent_mass=dissent_mass,
        uncertainty_mass=uncertainty_mass,
        novelty_mass=novelty_mass,
        alignment_index=alignment,
    )


def structural_proposals(
    votes: Sequence[SelectorVote],
    coalition: CoalitionTrace,
    alignment: PosteriorAlignment,
) -> tuple[StructuralProposal, ...]:
    proposals: list[StructuralProposal] = []

    low_activation = [
        vote.selector_id
        for vote in votes
        if vote.activation.routing_activation < 1e-3
    ]
    if low_activation:
        proposals.append(
            StructuralProposal(
                operation=StructuralOperation.BROADEN_SELECTOR,
                selector_ids=tuple(sorted(low_activation)),
                evidence=(
                    ("routing_activation_upper_bound", 1e-3),
                    ("affected_selector_count", float(len(low_activation))),
                ),
                preconditions=(
                    "base snapshot fingerprint still current",
                    "each named selector remains enabled",
                    "post-change precision satisfies spectral bounds",
                ),
                rationale=(
                    "Named selectors have negligible routing activation; "
                    "broader regions should be evaluated in shadow mode."
                ),
                estimated_gain=0.35,
                complexity_cost=0.15,
            )
        )

    if alignment.novelty_mass > 0.55:
        proposals.append(
            StructuralProposal(
                operation=StructuralOperation.CREATE_SPECIALIST_SELECTOR,
                selector_ids=coalition.selector_ids,
                evidence=(
                    ("novelty_mass", alignment.novelty_mass),
                    ("trigger_threshold", 0.55),
                ),
                preconditions=(
                    "base snapshot fingerprint still current",
                    "a reversible transform chain is available",
                    "candidate coverage remains complete",
                ),
                rationale="Coalition novelty is high; existing selector geometry may not cover the observation.",
                estimated_gain=alignment.novelty_mass,
                complexity_cost=0.45,
            )
        )

    if coalition.redundancy > 0.65:
        proposals.append(
            StructuralProposal(
                operation=StructuralOperation.DISCONNECT_REDUNDANT_SELECTORS,
                selector_ids=coalition.selector_ids,
                evidence=(
                    ("coalition_redundancy", coalition.redundancy),
                    ("trigger_threshold", 0.65),
                ),
                preconditions=(
                    "base snapshot fingerprint still current",
                    "at least one routing path remains for every enabled selector",
                ),
                rationale="Coalition carries excessive redundant evidence.",
                estimated_gain=coalition.redundancy * 0.5,
                complexity_cost=0.20,
            )
        )

    if alignment.dissent_mass > 0.55:
        proposals.append(
            StructuralProposal(
                operation=StructuralOperation.SHADOW_ALTERNATIVE_COALITION,
                selector_ids=coalition.selector_ids,
                evidence=(
                    ("dissent_mass", alignment.dissent_mass),
                    ("trigger_threshold", 0.55),
                ),
                preconditions=(
                    "base snapshot fingerprint still current",
                    "shadow evaluation cannot affect the committed decision",
                ),
                rationale="Strong dissent should be preserved and tested rather than averaged away.",
                estimated_gain=alignment.dissent_mass * 0.6,
                complexity_cost=0.30,
            )
        )

    return tuple(proposals)


def evaluate_snapshot(
    snapshot: CognitiveFabricSnapshot,
    input_vector: Sequence[float],
    *,
    exploration_seed: int | None = None,
    maximum_coalition_size: int = 4,
    evaluation_ns: int | None = None,
) -> DecisionDecomposition:
    vector = tuple(float(value) for value in input_vector)
    require_finite(vector, label="Input vector")
    if len(vector) != snapshot.fallback_precision.dimension:
        raise ValueError("Input/fabric dimension mismatch.")
    if maximum_coalition_size < 1:
        raise ValueError("Maximum coalition size must be positive.")
    transforms = snapshot.transform_map()
    candidates = snapshot.candidate_map()
    now_ns = monotonic_ns() if evaluation_ns is None else int(evaluation_ns)
    if now_ns < 0:
        raise ValueError("Evaluation time cannot be negative.")
    resolved_seed = 0 if exploration_seed is None else int(exploration_seed)

    provisional: list[tuple[SelectorRegion, TransformChainTrace, ActivationDecomposition]] = []

    for selector in snapshot.selectors:
        if not selector.enabled or selector.health.lifecycle == SelectorLifecycle.RETIRED:
            continue
        trace = apply_transform_chain(vector, selector.transform_chain, transforms)
        activation = selector_activation(selector, trace.final_vector)
        provisional.append((selector, trace, activation))

    if not provisional:
        raise RuntimeError("No enabled selectors.")

    ranked = sorted(
        provisional,
        key=lambda item: (
            -item[2].routing_activation,
            item[0].selector_id,
        ),
    )

    selected_ids = {item[0].selector_id for item in ranked[: min(3, len(ranked))]}

    remaining = [item for item in ranked if item[0].selector_id not in selected_ids]
    if remaining:
        exploratory = min(
            sorted(remaining, key=lambda item: item[0].selector_id),
            key=lambda item: (
                -(
                    item[2].exploration_probability
                    + 0.01
                    * seeded_jitter(resolved_seed, item[0].selector_id)
                ),
                item[0].selector_id,
            ),
        )
        selected_ids.add(exploratory[0].selector_id)

    votes: list[SelectorVote] = []
    top_ids = {
        item[0].selector_id
        for item in ranked[: min(3, len(ranked))]
    }
    for selector, _, activation in sorted(
        provisional,
        key=lambda item: item[0].selector_id,
    ):
        if selector.selector_id not in selected_ids:
            continue
        reason = (
            "top_activation"
            if selector.selector_id in top_ids
            else "exploration_recovery"
        )
        votes.append(
            evaluate_selector(
                selector,
                vector,
                transforms,
                candidates,
                now_ns=now_ns,
                evaluation_reason=reason,
            )
        )

    coalition = build_coalition(
        votes,
        snapshot.edges,
        maximum_size=maximum_coalition_size,
    )
    posteriors = aggregate_coalition(coalition, votes, candidates)
    selected_candidate = min(
        posteriors,
        key=lambda item: (-item.posterior, item.candidate_id),
    ).candidate_id
    alignment = posterior_alignment(coalition, votes, posteriors)

    max_activation = max(vote.activation.routing_activation for vote in votes)
    disagreement = normalized_entropy(item.posterior for item in posteriors)
    novelty = clamp(
        0.45 * (1.0 - max_activation)
        + 0.30 * disagreement
        + 0.25 * alignment.novelty_mass,
        0.0,
        1.0,
    )

    proposals = structural_proposals(votes, coalition, alignment)

    return DecisionDecomposition(
        generation=snapshot.generation,
        input_vector=vector,
        selector_votes=tuple(votes),
        coalition=coalition,
        candidate_posteriors=posteriors,
        selected_candidate_id=selected_candidate,
        posterior_alignment=alignment,
        novelty_score=novelty,
        structural_proposals=proposals,
        formal_model_id=FORMAL_MODEL_ID,
        snapshot_fingerprint=snapshot_fingerprint(snapshot),
        evaluated_ns=now_ns,
        exploration_seed=resolved_seed,
    )


def decision_invariant_violations(
    decision: DecisionDecomposition,
    *,
    tolerance: float = 1e-9,
) -> tuple[str, ...]:
    violations: list[str] = []
    if decision.formal_model_id != FORMAL_MODEL_ID:
        violations.append("formal model id mismatch")
    if decision.evaluated_ns < 0:
        violations.append("negative evaluation time")
    if not 0.0 <= decision.novelty_score <= 1.0:
        violations.append("novelty score outside [0, 1]")
    if not 0.0 <= decision.posterior_alignment.alignment_index <= 1.0:
        violations.append("alignment index outside [0, 1]")

    posterior_values = [
        item.posterior for item in decision.candidate_posteriors
    ]
    if not posterior_values:
        violations.append("empty coalition posterior")
    elif (
        any(
            not isfinite(value) or not 0.0 <= value <= 1.0
            for value in posterior_values
        )
        or abs(sum(posterior_values) - 1.0) > tolerance
    ):
        violations.append("coalition posterior is not on the simplex")

    posterior_map = {
        item.candidate_id: item.posterior
        for item in decision.candidate_posteriors
    }
    if decision.selected_candidate_id not in posterior_map:
        violations.append("winner missing from coalition posterior")
    elif posterior_map:
        expected_winner = min(
            posterior_map,
            key=lambda candidate_id: (
                -posterior_map[candidate_id],
                candidate_id,
            ),
        )
        if decision.selected_candidate_id != expected_winner:
            violations.append("winner violates canonical argmax")

    vote_ids = {vote.selector_id for vote in decision.selector_votes}
    coalition_ids = decision.coalition.selector_ids
    if not coalition_ids:
        violations.append("empty coalition")
    if len(coalition_ids) != len(set(coalition_ids)):
        violations.append("duplicate selector in coalition")
    if not set(coalition_ids).issubset(vote_ids):
        violations.append("coalition contains an unevaluated selector")

    for vote in decision.selector_votes:
        local = [score.local_posterior for score in vote.candidate_scores]
        if not local or abs(sum(local) - 1.0) > tolerance:
            violations.append(
                f"selector {vote.selector_id!r} posterior is not on the simplex"
            )
        if vote.transform_trace.chain_reconstruction_error > (
            tolerance
            * (1.0 + sqrt(squared_norm(vote.transform_trace.initial_vector)))
        ):
            violations.append(
                f"selector {vote.selector_id!r} transform round-trip exceeds tolerance"
            )
        for score in vote.candidate_scores:
            mass = score.support + score.opposition + score.uncertainty
            if abs(mass - 1.0) > tolerance:
                violations.append(
                    f"selector {vote.selector_id!r}, candidate "
                    f"{score.candidate_id!r} vote mass does not sum to one"
                )

    return tuple(violations)


def assert_decision_invariants(
    decision: DecisionDecomposition,
    *,
    tolerance: float = 1e-9,
) -> None:
    violations = decision_invariant_violations(
        decision,
        tolerance=tolerance,
    )
    if violations:
        raise AssertionError("; ".join(violations))


# ============================================================
# Lazy outcomes and learning
# ============================================================

@dataclass(frozen=True, slots=True)
class Outcome:
    generation: int
    input_vector: Vector
    selected_candidate_id: CandidateId
    reward: float
    succeeded: bool
    latency_ns: int
    observed_ns: int
    decomposition: DecisionDecomposition
    formal_model_id: str
    snapshot_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_vector",
            tuple(float(value) for value in self.input_vector),
        )
        if self.generation < 0:
            raise ValueError("Outcome generation cannot be negative.")
        require_finite(self.input_vector, label="Outcome input")
        if not self.selected_candidate_id:
            raise ValueError("Outcome candidate id cannot be empty.")
        if not isfinite(self.reward) or not 0.0 <= self.reward <= 1.0:
            raise ValueError("Outcome reward must lie in [0, 1].")
        if self.latency_ns < 0 or self.observed_ns < 0:
            raise ValueError("Outcome time values cannot be negative.")


def running_mean(mean: float, count: int, value: float) -> float:
    return mean + (value - mean) / (count + 1)


def update_candidate_statistics(
    statistics: CandidateStatistics,
    outcome: Outcome,
) -> CandidateStatistics:
    return CandidateStatistics(
        observations=statistics.observations + 1,
        successes=statistics.successes + int(outcome.succeeded),
        reward_mean=running_mean(
            statistics.reward_mean,
            statistics.observations,
            clamp(outcome.reward, 0.0, 1.0),
        ),
        latency_mean_ns=running_mean(
            statistics.latency_mean_ns,
            statistics.observations,
            float(max(0, outcome.latency_ns)),
        ),
        last_observed_ns=outcome.observed_ns,
    )


def learn_selector(
    selector: SelectorRegion,
    outcomes: Sequence[Outcome],
    fallback_precision: PrecisionModel,
    *,
    spline_learning_rate: float,
    reliability_learning_rate: float,
    precision_learning_rate: float,
) -> SelectorRegion:
    updated = selector

    for outcome in outcomes:
        matching_vote = next(
            (
                vote
                for vote in outcome.decomposition.selector_votes
                if vote.selector_id == selector.selector_id
            ),
            None,
        )
        if matching_vote is None:
            continue

        selected_score = next(
            score
            for score in matching_vote.candidate_scores
            if score.candidate_id == outcome.selected_candidate_id
        )

        prediction_error = clamp(outcome.reward, 0.0, 1.0) - selected_score.spline.probability
        locality = matching_vote.activation.learning_activation

        candidate_models = []
        update_norm = 0.0

        for model in updated.candidate_models:
            if model.candidate_id == outcome.selected_candidate_id:
                new_spline = model.spline.learned(
                    selected_score.spline,
                    prediction_error,
                    spline_learning_rate,
                    locality,
                )
                stats = update_candidate_statistics(model.statistics, outcome)
                update_norm += abs(prediction_error) * spline_learning_rate * locality
                candidate_models.append(
                    replace(model, spline=new_spline, statistics=stats)
                )
            else:
                candidate_models.append(model)

        reliability_target = clamp(
            1.0 - abs(prediction_error),
            0.0,
            1.0,
        )
        reliability = clamp(
            updated.reliability
            + reliability_learning_rate
            * (reliability_target - updated.reliability),
            0.01,
            1.0,
        )

        health = updated.health.observed(
            matching_vote.activation.routing_activation,
            matching_vote.activation.learning_activation,
            update_norm,
        )

        precision = updated.precision
        if health.lifecycle in {SelectorLifecycle.DORMANT, SelectorLifecycle.RECOVERY}:
            precision = precision.broaden(0.95)
            shrinkage = clamp(
                1.0 - health.observations / 64.0,
                0.05,
                0.60,
            )
            precision = precision.shrink_toward(fallback_precision, shrinkage)
        elif health.lifecycle == SelectorLifecycle.HEALTHY:
            # Conservative precision adaptation in the positive-definite cone.
            delta = matching_vote.activation.delta
            direction = outer(delta)
            current = precision.matrix()
            floor = matrix_scale(
                identity_matrix(precision.dimension),
                precision.spectral_floor,
            )
            current_base = matrix_subtract(current, floor)
            adaptation = clamp(
                precision_learning_rate * locality,
                0.0,
                0.25,
            )
            if adaptation > 0.0:
                blended_base = matrix_add(
                    matrix_scale(current_base, 1.0 - adaptation),
                    matrix_scale(direction, adaptation),
                )
                try:
                    precision = PrecisionModel.from_base_matrix(
                        blended_base,
                        template=precision,
                    )
                except ValueError:
                    precision = precision.shrink_toward(
                        fallback_precision,
                        0.50,
                    )
            if not precision.diagnostics().healthy:
                precision = precision.shrink_toward(fallback_precision, 0.50)

        updated = replace(
            updated,
            candidate_models=tuple(candidate_models),
            reliability=reliability,
            health=health,
            precision=precision,
        )

    return updated


def update_edges(
    edges: Sequence[SelectorEdge],
    outcomes: Sequence[Outcome],
    *,
    learning_rate: float = 0.02,
) -> tuple[SelectorEdge, ...]:
    updated: list[SelectorEdge] = []

    for edge in edges:
        compatibility = edge.compatibility
        information_gain = edge.information_gain
        redundancy = edge.redundancy
        conflict_rate = edge.conflict_rate

        for outcome in outcomes:
            vote_map = {
                vote.selector_id: vote
                for vote in outcome.decomposition.selector_votes
            }
            left = vote_map.get(edge.source_id)
            right = vote_map.get(edge.target_id)
            if left is None or right is None:
                continue

            distance = vote_distance(left, right)
            agreement = 1.0 - distance
            useful = clamp(outcome.reward, 0.0, 1.0)

            compatibility += learning_rate * (agreement - compatibility)
            conflict_rate += learning_rate * (distance - conflict_rate)
            redundancy_target = agreement * min(
                left.activation.routing_activation,
                right.activation.routing_activation,
            )
            redundancy += learning_rate * (redundancy_target - redundancy)
            information_target = useful * distance
            information_gain += learning_rate * (
                information_target - information_gain
            )

        updated.append(
            replace(
                edge,
                compatibility=clamp(compatibility, 0.0, 1.0),
                information_gain=clamp(information_gain, 0.0, 1.0),
                redundancy=clamp(redundancy, 0.0, 1.0),
                conflict_rate=clamp(conflict_rate, 0.0, 1.0),
            )
        )

    return tuple(updated)


def outcome_matches_snapshot(
    outcome: Outcome,
    snapshot: CognitiveFabricSnapshot,
    *,
    fingerprint: str | None = None,
) -> bool:
    expected_fingerprint = (
        snapshot_fingerprint(snapshot)
        if fingerprint is None
        else fingerprint
    )
    return (
        outcome.generation == snapshot.generation
        and outcome.formal_model_id == FORMAL_MODEL_ID
        and outcome.snapshot_fingerprint == expected_fingerprint
        and outcome.decomposition.formal_model_id == FORMAL_MODEL_ID
        and (
            outcome.decomposition.snapshot_fingerprint
            == expected_fingerprint
        )
        and outcome.decomposition.generation == outcome.generation
        and outcome.decomposition.input_vector == outcome.input_vector
        and (
            outcome.decomposition.selected_candidate_id
            == outcome.selected_candidate_id
        )
    )


def apply_outcomes(
    snapshot: CognitiveFabricSnapshot,
    outcomes: Iterable[Outcome],
    *,
    spline_learning_rate: float = 0.05,
    reliability_learning_rate: float = 0.02,
    precision_learning_rate: float = 0.002,
    commit_ns: int | None = None,
) -> CognitiveFabricSnapshot:
    learning_rates = (
        spline_learning_rate,
        reliability_learning_rate,
        precision_learning_rate,
    )
    require_finite(learning_rates, label="Learning rates")
    if any(not 0.0 <= value <= 1.0 for value in learning_rates):
        raise ValueError("Learning rates must lie in [0, 1].")
    fingerprint = snapshot_fingerprint(snapshot)
    valid = tuple(
        outcome
        for outcome in outcomes
        if outcome_matches_snapshot(
            outcome,
            snapshot,
            fingerprint=fingerprint,
        )
    )
    if not valid:
        return snapshot
    created_ns = monotonic_ns() if commit_ns is None else int(commit_ns)
    if created_ns < 0:
        raise ValueError("Commit time cannot be negative.")

    updated_selectors = tuple(
        learn_selector(
            selector,
            valid,
            snapshot.fallback_precision,
            spline_learning_rate=spline_learning_rate,
            reliability_learning_rate=reliability_learning_rate,
            precision_learning_rate=precision_learning_rate,
        )
        for selector in snapshot.selectors
    )

    updated_edges = update_edges(snapshot.edges, valid)

    return CognitiveFabricSnapshot(
        generation=snapshot.generation + 1,
        transforms=snapshot.transforms,
        candidates=snapshot.candidates,
        selectors=updated_selectors,
        edges=updated_edges,
        fallback_precision=snapshot.fallback_precision,
        created_ns=created_ns,
    )


class EvidenceDisposition(str, Enum):
    QUEUED = "queued"
    DROPPED = "dropped"
    REJECTED = "rejected"
    APPLIED = "applied"


@dataclass(frozen=True, slots=True)
class EvidenceOccurrence:
    occurrence_id: int
    generation: int
    snapshot_fingerprint: str
    disposition: EvidenceDisposition
    created_ns: int


@dataclass(frozen=True, slots=True)
class CognitiveAppeal:
    original_decision: DecisionDecomposition
    reconsidered_decision: DecisionDecomposition
    winner_changed: bool
    posterior_delta_l1: float
    original_alignment: float
    reconsidered_alignment: float
    elapsed_ns: int


@dataclass(frozen=True, slots=True)
class CognitiveOperatingProfile:
    resolve_actions: int
    appeal_actions: int
    observations_created: int
    observations_queued: int
    observations_applied: int
    observations_dropped: int
    observations_rejected: int
    generations_published: int
    resolve_mean_ns: float
    resolve_p95_ns: float
    resolve_p99_ns: float
    appeal_mean_ns: float
    appeal_p95_ns: float
    appeal_p99_ns: float
    appeal_revisions: int
    deterministic_replay_checks: int
    deterministic_replay_agreements: int
    accounting_invariant_holds: bool

    def as_list(self) -> tuple[str, ...]:
        return (
            f"Resolve actions executed: {self.resolve_actions}",
            f"Cognitive appeals executed: {self.appeal_actions}",
            f"Evidence occurrences created: {self.observations_created}",
            f"Evidence currently queued: {self.observations_queued}",
            f"Evidence applied: {self.observations_applied}",
            f"Evidence dropped: {self.observations_dropped}",
            f"Evidence rejected: {self.observations_rejected}",
            f"Generations published: {self.generations_published}",
            f"Resolve mean latency (ns): {self.resolve_mean_ns:.1f}",
            f"Resolve P95 latency (ns): {self.resolve_p95_ns:.1f}",
            f"Resolve P99 latency (ns): {self.resolve_p99_ns:.1f}",
            f"Appeal mean latency (ns): {self.appeal_mean_ns:.1f}",
            f"Appeal P95 latency (ns): {self.appeal_p95_ns:.1f}",
            f"Appeal P99 latency (ns): {self.appeal_p99_ns:.1f}",
            f"Appeal winner revisions: {self.appeal_revisions}",
            f"Deterministic replay agreement: {self.deterministic_replay_agreements}/{self.deterministic_replay_checks}",
            f"Evidence accounting invariant: {'PASS' if self.accounting_invariant_holds else 'FAIL'}",
        )


def _profile_percentile(samples: Sequence[int], percentile: int) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return float(samples[0])
    return float(quantiles(samples, n=100, method="inclusive")[percentile - 1])


class CognitiveFabric:
    """Concurrent Cognitive Fabric register with Phase-1 evidence instrumentation.

    Instrumentation is observational: it does not participate in selector evaluation,
    coalition choice, outcome validity, learning, or publication authority.
    """

    def __init__(
        self,
        snapshot: CognitiveFabricSnapshot,
        *,
        outcome_capacity: int = 4096,
        history_capacity: int = 100_000,
    ) -> None:
        if outcome_capacity <= 0 or history_capacity <= 0:
            raise ValueError("Outcome and history capacities must be positive.")
        self._snapshot = snapshot
        self._outcomes: Deque[tuple[int, Outcome]] = deque(maxlen=outcome_capacity)
        self._outcome_lock = Lock()
        self._publish_lock = Lock()
        self._metrics_lock = Lock()
        self._dropped_outcomes = 0
        self._rejected_outcomes = 0
        self._applied_outcomes = 0
        self._created_outcomes = 0
        self._published_generations = 0
        self._next_occurrence_id = 1
        self._occurrences: Deque[EvidenceOccurrence] = deque(maxlen=history_capacity)
        self._resolve_times_ns: Deque[int] = deque(maxlen=history_capacity)
        self._appeal_times_ns: Deque[int] = deque(maxlen=history_capacity)
        self._appeal_revisions = 0
        self._replay_checks = 0
        self._replay_agreements = 0

    @property
    def snapshot(self) -> CognitiveFabricSnapshot:
        with self._publish_lock:
            return self._snapshot

    @property
    def dropped_outcomes(self) -> int:
        with self._outcome_lock:
            return self._dropped_outcomes

    @property
    def rejected_outcomes(self) -> int:
        with self._outcome_lock:
            return self._rejected_outcomes

    def resolve(
        self,
        input_vector: Sequence[float],
        *,
        exploration_seed: int | None = None,
        evaluation_ns: int | None = None,
    ) -> DecisionDecomposition:
        started = perf_counter_ns()
        with self._publish_lock:
            snapshot = self._snapshot
        decision = evaluate_snapshot(
            snapshot, input_vector, exploration_seed=exploration_seed,
            evaluation_ns=evaluation_ns,
        )
        elapsed = perf_counter_ns() - started
        with self._metrics_lock:
            self._resolve_times_ns.append(elapsed)
        return decision

    def appeal(
        self,
        decision: DecisionDecomposition,
        *,
        exploration_seed: int | None = None,
        evaluation_ns: int | None = None,
    ) -> CognitiveAppeal:
        """Reconsider a completed decision against the current immutable snapshot.

        Appeal is read-only and does not override publication or learning. It provides a
        measured comparison suitable for stress testing and audit.
        """
        started = perf_counter_ns()
        reconsidered = self.resolve(
            decision.input_vector, exploration_seed=exploration_seed,
            evaluation_ns=evaluation_ns,
        )
        old = {item.candidate_id: item.posterior for item in decision.candidate_posteriors}
        new = {item.candidate_id: item.posterior for item in reconsidered.candidate_posteriors}
        candidate_ids = set(old) | set(new)
        delta = fsum(abs(old.get(cid, 0.0) - new.get(cid, 0.0)) for cid in candidate_ids)
        changed = decision.selected_candidate_id != reconsidered.selected_candidate_id
        elapsed = perf_counter_ns() - started
        with self._metrics_lock:
            self._appeal_times_ns.append(elapsed)
            if changed:
                self._appeal_revisions += 1
        return CognitiveAppeal(
            decision, reconsidered, changed, delta, decision.posterior_alignment.alignment_index,
            reconsidered.posterior_alignment.alignment_index, elapsed,
        )

    def check_deterministic_replay(
        self, input_vector: Sequence[float], *, exploration_seed: int = 0, evaluation_ns: int = 0
    ) -> bool:
        with self._publish_lock:
            snapshot = self._snapshot
        first = evaluate_snapshot(snapshot, input_vector, exploration_seed=exploration_seed, evaluation_ns=evaluation_ns)
        second = evaluate_snapshot(snapshot, input_vector, exploration_seed=exploration_seed, evaluation_ns=evaluation_ns)
        agrees = first == second
        with self._metrics_lock:
            self._replay_checks += 1
            self._replay_agreements += int(agrees)
        return agrees

    def observe(
        self,
        decision: DecisionDecomposition,
        *,
        reward: float,
        succeeded: bool,
        latency_ns: int,
    ) -> None:
        outcome = Outcome(
            generation=decision.generation, input_vector=decision.input_vector,
            selected_candidate_id=decision.selected_candidate_id, reward=clamp(reward, 0.0, 1.0),
            succeeded=succeeded, latency_ns=max(0, latency_ns), observed_ns=monotonic_ns(),
            decomposition=decision, formal_model_id=decision.formal_model_id,
            snapshot_fingerprint=decision.snapshot_fingerprint,
        )
        with self._outcome_lock:
            occurrence_id = self._next_occurrence_id
            self._next_occurrence_id += 1
            self._created_outcomes += 1
            if len(self._outcomes) == self._outcomes.maxlen:
                evicted_id, evicted = self._outcomes[0]
                self._dropped_outcomes += 1
                self._occurrences.append(EvidenceOccurrence(
                    evicted_id, evicted.generation, evicted.snapshot_fingerprint,
                    EvidenceDisposition.DROPPED, monotonic_ns(),
                ))
            self._outcomes.append((occurrence_id, outcome))

    def publish_generation(self, *, commit_ns: int | None = None) -> CognitiveFabricSnapshot:
        with self._publish_lock:
            base_snapshot = self._snapshot
            fingerprint = snapshot_fingerprint(base_snapshot)
            with self._outcome_lock:
                entries = tuple(self._outcomes)
                if not entries:
                    return base_snapshot
                valid_entries = tuple(
                    (oid, outcome) for oid, outcome in entries
                    if outcome_matches_snapshot(outcome, base_snapshot, fingerprint=fingerprint)
                )
                valid_ids = {oid for oid, _ in valid_entries}
                valid = tuple(outcome for _, outcome in valid_entries)
                rejected_count = len(entries) - len(valid_entries)
                successor = apply_outcomes(base_snapshot, valid, commit_ns=commit_ns) if valid else base_snapshot
                self._outcomes.clear()
                self._rejected_outcomes += rejected_count
                self._applied_outcomes += len(valid_entries)
                for oid, outcome in entries:
                    disposition = EvidenceDisposition.APPLIED if oid in valid_ids else EvidenceDisposition.REJECTED
                    self._occurrences.append(EvidenceOccurrence(
                        oid, outcome.generation, outcome.snapshot_fingerprint, disposition, monotonic_ns(),
                    ))
                if successor is not base_snapshot:
                    self._published_generations += 1
                self._snapshot = successor
            return self._snapshot

    def evidence_occurrences(self) -> tuple[EvidenceOccurrence, ...]:
        with self._outcome_lock:
            queued = tuple(
                EvidenceOccurrence(oid, outcome.generation, outcome.snapshot_fingerprint, EvidenceDisposition.QUEUED, outcome.observed_ns)
                for oid, outcome in self._outcomes
            )
            return tuple(self._occurrences) + queued

    def operating_profile(self) -> CognitiveOperatingProfile:
        with self._outcome_lock:
            created = self._created_outcomes
            queued = len(self._outcomes)
            applied = self._applied_outcomes
            dropped = self._dropped_outcomes
            rejected = self._rejected_outcomes
            published = self._published_generations
        with self._metrics_lock:
            resolves = tuple(self._resolve_times_ns)
            appeals = tuple(self._appeal_times_ns)
            revisions = self._appeal_revisions
            checks = self._replay_checks
            agreements = self._replay_agreements
        return CognitiveOperatingProfile(
            len(resolves), len(appeals), created, queued, applied, dropped, rejected, published,
            fsum(resolves) / len(resolves) if resolves else 0.0,
            _profile_percentile(resolves, 95), _profile_percentile(resolves, 99),
            fsum(appeals) / len(appeals) if appeals else 0.0,
            _profile_percentile(appeals, 95), _profile_percentile(appeals, 99),
            revisions, checks, agreements, created == queued + applied + dropped + rejected,
        )


# ============================================================
# Phase 1.3 structural transactions, rollback, histories, and
# multiprocessing-isolated advisory candidate generation
# ============================================================

class InvocationEventKind(str, Enum):
    CALL = "call"
    LINEARIZATION = "linearization"
    RETURN = "return"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class InvocationEvent:
    invocation_id: int
    operation: str
    kind: InvocationEventKind
    timestamp_ns: int
    thread_id: int
    process_id: int
    detail: str = ""


class StructuralRejection(str, Enum):
    STALE_PARENT = "stale_parent"
    PRECONDITION = "precondition"
    PARAMETER_SCHEMA = "parameter_schema"
    SUCCESSOR_VALIDATION = "successor_validation"
    OWNER_ONLY = "owner_only"


@dataclass(frozen=True, slots=True)
class StructuralTransactionRequest:
    operation: StructuralOperation
    selector_ids: tuple[SelectorId, ...]
    parent_generation: int
    parent_fingerprint: str
    parameters: tuple[tuple[str, object], ...] = ()
    preconditions: tuple[str, ...] = ()
    estimated_gain: float = 0.0
    transaction_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "selector_ids", tuple(self.selector_ids))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "preconditions", tuple(self.preconditions))
        if not self.transaction_id:
            payload = repr((self.operation.value, self.selector_ids, self.parent_generation,
                            self.parent_fingerprint, self.parameters, self.preconditions,
                            self.estimated_gain)).encode("utf-8")
            object.__setattr__(self, "transaction_id", sha256(payload).hexdigest()[:24])

    def parameter_map(self) -> dict[str, object]:
        return dict(self.parameters)


@dataclass(frozen=True, slots=True)
class RollbackCertificate:
    transaction_id: str
    retained_parent_fingerprint: str
    committed_successor_fingerprint: str
    digest: str


@dataclass(frozen=True, slots=True)
class StructuralTransactionResult:
    committed: bool
    transaction_id: str
    parent_generation: int
    successor_generation: int
    rejection: StructuralRejection | None
    certificate: RollbackCertificate | None
    estimated_gain: float
    realized_gain: float | None
    elapsed_ns: int


@dataclass(frozen=True, slots=True)
class WorkerCandidate:
    job_id: int
    worker_pid: int
    parent_generation: int
    parent_fingerprint: str
    request: StructuralTransactionRequest
    completed_ns: int


def _phase13_worker_candidate(payload: tuple[int, int, str, str]) -> WorkerCandidate:
    """Pure worker entry point. It cannot access or mutate the owner register."""
    job_id, generation, fingerprint, selector_id = payload
    request = StructuralTransactionRequest(
        operation=StructuralOperation.BROADEN_SELECTOR,
        selector_ids=(selector_id,),
        parent_generation=generation,
        parent_fingerprint=fingerprint,
        parameters=(("factor", 0.92),),
        preconditions=("selectors_exist",),
        estimated_gain=0.02,
        transaction_id=f"worker-{job_id}-{fingerprint[:10]}",
    )
    return WorkerCandidate(job_id, os.getpid(), generation, fingerprint, request, monotonic_ns())


@dataclass(frozen=True, slots=True)
class Phase13OperatingProfile:
    values: tuple[tuple[str, object], ...]

    def as_mapping(self) -> dict[str, object]:
        return dict(self.values)

    def render(self) -> str:
        data = self.as_mapping()
        sections = (
            ("Resolve Action", (
                "Resolve actions", "Sustained throughput", "Mean latency", "P95 latency",
                "P99 latency", "Deterministic replay agreement")),
            ("Cognitive Appeal", (
                "Appeal actions", "Appeal throughput", "Mean appeal latency", "P95 appeal latency",
                "P99 appeal latency", "Winner revisions", "Revision rate")),
            ("Evidence and Publication", (
                "Occurrences created", "Queued", "Applied", "Dropped", "Rejected",
                "Queue saturation", "Publication attempts", "Successful generations",
                "Empty publications", "Accounting invariant")),
            ("Structural Transactions", (
                "Transactions attempted", "Committed", "Transactions rejected", "Commit rate",
                "Transaction throughput", "Mean transaction latency", "P95 transaction latency",
                "P99 transaction latency", "Stale-parent rejections", "Precondition rejections",
                "Parameter-schema rejections", "Successor-validation rejections",
                "Owner-only promotion violations")),
            ("Realized Cognitive Gain", (
                "Estimated-gain samples", "Realized-gain samples", "Mean estimated gain",
                "Mean realized gain", "Minimum realized gain", "Maximum realized gain",
                "Mean prediction error")),
            ("Rollback", (
                "Rollbacks attempted", "Rollbacks completed", "Certificate rejections",
                "Mean rollback latency", "P95 rollback latency",
                "Generation-monotonicity violations", "Parent-retention failures")),
            ("History Discipline", (
                "Completed invocations", "Call events", "Linearization events", "Return events",
                "Raise events", "History-discipline violations")),
            ("Multiprocessing Isolation", (
                "Worker jobs admitted", "Worker jobs completed", "Worker failures",
                "Lease expirations", "Cancellations", "Stale results rejected",
                "Valid candidates promoted", "Non-owner promotion attempts",
                "Worker-process isolation violations")),
            ("Verification", (
                "Total tests", "Passed", "Failed", "Stress duration",
                "External storage/API calls")),
        )
        lines = ["COGNITIVE FABRIC PHASE 1.3 OPERATING PROFILE", ""]
        for title, keys in sections:
            lines.append(title)
            for key in keys:
                lines.append(f"- {key}: {data.get(key, 'n/a')}")
            lines.append("")
        return "\n".join(lines).rstrip()


class Phase13CognitiveFabric(CognitiveFabric):
    """Owner-controlled Phase 1.3 verification register.

    Worker processes may construct immutable transaction candidates only. The owner process
    alone validates and publishes a successor. Structural transactions never execute on the
    evidence hot path and preserve immutable parent snapshots for certified rollback.
    """

    def __init__(self, snapshot: CognitiveFabricSnapshot, *, outcome_capacity: int = 4096,
                 history_capacity: int = 100_000, parent_capacity: int = 128) -> None:
        super().__init__(snapshot, outcome_capacity=outcome_capacity, history_capacity=history_capacity)
        self._owner_pid = os.getpid()
        self._phase_lock = Lock()
        self._invocation_ids = count(1)
        self._history: Deque[InvocationEvent] = deque(maxlen=history_capacity)
        self._publication_attempts = 0
        self._empty_publications = 0
        self._queue_saturations = 0
        self._transaction_times: Deque[int] = deque(maxlen=history_capacity)
        self._transaction_attempts = 0
        self._transaction_commits = 0
        self._transaction_rejections: Counter[StructuralRejection] = Counter()
        self._estimated_gains: Deque[float] = deque(maxlen=history_capacity)
        self._realized_gains: Deque[float] = deque(maxlen=history_capacity)
        self._rollback_times: Deque[int] = deque(maxlen=history_capacity)
        self._rollback_attempts = 0
        self._rollback_completed = 0
        self._certificate_rejections = 0
        self._generation_violations = 0
        self._parent_retention_failures = 0
        self._retained_parents: Deque[tuple[str, CognitiveFabricSnapshot]] = deque(maxlen=parent_capacity)
        self._certificates: dict[str, RollbackCertificate] = {}
        self._worker_jobs_admitted = 0
        self._worker_jobs_completed = 0
        self._worker_failures = 0
        self._lease_expirations = 0
        self._cancellations = 0
        self._stale_results_rejected = 0
        self._valid_candidates_promoted = 0
        self._non_owner_promotion_attempts = 0
        self._worker_isolation_violations = 0

    def _event(self, invocation_id: int, operation: str, kind: InvocationEventKind,
               detail: str = "") -> None:
        with self._phase_lock:
            self._history.append(InvocationEvent(invocation_id, operation, kind, monotonic_ns(),
                                                 get_ident(), os.getpid(), detail))

    def _invoke(self, operation: str, linearization: Callable[[], Any]) -> Any:
        invocation_id = next(self._invocation_ids)
        self._event(invocation_id, operation, InvocationEventKind.CALL)
        try:
            result = linearization()
            self._event(invocation_id, operation, InvocationEventKind.LINEARIZATION)
            self._event(invocation_id, operation, InvocationEventKind.RETURN)
            return result
        except BaseException as exc:
            self._event(invocation_id, operation, InvocationEventKind.RAISE, type(exc).__name__)
            raise

    def resolve(self, input_vector: Sequence[float], *, exploration_seed: int | None = None,
                evaluation_ns: int | None = None) -> DecisionDecomposition:
        return self._invoke("resolve", lambda: super(Phase13CognitiveFabric, self).resolve(
            input_vector, exploration_seed=exploration_seed, evaluation_ns=evaluation_ns))

    def appeal(self, decision: DecisionDecomposition, *, exploration_seed: int | None = None,
               evaluation_ns: int | None = None) -> CognitiveAppeal:
        return self._invoke("appeal", lambda: super(Phase13CognitiveFabric, self).appeal(
            decision, exploration_seed=exploration_seed, evaluation_ns=evaluation_ns))

    def observe(self, decision: DecisionDecomposition, *, reward: float, succeeded: bool,
                latency_ns: int) -> None:
        def action() -> None:
            with self._outcome_lock:
                full = len(self._outcomes) == self._outcomes.maxlen
            if full:
                with self._phase_lock:
                    self._queue_saturations += 1
            super(Phase13CognitiveFabric, self).observe(
                decision, reward=reward, succeeded=succeeded, latency_ns=latency_ns)
        return self._invoke("observe", action)

    def publish_generation(self, *, commit_ns: int | None = None) -> CognitiveFabricSnapshot:
        def action() -> CognitiveFabricSnapshot:
            with self._phase_lock:
                self._publication_attempts += 1
            with self._outcome_lock:
                empty = not self._outcomes
            if empty:
                with self._phase_lock:
                    self._empty_publications += 1
            return super(Phase13CognitiveFabric, self).publish_generation(commit_ns=commit_ns)
        return self._invoke("publish_generation", action)

    @staticmethod
    def _alignment_mean(snapshot: CognitiveFabricSnapshot, vectors: Sequence[Vector]) -> float:
        if not vectors:
            return 0.0
        return fsum(evaluate_snapshot(snapshot, vector, exploration_seed=0, evaluation_ns=0)
                    .posterior_alignment.alignment_index for vector in vectors) / len(vectors)

    def _reject_transaction(self, request: StructuralTransactionRequest,
                            reason: StructuralRejection, started: int) -> StructuralTransactionResult:
        elapsed = perf_counter_ns() - started
        with self._phase_lock:
            self._transaction_rejections[reason] += 1
            self._transaction_times.append(elapsed)
        return StructuralTransactionResult(False, request.transaction_id, request.parent_generation,
                                           self._snapshot.generation, reason, None,
                                           request.estimated_gain, None, elapsed)

    def transact(self, request: StructuralTransactionRequest, *, actor_pid: int | None = None,
                 evaluation_vectors: Sequence[Vector] = ()) -> StructuralTransactionResult:
        actor = os.getpid() if actor_pid is None else actor_pid
        started = perf_counter_ns()
        with self._phase_lock:
            self._transaction_attempts += 1
        if actor != self._owner_pid:
            with self._phase_lock:
                self._non_owner_promotion_attempts += 1
            return self._reject_transaction(request, StructuralRejection.OWNER_ONLY, started)
        with self._publish_lock:
            parent = self._snapshot
            parent_fp = snapshot_fingerprint(parent)
            if request.parent_generation != parent.generation or request.parent_fingerprint != parent_fp:
                return self._reject_transaction(request, StructuralRejection.STALE_PARENT, started)
            selector_map = {selector.selector_id: selector for selector in parent.selectors}
            if not request.selector_ids or any(sid not in selector_map for sid in request.selector_ids):
                return self._reject_transaction(request, StructuralRejection.PRECONDITION, started)
            params = request.parameter_map()
            try:
                if request.operation is StructuralOperation.BROADEN_SELECTOR:
                    factor = float(params.get("factor", 0.95))
                    if not 0.0 < factor <= 1.0:
                        raise ValueError("factor")
                    selectors = tuple(replace(selector, precision=selector.precision.broaden(factor))
                                      if selector.selector_id in request.selector_ids else selector
                                      for selector in parent.selectors)
                    edges = parent.edges
                elif request.operation is StructuralOperation.CREATE_SPECIALIST_SELECTOR:
                    source = selector_map[request.selector_ids[0]]
                    new_id = params.get("new_selector_id")
                    if not isinstance(new_id, str) or not new_id or new_id in selector_map:
                        raise ValueError("new_selector_id")
                    specialist = replace(source, selector_id=new_id,
                                         reliability=clamp(float(params.get("reliability", source.reliability)), 0.0, 1.0))
                    selectors = parent.selectors + (specialist,)
                    edges = parent.edges + tuple(
                        SelectorEdge(new_id, selector.selector_id) for selector in parent.selectors
                    ) + tuple(
                        SelectorEdge(selector.selector_id, new_id) for selector in parent.selectors
                    )
                elif request.operation is StructuralOperation.DISCONNECT_REDUNDANT_SELECTORS:
                    selected = set(request.selector_ids)
                    selectors = parent.selectors
                    edges = tuple(edge for edge in parent.edges
                                  if not (edge.source_id in selected and edge.target_id in selected))
                elif request.operation is StructuralOperation.SHADOW_ALTERNATIVE_COALITION:
                    source = selector_map[request.selector_ids[0]]
                    new_id = params.get("shadow_selector_id")
                    if not isinstance(new_id, str) or not new_id or new_id in selector_map:
                        raise ValueError("shadow_selector_id")
                    shadow = replace(source, selector_id=new_id, enabled=False)
                    selectors = parent.selectors + (shadow,)
                    edges = parent.edges
                else:
                    raise ValueError("operation")
            except (TypeError, ValueError):
                return self._reject_transaction(request, StructuralRejection.PARAMETER_SCHEMA, started)
            before = self._alignment_mean(parent, evaluation_vectors)
            if params.get("verification_invalid_successor") is True:
                # Verification-only fault injection: exercise the successor validator without
                # weakening normal parameter or precondition checks.
                selectors = selectors + (selectors[0],)
            try:
                successor = CognitiveFabricSnapshot(
                    generation=parent.generation + 1, transforms=parent.transforms,
                    candidates=parent.candidates, selectors=selectors, edges=edges,
                    fallback_precision=parent.fallback_precision, created_ns=monotonic_ns())
                validate_snapshot(successor)
            except Exception:
                return self._reject_transaction(request, StructuralRejection.SUCCESSOR_VALIDATION, started)
            successor_fp = snapshot_fingerprint(successor)
            digest = sha256((request.transaction_id + parent_fp + successor_fp).encode("utf-8")).hexdigest()
            certificate = RollbackCertificate(request.transaction_id, parent_fp, successor_fp, digest)
            self._retained_parents.append((parent_fp, parent))
            self._certificates[request.transaction_id] = certificate
            self._snapshot = successor
            after = self._alignment_mean(successor, evaluation_vectors)
        elapsed = perf_counter_ns() - started
        realized = after - before if evaluation_vectors else None
        with self._phase_lock:
            self._transaction_commits += 1
            self._transaction_times.append(elapsed)
            self._estimated_gains.append(request.estimated_gain)
            if realized is not None:
                self._realized_gains.append(realized)
        return StructuralTransactionResult(True, request.transaction_id, parent.generation,
                                           successor.generation, None, certificate,
                                           request.estimated_gain, realized, elapsed)

    def rollback(self, certificate: RollbackCertificate) -> bool:
        started = perf_counter_ns()
        with self._phase_lock:
            self._rollback_attempts += 1
        with self._publish_lock:
            current = self._snapshot
            known = self._certificates.get(certificate.transaction_id)
            expected = sha256((certificate.transaction_id + certificate.retained_parent_fingerprint +
                               certificate.committed_successor_fingerprint).encode("utf-8")).hexdigest()
            if known != certificate or certificate.digest != expected or \
                    snapshot_fingerprint(current) != certificate.committed_successor_fingerprint:
                with self._phase_lock:
                    self._certificate_rejections += 1
                    self._rollback_times.append(perf_counter_ns() - started)
                return False
            parent = next((snapshot for fingerprint, snapshot in reversed(self._retained_parents)
                           if fingerprint == certificate.retained_parent_fingerprint), None)
            if parent is None:
                with self._phase_lock:
                    self._parent_retention_failures += 1
                    self._rollback_times.append(perf_counter_ns() - started)
                return False
            restored = replace(parent, generation=current.generation + 1, created_ns=monotonic_ns())
            if restored.generation <= current.generation:
                with self._phase_lock:
                    self._generation_violations += 1
                return False
            validate_snapshot(restored)
            self._snapshot = restored
        with self._phase_lock:
            self._rollback_completed += 1
            self._rollback_times.append(perf_counter_ns() - started)
        return True

    def run_worker_candidates(self, selector_id: str, *, jobs: int = 2,
                              promote: bool = True) -> tuple[WorkerCandidate, ...]:
        if jobs <= 0:
            return ()
        with self._publish_lock:
            generation = self._snapshot.generation
            fingerprint = snapshot_fingerprint(self._snapshot)
        payloads = [(index, generation, fingerprint, selector_id) for index in range(jobs)]
        with self._phase_lock:
            self._worker_jobs_admitted += jobs
        completed: list[WorkerCandidate] = []
        try:
            context = mp.get_context("spawn")
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(jobs, 2), mp_context=context) as executor:
                futures = [executor.submit(_phase13_worker_candidate, payload) for payload in payloads]
                for future in futures:
                    try:
                        candidate = future.result(timeout=30.0)
                        completed.append(candidate)
                        with self._phase_lock:
                            self._worker_jobs_completed += 1
                            if candidate.worker_pid == self._owner_pid:
                                self._worker_isolation_violations += 1
                    except Exception:
                        with self._phase_lock:
                            self._worker_failures += 1
        except Exception:
            with self._phase_lock:
                self._worker_failures += jobs - len(completed)
        if promote:
            for candidate in completed:
                with self._publish_lock:
                    current_generation = self._snapshot.generation
                    current_fp = snapshot_fingerprint(self._snapshot)
                if (candidate.parent_generation != current_generation or
                        candidate.parent_fingerprint != current_fp):
                    with self._phase_lock:
                        self._stale_results_rejected += 1
                    continue
                result = self.transact(candidate.request)
                if result.committed:
                    with self._phase_lock:
                        self._valid_candidates_promoted += 1
        return tuple(completed)

    def invocation_history(self) -> tuple[InvocationEvent, ...]:
        with self._phase_lock:
            return tuple(self._history)

    def history_discipline(self) -> tuple[int, Counter[InvocationEventKind], int]:
        history = self.invocation_history()
        counts = Counter(event.kind for event in history)
        grouped: dict[int, list[InvocationEventKind]] = {}
        for event in history:
            grouped.setdefault(event.invocation_id, []).append(event.kind)
        violations = 0
        completed = 0
        for sequence in grouped.values():
            if sequence and sequence[0] is not InvocationEventKind.CALL:
                violations += 1
            terminals = sequence.count(InvocationEventKind.RETURN) + sequence.count(InvocationEventKind.RAISE)
            if terminals == 1:
                completed += 1
            else:
                violations += 1
            if InvocationEventKind.RETURN in sequence and InvocationEventKind.LINEARIZATION not in sequence:
                violations += 1
            if sequence.count(InvocationEventKind.LINEARIZATION) > 1:
                violations += 1
        return completed, counts, violations

    def phase13_profile(self, *, stress_duration_ns: int = 0, total_tests: int = 0,
                        passed: int = 0, failed: int = 0) -> Phase13OperatingProfile:
        base = self.operating_profile()
        with self._phase_lock:
            tx_times = tuple(self._transaction_times)
            rollback_times = tuple(self._rollback_times)
            estimated = tuple(self._estimated_gains)
            realized = tuple(self._realized_gains)
            transaction_attempts = self._transaction_attempts
            transaction_commits = self._transaction_commits
            rejection_counts = Counter(self._transaction_rejections)
            publication_attempts = self._publication_attempts
            empty_publications = self._empty_publications
            queue_saturations = self._queue_saturations
            rollback_attempts = self._rollback_attempts
            rollback_completed = self._rollback_completed
            certificate_rejections = self._certificate_rejections
            generation_violations = self._generation_violations
            retention_failures = self._parent_retention_failures
            worker = (self._worker_jobs_admitted, self._worker_jobs_completed, self._worker_failures,
                      self._lease_expirations, self._cancellations, self._stale_results_rejected,
                      self._valid_candidates_promoted, self._non_owner_promotion_attempts,
                      self._worker_isolation_violations)
        completed, history_counts, history_violations = self.history_discipline()
        resolve_total_ns = fsum(self._resolve_times_ns)
        appeal_total_ns = fsum(self._appeal_times_ns)
        prediction_errors = tuple(abs(e - r) for e, r in zip(estimated, realized))
        values = (
            ("Resolve actions", base.resolve_actions),
            ("Sustained throughput", f"{base.resolve_actions / (resolve_total_ns / 1e9):.2f} actions/s" if resolve_total_ns else "0.00 actions/s"),
            ("Mean latency", f"{base.resolve_mean_ns / 1e6:.3f} ms"),
            ("P95 latency", f"{base.resolve_p95_ns / 1e6:.3f} ms"),
            ("P99 latency", f"{base.resolve_p99_ns / 1e6:.3f} ms"),
            ("Deterministic replay agreement", f"{base.deterministic_replay_agreements}/{base.deterministic_replay_checks}"),
            ("Appeal actions", base.appeal_actions),
            ("Appeal throughput", f"{base.appeal_actions / (appeal_total_ns / 1e9):.2f} appeals/s" if appeal_total_ns else "0.00 appeals/s"),
            ("Mean appeal latency", f"{base.appeal_mean_ns / 1e6:.3f} ms"),
            ("P95 appeal latency", f"{base.appeal_p95_ns / 1e6:.3f} ms"),
            ("P99 appeal latency", f"{base.appeal_p99_ns / 1e6:.3f} ms"),
            ("Winner revisions", base.appeal_revisions),
            ("Revision rate", f"{(base.appeal_revisions / base.appeal_actions * 100.0) if base.appeal_actions else 0.0:.2f}%"),
            ("Occurrences created", base.observations_created), ("Queued", base.observations_queued),
            ("Applied", base.observations_applied), ("Dropped", base.observations_dropped),
            ("Rejected", base.observations_rejected), ("Queue saturation", queue_saturations),
            ("Publication attempts", publication_attempts), ("Successful generations", base.generations_published),
            ("Empty publications", empty_publications),
            ("Accounting invariant", "PASS" if base.accounting_invariant_holds else "FAIL"),
            ("Transactions attempted", transaction_attempts), ("Committed", transaction_commits),
            ("Transactions rejected", transaction_attempts - transaction_commits),
            ("Commit rate", f"{(transaction_commits / transaction_attempts * 100.0) if transaction_attempts else 0.0:.2f}%"),
            ("Transaction throughput", f"{transaction_attempts / (fsum(tx_times) / 1e9):.2f} tx/s" if tx_times and fsum(tx_times) else "0.00 tx/s"),
            ("Mean transaction latency", f"{(fsum(tx_times) / len(tx_times) / 1e6) if tx_times else 0.0:.3f} ms"),
            ("P95 transaction latency", f"{_profile_percentile(tx_times, 95) / 1e6:.3f} ms"),
            ("P99 transaction latency", f"{_profile_percentile(tx_times, 99) / 1e6:.3f} ms"),
            ("Stale-parent rejections", rejection_counts[StructuralRejection.STALE_PARENT]),
            ("Precondition rejections", rejection_counts[StructuralRejection.PRECONDITION]),
            ("Parameter-schema rejections", rejection_counts[StructuralRejection.PARAMETER_SCHEMA]),
            ("Successor-validation rejections", rejection_counts[StructuralRejection.SUCCESSOR_VALIDATION]),
            ("Owner-only promotion violations", rejection_counts[StructuralRejection.OWNER_ONLY]),
            ("Estimated-gain samples", len(estimated)), ("Realized-gain samples", len(realized)),
            ("Mean estimated gain", f"{(fsum(estimated) / len(estimated)) if estimated else 0.0:.6f}"),
            ("Mean realized gain", f"{(fsum(realized) / len(realized)) if realized else 0.0:.6f}"),
            ("Minimum realized gain", f"{min(realized) if realized else 0.0:.6f}"),
            ("Maximum realized gain", f"{max(realized) if realized else 0.0:.6f}"),
            ("Mean prediction error", f"{(fsum(prediction_errors) / len(prediction_errors)) if prediction_errors else 0.0:.6f}"),
            ("Rollbacks attempted", rollback_attempts), ("Rollbacks completed", rollback_completed),
            ("Certificate rejections", certificate_rejections),
            ("Mean rollback latency", f"{(fsum(rollback_times) / len(rollback_times) / 1e6) if rollback_times else 0.0:.3f} ms"),
            ("P95 rollback latency", f"{_profile_percentile(rollback_times, 95) / 1e6:.3f} ms"),
            ("Generation-monotonicity violations", generation_violations),
            ("Parent-retention failures", retention_failures),
            ("Completed invocations", completed), ("Call events", history_counts[InvocationEventKind.CALL]),
            ("Linearization events", history_counts[InvocationEventKind.LINEARIZATION]),
            ("Return events", history_counts[InvocationEventKind.RETURN]),
            ("Raise events", history_counts[InvocationEventKind.RAISE]),
            ("History-discipline violations", history_violations),
            ("Worker jobs admitted", worker[0]), ("Worker jobs completed", worker[1]),
            ("Worker failures", worker[2]), ("Lease expirations", worker[3]),
            ("Cancellations", worker[4]), ("Stale results rejected", worker[5]),
            ("Valid candidates promoted", worker[6]), ("Non-owner promotion attempts", worker[7]),
            ("Worker-process isolation violations", worker[8]),
            ("Total tests", total_tests), ("Passed", passed), ("Failed", failed),
            ("Stress duration", f"{stress_duration_ns / 1e9:.3f} s"),
            ("External storage/API calls", 0),
        )
        return Phase13OperatingProfile(values)


def build_phase13_example_fabric(*, outcome_capacity: int = 64) -> Phase13CognitiveFabric:
    base = build_example_fabric()
    return Phase13CognitiveFabric(base.snapshot, outcome_capacity=outcome_capacity)


class _VerificationLedger:
    def __init__(self) -> None:
        self.total = 0
        self.passed = 0
        self.failed = 0

    def check(self, condition: bool, label: str) -> None:
        self.total += 1
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            raise AssertionError(label)


def run_phase13_verification(*, resolve_actions: int = 120, appeal_actions: int = 24,
                             worker_jobs: int = 2) -> Phase13OperatingProfile:
    started = perf_counter_ns()
    ledger = _VerificationLedger()
    fabric = build_phase13_example_fabric(outcome_capacity=32)
    workloads: tuple[Vector, ...] = ((-0.8, -0.7, -0.5), (1.0, 0.1, 0.9),
                                    (0.1, 1.3, 0.0), (0.8, 0.8, 0.7))
    decisions: list[DecisionDecomposition] = []
    for index in range(resolve_actions):
        decisions.append(fabric.resolve(workloads[index % len(workloads)], exploration_seed=index % 7,
                                        evaluation_ns=index))
    ledger.check(len(decisions) == resolve_actions, "resolve count")
    for index in range(appeal_actions):
        fabric.appeal(decisions[index % len(decisions)], exploration_seed=index % 7,
                      evaluation_ns=index)
    ledger.check(fabric.operating_profile().appeal_actions == appeal_actions, "appeal count")
    replay_ok = all(fabric.check_deterministic_replay(vector, exploration_seed=9, evaluation_ns=11)
                    for vector in workloads for _ in range(4))
    ledger.check(replay_ok, "deterministic replay")
    for index in range(48):
        decision = decisions[index % len(decisions)]
        reward, succeeded, latency = _demo_reward(decision.selected_candidate_id, decision.input_vector)
        fabric.observe(decision, reward=reward, succeeded=succeeded, latency_ns=latency)
    ledger.check(fabric.operating_profile().observations_dropped == 16, "bounded queue drop accounting")
    fabric.publish_generation()
    fabric.publish_generation()
    base_profile = fabric.operating_profile()
    ledger.check(base_profile.accounting_invariant_holds, "evidence accounting")
    ledger.check(base_profile.observations_applied + base_profile.observations_rejected == 32,
                 "captured queue classification")

    parent = fabric.snapshot
    parent_fp = snapshot_fingerprint(parent)
    request = StructuralTransactionRequest(
        StructuralOperation.BROADEN_SELECTOR, (parent.selectors[0].selector_id,), parent.generation,
        parent_fp, (("factor", 0.90),), ("selectors_exist",), 0.03)
    committed = fabric.transact(request, evaluation_vectors=workloads)
    ledger.check(committed.committed and committed.certificate is not None, "structural commit")
    stale = fabric.transact(request)
    ledger.check(stale.rejection is StructuralRejection.STALE_PARENT, "stale parent rejection")
    current = fabric.snapshot
    current_fp = snapshot_fingerprint(current)
    bad_selector = StructuralTransactionRequest(
        StructuralOperation.BROADEN_SELECTOR, ("missing",), current.generation, current_fp,
        (("factor", 0.9),), (), 0.0)
    ledger.check(fabric.transact(bad_selector).rejection is StructuralRejection.PRECONDITION,
                 "precondition rejection")
    bad_parameter = StructuralTransactionRequest(
        StructuralOperation.BROADEN_SELECTOR, (current.selectors[0].selector_id,), current.generation,
        current_fp, (("factor", 2.0),), (), 0.0)
    ledger.check(fabric.transact(bad_parameter).rejection is StructuralRejection.PARAMETER_SCHEMA,
                 "parameter rejection")
    invalid_successor = StructuralTransactionRequest(
        StructuralOperation.BROADEN_SELECTOR, (current.selectors[0].selector_id,), current.generation,
        current_fp, (("factor", 0.9), ("verification_invalid_successor", True)), (), 0.0)
    ledger.check(fabric.transact(invalid_successor).rejection is StructuralRejection.SUCCESSOR_VALIDATION,
                 "successor validation rejection")
    owner_violation = StructuralTransactionRequest(
        StructuralOperation.BROADEN_SELECTOR, (current.selectors[0].selector_id,), current.generation,
        current_fp, (("factor", 0.9),), (), 0.0)
    ledger.check(fabric.transact(owner_violation, actor_pid=fabric._owner_pid + 1).rejection
                 is StructuralRejection.OWNER_ONLY, "owner-only enforcement")
    assert committed.certificate is not None
    forged = replace(committed.certificate, digest="0" * 64)
    ledger.check(not fabric.rollback(forged), "forged rollback certificate rejection")
    ledger.check(fabric.rollback(committed.certificate), "certified rollback")
    ledger.check(fabric.snapshot.generation > committed.successor_generation, "rollback generation monotonicity")

    worker_results = fabric.run_worker_candidates(fabric.snapshot.selectors[0].selector_id,
                                                   jobs=worker_jobs, promote=True)
    ledger.check(len(worker_results) == worker_jobs, "worker completion")
    ledger.check(all(item.worker_pid != os.getpid() for item in worker_results), "worker isolation")

    completed, counts, violations = fabric.history_discipline()
    ledger.check(violations == 0, "history discipline")
    ledger.check(completed == counts[InvocationEventKind.RETURN] + counts[InvocationEventKind.RAISE],
                 "history completion accounting")
    elapsed = perf_counter_ns() - started
    return fabric.phase13_profile(stress_duration_ns=elapsed, total_tests=ledger.total,
                                  passed=ledger.passed, failed=ledger.failed)


# ============================================================
# Construction helpers
# ============================================================

def planar_rotation(
    dimension: int,
    axis_a: int,
    axis_b: int,
    angle_radians: float,
) -> Matrix:
    if dimension <= 0:
        raise ValueError("Rotation dimension must be positive.")
    if not 0 <= axis_a < dimension or not 0 <= axis_b < dimension:
        raise ValueError("Rotation axis is outside the matrix dimension.")
    if axis_a == axis_b:
        raise ValueError("Rotation axes must be distinct.")
    if not isfinite(angle_radians):
        raise ValueError("Rotation angle must be finite.")
    matrix = [list(row) for row in identity_matrix(dimension)]
    c = cos(angle_radians)
    s = sin(angle_radians)
    matrix[axis_a][axis_a] = c
    matrix[axis_a][axis_b] = -s
    matrix[axis_b][axis_a] = s
    matrix[axis_b][axis_b] = c
    return tuple(tuple(row) for row in matrix)


def open_uniform_knots(
    minimum: float,
    maximum: float,
    *,
    basis_count: int,
    degree: int,
) -> tuple[float, ...]:
    if not isfinite(minimum) or not isfinite(maximum):
        raise ValueError("Spline domain bounds must be finite.")
    if maximum <= minimum:
        raise ValueError("Maximum must exceed minimum.")
    if basis_count <= degree:
        raise ValueError("Basis count must exceed degree.")

    interior_count = basis_count - degree - 1
    knots = [minimum] * (degree + 1)

    for index in range(1, interior_count + 1):
        fraction = index / (interior_count + 1)
        knots.append(minimum + fraction * (maximum - minimum))

    knots.extend([maximum] * (degree + 1))
    return tuple(knots)


def zero_spline(
    dimension: int,
    *,
    minimum: float = -3.0,
    maximum: float = 3.0,
    basis_count: int = 7,
    degree: int = 2,
    bias: float = 0.0,
) -> SplineModel:
    knots = open_uniform_knots(
        minimum,
        maximum,
        basis_count=basis_count,
        degree=degree,
    )
    axes = tuple(SplineAxis(knots=knots, degree=degree) for _ in range(dimension))
    coefficients = tuple(
        tuple(0.0 for _ in range(basis_count))
        for _ in range(dimension)
    )
    return SplineModel(axes=axes, coefficients=coefficients, bias=bias)


def default_precision(dimension: int, diagonal_raw: float = 0.0) -> PrecisionModel:
    raw = tuple(
        tuple(diagonal_raw if row == col else 0.0 for col in range(dimension))
        for row in range(dimension)
    )
    return PrecisionModel(raw_lower=raw)


def build_example_fabric() -> CognitiveFabric:
    dimension = 3

    transforms: tuple[ReversibleTransform, ...] = (
        OrthogonalTransform("identity", identity_matrix(dimension), cost=0.1),
        OrthogonalTransform(
            "rotate_01",
            planar_rotation(dimension, 0, 1, 0.37),
            cost=0.8,
        ),
        OrthogonalTransform(
            "rotate_12",
            planar_rotation(dimension, 1, 2, -0.24),
            cost=0.8,
        ),
        AffineTransform(
            "offset_scale",
            matrix=(
                (1.10, 0.00, 0.00),
                (0.00, 0.90, 0.00),
                (0.00, 0.00, 1.05),
            ),
            offset=(0.05, -0.02, 0.03),
            cost=1.0,
        ),
    )

    candidates = (
        CandidateModel("direct_path", prior=0.38),
        CandidateModel("batched_path", prior=0.34),
        CandidateModel("contention_safe_path", prior=0.28),
    )

    def selector_models(biases: Mapping[str, float]) -> tuple[SelectorCandidateModel, ...]:
        return tuple(
            SelectorCandidateModel(
                candidate_id=candidate.candidate_id,
                spline=zero_spline(
                    dimension,
                    bias=biases.get(candidate.candidate_id, 0.0),
                ),
            )
            for candidate in candidates
        )

    precision = default_precision(dimension)

    selectors = (
        SelectorRegion(
            selector_id="local_direct",
            transform_chain=("identity",),
            center=(-0.8, -0.7, -0.5),
            precision=precision,
            candidate_models=selector_models(
                {
                    "direct_path": 0.35,
                    "batched_path": -0.10,
                    "contention_safe_path": -0.20,
                }
            ),
            reliability=0.62,
            novelty_tolerance=0.45,
        ),
        SelectorRegion(
            selector_id="batch_shape",
            transform_chain=("rotate_01", "offset_scale"),
            center=(0.9, 0.1, 0.8),
            precision=precision,
            candidate_models=selector_models(
                {
                    "direct_path": -0.15,
                    "batched_path": 0.40,
                    "contention_safe_path": -0.05,
                }
            ),
            reliability=0.58,
            novelty_tolerance=0.60,
        ),
        SelectorRegion(
            selector_id="contention_shape",
            transform_chain=("rotate_12",),
            center=(0.1, 1.1, 0.0),
            precision=precision,
            candidate_models=selector_models(
                {
                    "direct_path": -0.20,
                    "batched_path": 0.00,
                    "contention_safe_path": 0.45,
                }
            ),
            reliability=0.60,
            novelty_tolerance=0.55,
        ),
        SelectorRegion(
            selector_id="novelty_observer",
            transform_chain=("rotate_01", "rotate_12"),
            center=(0.0, 0.0, 0.0),
            precision=precision.broaden(0.70),
            candidate_models=selector_models({}),
            reliability=0.45,
            novelty_tolerance=0.90,
            exploration_floor=0.10,
        ),
    )

    edges = tuple(
        SelectorEdge(source.selector_id, target.selector_id)
        for source in selectors
        for target in selectors
        if source.selector_id != target.selector_id
    )

    snapshot = CognitiveFabricSnapshot(
        generation=1,
        transforms=transforms,
        candidates=candidates,
        selectors=selectors,
        edges=edges,
        fallback_precision=default_precision(dimension, diagonal_raw=-0.4),
        created_ns=monotonic_ns(),
    )
    return CognitiveFabric(snapshot)


# ============================================================
# Complete decomposition renderer
# ============================================================

def render_decision(decision: DecisionDecomposition) -> str:
    lines = [
        f"formal_model={decision.formal_model_id}",
        f"generation={decision.generation}",
        f"snapshot_fingerprint={decision.snapshot_fingerprint}",
        f"evaluated_ns={decision.evaluated_ns}",
        f"exploration_seed={decision.exploration_seed}",
        f"input={decision.input_vector}",
        f"selected_candidate={decision.selected_candidate_id}",
        f"novelty_score={decision.novelty_score:.8f}",
        "",
        "POSTERIOR ALIGNMENT",
        f"  winning_posterior={decision.posterior_alignment.winning_posterior:.8f}",
        f"  selector_agreement={decision.posterior_alignment.selector_agreement:.8f}",
        f"  independent_support={decision.posterior_alignment.independent_support:.8f}",
        f"  redundancy_penalty={decision.posterior_alignment.redundancy_penalty:.8f}",
        f"  dissent_mass={decision.posterior_alignment.dissent_mass:.8f}",
        f"  uncertainty_mass={decision.posterior_alignment.uncertainty_mass:.8f}",
        f"  novelty_mass={decision.posterior_alignment.novelty_mass:.8f}",
        f"  alignment_index={decision.posterior_alignment.alignment_index:.8f}",
        "",
        "COALITION",
        f"  selectors={decision.coalition.selector_ids}",
        f"  coverage={decision.coalition.coverage:.8f}",
        f"  diversity={decision.coalition.diversity:.8f}",
        f"  historical_utility={decision.coalition.historical_utility:.8f}",
        f"  redundancy={decision.coalition.redundancy:.8f}",
        f"  conflict_cost={decision.coalition.conflict_cost:.8f}",
        f"  complexity_cost={decision.coalition.complexity_cost:.8f}",
        f"  score={decision.coalition.score:.8f}",
        "",
        "CANDIDATE POSTERIORS",
    ]

    for candidate in decision.candidate_posteriors:
        lines.extend(
            [
                f"  candidate={candidate.candidate_id}",
                f"    prior_log_probability={candidate.prior_log_probability:.8f}",
                f"    selector_log_evidence={candidate.selector_log_evidence}",
                f"    support_total={candidate.support_total:.8f}",
                f"    opposition_total={candidate.opposition_total:.8f}",
                f"    uncertainty_total={candidate.uncertainty_total:.8f}",
                f"    posterior={candidate.posterior:.8f}",
            ]
        )

    lines.append("")
    lines.append("SELECTOR DECOMPOSITIONS")

    for vote in decision.selector_votes:
        a = vote.activation
        lines.extend(
            [
                f"  selector={vote.selector_id}",
                f"    evaluation_reason={vote.evaluation_reason}",
                f"    transform_chain={vote.transform_trace.transform_ids}",
                f"    transform_cost={vote.transform_trace.total_complexity_cost:.8f}",
                f"    transformed={vote.transform_trace.final_vector}",
                f"    chain_reconstructed={vote.transform_trace.reconstructed_initial_vector}",
                f"    chain_reconstruction_error={vote.transform_trace.chain_reconstruction_error:.12e}",
                f"    mahalanobis_squared={a.mahalanobis_squared:.8f}",
                f"    log_activation={a.log_activation:.8f}",
                f"    routing_activation={a.routing_activation:.8f}",
                f"    learning_activation={a.learning_activation:.8f}",
                f"    exploration_probability={a.exploration_probability:.8f}",
                f"    precision_min_eigen={a.precision_diagnostics.minimum_eigenvalue:.8e}",
                f"    precision_max_eigen={a.precision_diagnostics.maximum_eigenvalue:.8e}",
                f"    precision_condition={a.precision_diagnostics.condition_number:.8e}",
                f"    precision_log_det={a.precision_diagnostics.log_determinant:.8f}",
                f"    precision_healthy={a.precision_diagnostics.healthy}",
                f"    novelty={vote.novelty:.8f}",
                f"    expected_information_gain={vote.expected_information_gain:.8f}",
            ]
        )

        for step in vote.transform_trace.steps:
            lines.extend(
                [
                    f"    transform={step.transform_id}",
                    f"      input={step.input_vector}",
                    f"      output={step.output_vector}",
                    f"      reconstructed={step.reconstructed_vector}",
                    f"      reconstruction_error={step.reconstruction_error:.12e}",
                ]
            )

        for score in vote.candidate_scores:
            lines.extend(
                [
                    f"    candidate={score.candidate_id}",
                    f"      local_posterior={score.local_posterior:.8f}",
                    f"      support={score.support:.8f}",
                    f"      opposition={score.opposition:.8f}",
                    f"      uncertainty={score.uncertainty:.8f}",
                    f"      raw_product={score.raw_product:.12e}",
                    f"      log_score={score.log_score:.8f}",
                ]
            )
            for name, value in score.components.as_mapping().items():
                lines.append(f"      component.{name}={value:.8f}")

            lines.append(f"      spline.bias={score.spline.bias:.8f}")
            lines.append(f"      spline.raw_score={score.spline.raw_score:.8f}")
            lines.append(f"      spline.probability={score.spline.probability:.8f}")

            for dimension in score.spline.dimensions:
                lines.extend(
                    [
                        f"      spline.dimension={dimension.dimension}",
                        f"        value={dimension.value:.8f}",
                        f"        basis_values={dimension.basis_values}",
                        f"        coefficients={dimension.coefficients}",
                        f"        contributions={dimension.contributions}",
                        f"        subtotal={dimension.subtotal:.8f}",
                    ]
                )

    lines.append("")
    lines.append("CONNECTIVITY")
    for trace in decision.coalition.connectivity:
        lines.extend(
            [
                f"  {trace.source_id}->{trace.target_id}",
                f"    compatibility={trace.compatibility:.8f}",
                f"    information_gain={trace.information_gain:.8f}",
                f"    redundancy={trace.redundancy:.8f}",
                f"    conflict_rate={trace.conflict_rate:.8f}",
                f"    effective_weight={trace.effective_weight:.8f}",
            ]
        )

    lines.append("")
    lines.append("STRUCTURAL PROPOSALS")
    if not decision.structural_proposals:
        lines.append("  none")
    else:
        for proposal in decision.structural_proposals:
            lines.extend(
                [
                    f"  operation={proposal.operation.value}",
                    f"    selectors={proposal.selector_ids}",
                    f"    advisory_only={proposal.advisory_only}",
                    f"    evidence={proposal.evidence}",
                    f"    preconditions={proposal.preconditions}",
                    f"    rationale={proposal.rationale}",
                    f"    estimated_gain={proposal.estimated_gain:.8f}",
                    f"    complexity_cost={proposal.complexity_cost:.8f}",
                ]
            )

    return "\n".join(lines)


def _demo_reward(candidate_id: CandidateId, vector: Vector) -> tuple[float, bool, int]:
    item_count, contention, payload = vector

    if candidate_id == "direct_path":
        reward = 0.90 - 0.20 * max(item_count, 0.0) - 0.25 * max(contention, 0.0)
    elif candidate_id == "batched_path":
        reward = 0.52 + 0.24 * max(item_count, 0.0) + 0.16 * max(payload, 0.0)
    elif candidate_id == "contention_safe_path":
        reward = 0.46 + 0.34 * max(contention, 0.0)
    else:
        raise KeyError(candidate_id)

    reward = clamp(reward, 0.0, 1.0)
    return reward, reward >= 0.35, int((1.1 - reward) * 1_000_000)


def demonstration() -> None:
    fabric = build_example_fabric()
    workloads: tuple[Vector, ...] = (
        (-0.8, -0.7, -0.5),
        (1.0, 0.1, 0.9),
        (0.1, 1.3, 0.0),
        (0.8, 0.8, 0.7),
    )

    for cycle in range(8):
        for workload in workloads:
            decision = fabric.resolve(
                workload,
                exploration_seed=cycle,
            )
            reward, succeeded, latency_ns = _demo_reward(
                decision.selected_candidate_id,
                workload,
            )
            fabric.observe(
                decision,
                reward=reward,
                succeeded=succeeded,
                latency_ns=latency_ns,
            )
        fabric.publish_generation()

    decision = fabric.resolve((0.25, 1.25, 0.10), exploration_seed=99)
    print(render_decision(decision))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cognitive Fabric reference realization")
    parser.add_argument("--phase-1.3-profile", dest="phase_1_3_profile", action="store_true",
                        help="run the Phase 1.3 operating-profile verification structure")
    parser.add_argument("--resolve-actions", type=int, default=120)
    parser.add_argument("--appeal-actions", type=int, default=24)
    parser.add_argument("--worker-jobs", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.phase_1_3_profile:
        print(run_phase13_verification(resolve_actions=arguments.resolve_actions,
                                       appeal_actions=arguments.appeal_actions,
                                       worker_jobs=arguments.worker_jobs).render())
    else:
        demonstration()
