------------------ MODULE CognitiveFabricPublicationProof ------------------
\* CF-PUB-1.0 proof module.
\*
\* The rich operational specification remains in CognitiveFabricPublication.
\* This module contains only theorem statements and TLAPS proof scripts.

EXTENDS CognitiveFabricPublication,
        FiniteSetTheorems,
        SequenceTheorems,
        TLAPS

LEMMA IdsEqualsRange ==
    ASSUME NEW S,
           NEW s \in Seq(S)
    PROVE  Ids(s) = Range(s)
<1>1. QED BY RangeEquality
    DEF Ids

LEMMA IdsAppend ==
    ASSUME NEW S,
           NEW s \in Seq(S),
           NEW x \in S
    PROVE  Ids(Append(s, x)) = Ids(s) \cup {x}
<1>1. QED BY RangeEquality, AppendProperties
    DEF Ids

LEMMA TailPreservesInjectivity ==
    ASSUME NEW S,
           NEW s \in Seq(S),
           s # <<>>,
           SeqInjective(s)
    PROVE  SeqInjective(Tail(s))
<1>1. QED BY HeadTailProperties, SMT
    DEF SeqInjective

LEMMA FreshAppendPreservesInjectivity ==
    ASSUME NEW S,
           NEW s \in Seq(S),
           NEW x \in S,
           SeqInjective(s),
           x \notin Ids(s)
    PROVE  SeqInjective(Append(s, x))
<1>1. /\ Len(Append(s, x)) = Len(s) + 1
       /\ \A i \in 1..Len(s) : Append(s, x)[i] = s[i]
       /\ Append(s, x)[Len(s) + 1] = x
    BY AppendProperties
<1>2. QED BY <1>1, SMT
    DEF SeqInjective, Ids

LEMMA IdsTailOfInjectiveSequence ==
    ASSUME NEW S,
           NEW s \in Seq(S),
           s # <<>>,
           SeqInjective(s)
    PROVE  Ids(Tail(s)) = Ids(s) \ {Head(s)}
<1>1. IsInjective(s)
    BY LenProperties
    DEF SeqInjective, IsInjective
<1>2. /\ Tail(s) \in Seq(S)
       /\ Range(Tail(s)) = Range(s) \ {Head(s)}
    BY <1>1, HeadTailProperties, TailInjectiveSeq
<1>3. /\ Ids(s) = Range(s)
       /\ Ids(Tail(s)) = Range(Tail(s))
    BY <1>2, RangeEquality
    DEF Ids
<1>4. QED BY <1>2, <1>3

LEMMA CreatedIdsAdvance ==
    ASSUME NEW n \in Nat \ {0}
    PROVE  1..((n + 1) - 1) = (1..(n - 1)) \cup {n}
<1>1. QED BY SMT

LEMMA OutcomeIdsFinite ==
    ASSUME ParameterAssumptions
    PROVE  IsFiniteSet(OutcomeIds)
<1>1. QED BY FS_Interval, SMT
    DEF ParameterAssumptions, OutcomeIds

LEMMA SubsetOfOutcomeIdsFinite ==
    ASSUME ParameterAssumptions,
           NEW S \in SUBSET OutcomeIds
    PROVE  IsFiniteSet(S)
<1>1. QED BY OutcomeIdsFinite, FS_Subset

LEMMA ExceptAtAndAway ==
    ASSUME NEW S,
           NEW T,
           NEW f \in [S -> T],
           NEW x \in S,
           NEW y \in T
    PROVE  /\ [f EXCEPT ![x] = y][x] = y
           /\ \A z \in S \ {x} :
                 [f EXCEPT ![x] = y][z] = f[z]
<1>1. QED BY SMT

LEMMA EnqueueBelowCapacity ==
    ASSUME NEW id
    PROVE  Enqueue(id) /\ Len(queue) < QueueCapacity
              => /\ queue' = Append(queue, id)
                 /\ dropped' = dropped
                 /\ droppedCount' = droppedCount
<1>1. QED BY SMT
    DEF Enqueue

LEMMA EnqueueAtCapacity ==
    ASSUME NEW id,
           Len(queue) \in Nat,
           QueueCapacity \in Nat
    PROVE  Enqueue(id) /\ Len(queue) >= QueueCapacity
              => /\ queue' = Append(Tail(queue), id)
                 /\ dropped' = dropped \cup {Head(queue)}
                 /\ droppedCount' = droppedCount + 1
<1>1. QED BY SMT
    DEF Enqueue

LEMMA ValidIdsAtSubset ==
    ASSUME NEW s,
           NEW gen,
           NEW fp
    PROVE  ValidIdsAt(s, gen, fp) \subseteq Ids(s)
<1>1. QED BY SMT
    DEF ValidIdsAt, IsValidAt, Ids

LEMMA ValidIdsAtCharacterization ==
    ASSUME NEW s,
           NEW gen,
           NEW fp,
           NEW id \in Ids(s)
    PROVE  (id \in ValidIdsAt(s, gen, fp))
              = IsValidAt(outcomeById[id], gen, fp)
<1>1. QED BY SMT
    DEF ValidIdsAt, IsValidAt, Ids

THEOREM PublishFailPreservesEvidence ==
    PublishFail
        => UNCHANGED <<generation,
                       fingerprint,
                       queue,
                       nextOutcomeId,
                       outcomeById,
                       dropped,
                       rejected,
                       applied,
                       droppedCount,
                       rejectedCount,
                       appliedCount,
                       updateCount,
                       commitLog>>
<1>1. QED BY DEF PublishFail

THEOREM PublishCommitGenerationStep ==
    ASSUME InductiveSafety,
           PublishCommit
    PROVE  /\ (pendingValid # {})
                 = (generation' = generation + 1)
           /\ (pendingValid = {})
                 = (generation' = generation)
<1>1. QED BY SMT
    DEF PublishCommit, InductiveSafety, Safety, TypeOK

LEMMA InitTypeOK ==
    ASSUME ParameterAssumptions
    PROVE  Init => TypeOK
<1>1. QED BY EmptySeq, SMT
    DEF Init,
        TypeOK,
        OutcomeType,
        CommitEventType,
        ResolverPhases,
        PublisherPhases,
        OutcomeSources,
        OutcomeIds,
        ParameterAssumptions

LEMMA InitBoundedQueue ==
    ASSUME ParameterAssumptions
    PROVE  Init => BoundedQueue
<1>1. QED BY EmptySeq, SMT
    DEF Init, BoundedQueue, ParameterAssumptions

LEMMA InitUniqueOutcomeIds ==
    ASSUME ParameterAssumptions
    PROVE  Init => UniqueOutcomeIds
<1>1. QED BY FS_EmptySet, EmptySeq, SMT
    DEF Init, UniqueOutcomeIds, SeqInjective, Ids

LEMMA InitDispositionDisjoint ==
    ASSUME ParameterAssumptions
    PROVE  Init => DispositionDisjoint
<1>1. QED BY EmptySeq, SMT
    DEF Init, DispositionDisjoint, Ids

LEMMA InitOutcomeAccounting ==
    ASSUME ParameterAssumptions
    PROVE  Init => OutcomeAccounting
<1>1. QED BY EmptySeq, SMT
    DEF Init, OutcomeAccounting, CreatedIds, Ids

LEMMA InitCountAccuracy ==
    ASSUME ParameterAssumptions
    PROVE  Init => CountAccuracy
<1>1. QED BY FS_EmptySet, SMT
    DEF Init, CountAccuracy

LEMMA InitSnapshotGenerationAgreement ==
    ASSUME ParameterAssumptions
    PROVE  Init => SnapshotGenerationAgreement
<1>1. QED BY SMT
    DEF Init, SnapshotGenerationAgreement

LEMMA InitPublisherLockInvariant ==
    ASSUME ParameterAssumptions
    PROVE  Init => PublisherLockInvariant
<1>1. QED BY SMT
    DEF Init, PublisherLockInvariant

LEMMA InitResolutionSnapshotConsistency ==
    ASSUME ParameterAssumptions
    PROVE  Init => ResolutionSnapshotConsistency
<1>1. QED BY SMT
    DEF Init, ResolutionSnapshotConsistency

LEMMA InitCorrectCommitClassification ==
    ASSUME ParameterAssumptions
    PROVE  Init => CorrectCommitClassification
<1>1. QED BY EmptySeq, SMT
    DEF Init, CorrectCommitClassification

LEMMA InitForeignApplicationExclusion ==
    ASSUME ParameterAssumptions
    PROVE  Init => ForeignApplicationExclusion
<1>1. QED BY SMT
    DEF Init, ForeignApplicationExclusion

LEMMA InitInductiveStrengthening ==
    ASSUME ParameterAssumptions
    PROVE  Init => InductiveStrengthening
<1>1. QED BY FS_EmptySet, EmptySeq, SMT
    DEF Init,
        InductiveStrengthening,
        CreatedOutcomeIdentity,
        CreatedCanonicalValidity,
        CreatedIds

THEOREM InitEstablishesInductiveSafety ==
    ASSUME ParameterAssumptions
    PROVE  Init => InductiveSafety
<1>1. QED BY InitTypeOK,
              InitBoundedQueue,
              InitUniqueOutcomeIds,
              InitDispositionDisjoint,
              InitOutcomeAccounting,
              InitCountAccuracy,
              InitSnapshotGenerationAgreement,
              InitPublisherLockInvariant,
              InitResolutionSnapshotConsistency,
              InitCorrectCommitClassification,
              InitForeignApplicationExclusion,
              InitInductiveStrengthening,
              SMT
    DEF InductiveSafety, Safety

PublicationState ==
    <<generation,
      fingerprint,
      queue,
      publisherPhase,
      publisherBaseGen,
      publisherBaseFp,
      batch,
      pendingValid,
      pendingRejected,
      nextOutcomeId,
      outcomeById,
      dropped,
      rejected,
      applied,
      droppedCount,
      rejectedCount,
      appliedCount,
      updateCount,
      failureSeen,
      commitLog>>

LEMMA EqualPublicationStatePreservesInductiveSafety ==
    ASSUME InductiveSafety,
           PublicationState' = PublicationState,
           TypeOK',
           ResolutionSnapshotConsistency'
    PROVE  InductiveSafety'
<1>1. /\ generation' = generation
       /\ fingerprint' = fingerprint
       /\ queue' = queue
       /\ publisherPhase' = publisherPhase
       /\ publisherBaseGen' = publisherBaseGen
       /\ publisherBaseFp' = publisherBaseFp
       /\ batch' = batch
       /\ pendingValid' = pendingValid
       /\ pendingRejected' = pendingRejected
       /\ nextOutcomeId' = nextOutcomeId
       /\ outcomeById' = outcomeById
       /\ dropped' = dropped
       /\ rejected' = rejected
       /\ applied' = applied
       /\ droppedCount' = droppedCount
       /\ rejectedCount' = rejectedCount
       /\ appliedCount' = appliedCount
       /\ updateCount' = updateCount
       /\ failureSeen' = failureSeen
       /\ commitLog' = commitLog
    BY DEF PublicationState
<1>2. BoundedQueue'
    BY <1>1
    DEF InductiveSafety, Safety, BoundedQueue
<1>3. UniqueOutcomeIds'
    BY <1>1
    DEF InductiveSafety, Safety, UniqueOutcomeIds, Ids
<1>4. DispositionDisjoint'
    BY <1>1
    DEF InductiveSafety, Safety, DispositionDisjoint, Ids
<1>5. OutcomeAccounting'
    BY <1>1
    DEF InductiveSafety, Safety, OutcomeAccounting, CreatedIds, Ids
<1>6. CountAccuracy'
    BY <1>1
    DEF InductiveSafety, Safety, CountAccuracy
<1>7. SnapshotGenerationAgreement'
    BY <1>1
    DEF InductiveSafety, Safety, SnapshotGenerationAgreement
<1>8. PublisherLockInvariant'
    BY <1>1, SMT
    DEF InductiveSafety,
        Safety,
        PublisherLockInvariant,
        ValidIdsAt,
        IsValidAt,
        Ids
<1>9. CorrectCommitClassification'
    BY <1>1
    DEF InductiveSafety,
        Safety,
        CorrectCommitClassification,
        CreatedIds
<1>10. ForeignApplicationExclusion'
    BY <1>1
    DEF InductiveSafety, Safety, ForeignApplicationExclusion
<1>11. InductiveStrengthening'
    BY <1>1
    DEF InductiveSafety,
        InductiveStrengthening,
        CreatedOutcomeIdentity,
        CreatedCanonicalValidity,
        CreatedIds
<1>12. QED BY <1>2,
                <1>3,
                <1>4,
                <1>5,
                <1>6,
                <1>7,
                <1>8,
                <1>9,
                <1>10,
                <1>11
    DEF InductiveSafety, Safety, InductiveStrengthening

EvidenceState ==
    <<generation,
      fingerprint,
      queue,
      resolverPhase,
      capturedGen,
      capturedFp,
      decisionGen,
      decisionFp,
      nextOutcomeId,
      outcomeById,
      dropped,
      rejected,
      applied,
      droppedCount,
      rejectedCount,
      appliedCount,
      updateCount,
      commitLog>>

LEMMA EqualEvidenceStatePreservesInductiveSafety ==
    ASSUME InductiveSafety,
           EvidenceState' = EvidenceState,
           TypeOK',
           BoundedQueue',
           PublisherLockInvariant'
    PROVE  InductiveSafety'
<1>1. QED BY
    DEF EvidenceState,
        InductiveSafety,
        InductiveStrengthening,
        Safety,
        UniqueOutcomeIds,
        DispositionDisjoint,
        OutcomeAccounting,
        CountAccuracy,
        SnapshotGenerationAgreement,
        ResolutionSnapshotConsistency,
        CorrectCommitClassification,
        ForeignApplicationExclusion,
        CreatedOutcomeIdentity,
        CreatedCanonicalValidity,
        CreatedIds,
        Ids

THEOREM ResolveStartPreservesInductiveSafety ==
    ASSUME ParameterAssumptions,
           NEW r,
           InductiveSafety,
           ResolveStart(r)
    PROVE  InductiveSafety'
<1>1. TypeOK'
    BY SMT
    DEF ResolveStart,
        InductiveSafety,
        Safety,
        TypeOK,
        ResolverPhases
<1>2. BoundedQueue'
    BY SMT
    DEF ResolveStart, InductiveSafety, Safety, BoundedQueue
<1>3. UniqueOutcomeIds'
    BY SMT
    DEF ResolveStart, InductiveSafety, Safety, UniqueOutcomeIds, Ids
<1>4. DispositionDisjoint'
    BY SMT
    DEF ResolveStart, InductiveSafety, Safety, DispositionDisjoint, Ids
<1>5. OutcomeAccounting'
    BY SMT
    DEF ResolveStart,
        InductiveSafety,
        Safety,
        OutcomeAccounting,
        CreatedIds,
        Ids
<1>6. CountAccuracy'
    BY SMT
    DEF ResolveStart, InductiveSafety, Safety, CountAccuracy
<1>7. SnapshotGenerationAgreement'
    BY SMT
    DEF ResolveStart,
        InductiveSafety,
        Safety,
        SnapshotGenerationAgreement
<1>8. PublisherLockInvariant'
    BY SMT
    DEF ResolveStart,
        InductiveSafety,
        Safety,
        PublisherLockInvariant
<1>9. ResolutionSnapshotConsistency'
    BY SMT
    DEF ResolveStart,
        InductiveSafety,
        Safety,
        ResolutionSnapshotConsistency,
        SnapshotGenerationAgreement,
        TypeOK,
        ResolverPhases
<1>10. CorrectCommitClassification'
    BY
    DEF ResolveStart,
        InductiveSafety,
        Safety,
        CorrectCommitClassification,
        CreatedIds
<1>11. ForeignApplicationExclusion'
    BY
    DEF ResolveStart,
        InductiveSafety,
        Safety,
        ForeignApplicationExclusion
<1>12. InductiveStrengthening'
    BY
    DEF ResolveStart,
        InductiveSafety,
        InductiveStrengthening,
        CreatedOutcomeIdentity,
        CreatedCanonicalValidity,
        CreatedIds
<1>13. QED BY <1>1,
                <1>2,
                <1>3,
                <1>4,
                <1>5,
                <1>6,
                <1>7,
                <1>8,
                <1>9,
                <1>10,
                <1>11,
                <1>12,
                SMT
    DEF InductiveSafety, Safety

THEOREM ResolveFinishPreservesInductiveSafety ==
    ASSUME ParameterAssumptions,
           NEW r,
           InductiveSafety,
           ResolveFinish(r)
    PROVE  InductiveSafety'
<1>1. TypeOK'
    BY SMT
    DEF ResolveFinish,
        InductiveSafety,
        Safety,
        TypeOK,
        ResolverPhases
<1>2. ResolutionSnapshotConsistency'
    BY SMT
    DEF ResolveFinish,
        InductiveSafety,
        Safety,
        ResolutionSnapshotConsistency,
        TypeOK,
        ResolverPhases
<1>3. PublicationState' = PublicationState
    BY DEF ResolveFinish, PublicationState
<1>4. QED BY <1>1,
                <1>2,
                <1>3,
                EqualPublicationStatePreservesInductiveSafety

THEOREM PublishBeginPreservesInductiveSafety ==
    ASSUME ParameterAssumptions,
           InductiveSafety,
           PublishBegin
    PROVE  InductiveSafety'
<1>1. TypeOK'
    BY EmptySeq, SMT
    DEF PublishBegin,
        InductiveSafety,
        Safety,
        TypeOK,
        PublisherPhases
<1>2. BoundedQueue'
    BY EmptySeq, SMT
    DEF PublishBegin,
        InductiveSafety,
        Safety,
        BoundedQueue,
        ParameterAssumptions
<1>3. PublisherLockInvariant'
    BY SMT
    DEF PublishBegin,
        InductiveSafety,
        Safety,
        PublisherLockInvariant
<1>4. EvidenceState' = EvidenceState
    BY DEF PublishBegin, EvidenceState
<1>5. QED BY <1>1,
                <1>2,
                <1>3,
                <1>4,
                EqualEvidenceStatePreservesInductiveSafety

THEOREM PublishCutPreservesInductiveSafety ==
    ASSUME ParameterAssumptions,
           InductiveSafety,
           PublishCut
    PROVE  InductiveSafety'
<1>1. TypeOK'
    BY SMT
    DEF PublishCut,
        InductiveSafety,
        Safety,
        TypeOK,
        PublisherPhases
<1>2. BoundedQueue'
    BY SMT
    DEF PublishCut,
        InductiveSafety,
        Safety,
        BoundedQueue
<1>3. PublisherLockInvariant'
    BY SMT
    DEF PublishCut,
        InductiveSafety,
        Safety,
        PublisherLockInvariant
<1>4. EvidenceState' = EvidenceState
    BY DEF PublishCut, EvidenceState
<1>5. QED BY <1>1,
                <1>2,
                <1>3,
                <1>4,
                EqualEvidenceStatePreservesInductiveSafety

THEOREM PublishEmptyPreservesInductiveSafety ==
    ASSUME ParameterAssumptions,
           InductiveSafety,
           PublishEmpty
    PROVE  InductiveSafety'
<1>1. TypeOK'
    BY EmptySeq, SMT
    DEF PublishEmpty,
        InductiveSafety,
        Safety,
        TypeOK,
        PublisherPhases
<1>2. BoundedQueue'
    BY EmptySeq, SMT
    DEF PublishEmpty,
        InductiveSafety,
        Safety,
        BoundedQueue,
        ParameterAssumptions
<1>3. PublisherLockInvariant'
    BY SMT
    DEF PublishEmpty,
        InductiveSafety,
        Safety,
        PublisherLockInvariant
<1>4. EvidenceState' = EvidenceState
    BY DEF PublishEmpty, EvidenceState
<1>5. QED BY <1>1,
                <1>2,
                <1>3,
                <1>4,
                EqualEvidenceStatePreservesInductiveSafety

THEOREM PublishBuildPreservesInductiveSafety ==
    ASSUME ParameterAssumptions,
           InductiveSafety,
           PublishBuild
    PROVE  InductiveSafety'
<1>1. TypeOK'
    BY SMT
    DEF PublishBuild,
        InductiveSafety,
        Safety,
        TypeOK,
        PublisherPhases,
        OutcomeType,
        OutcomeIds,
        Ids,
        ValidIdsAt,
        IsValidAt
<1>2. BoundedQueue'
    BY SMT
    DEF PublishBuild,
        InductiveSafety,
        Safety,
        BoundedQueue
<1>3. PublisherLockInvariant'
    BY SMT
    DEF PublishBuild,
        InductiveSafety,
        Safety,
        PublisherLockInvariant,
        ValidIdsAt,
        IsValidAt,
        Ids
<1>4. EvidenceState' = EvidenceState
    BY DEF PublishBuild, EvidenceState
<1>5. QED BY <1>1,
                <1>2,
                <1>3,
                <1>4,
                EqualEvidenceStatePreservesInductiveSafety

THEOREM PublishFailPreservesInductiveSafety ==
    ASSUME ParameterAssumptions,
           InductiveSafety,
           PublishFail
    PROVE  InductiveSafety'
<1>1. TypeOK'
    BY EmptySeq, SMT
    DEF PublishFail,
        InductiveSafety,
        Safety,
        TypeOK,
        PublisherPhases
<1>2. BoundedQueue'
    BY EmptySeq, SMT
    DEF PublishFail,
        InductiveSafety,
        Safety,
        BoundedQueue,
        ParameterAssumptions
<1>3. PublisherLockInvariant'
    BY SMT
    DEF PublishFail,
        InductiveSafety,
        Safety,
        PublisherLockInvariant
<1>4. EvidenceState' = EvidenceState
    BY DEF PublishFail, EvidenceState
<1>5. QED BY <1>1,
                <1>2,
                <1>3,
                <1>4,
                EqualEvidenceStatePreservesInductiveSafety

CanonicalSource(o) ==
    (o.bindingOK /\ o.fp = o.gen) => o.source = "local"

RecordOutcome(o) ==
    /\ o \in OutcomeType
    /\ o.id = nextOutcomeId
    /\ publisherPhase \in {"idle", "publish_locked"}
    /\ nextOutcomeId \in OutcomeIds
    /\ Enqueue(o.id)
    /\ outcomeById' = [outcomeById EXCEPT ![o.id] = o]
    /\ nextOutcomeId' = nextOutcomeId + 1
    /\ UNCHANGED <<generation,
                   fingerprint,
                   publisherPhase,
                   publisherBaseGen,
                   publisherBaseFp,
                   batch,
                   pendingValid,
                   pendingRejected,
                   rejected,
                   applied,
                   rejectedCount,
                   appliedCount,
                   updateCount,
                   failureSeen,
                   commitLog>>

LEMMA RecordOutcomeStorageType ==
    ASSUME ParameterAssumptions,
           NEW o,
           InductiveSafety,
           RecordOutcome(o)
    PROVE  /\ queue' \in Seq(OutcomeIds)
           /\ outcomeById' \in [OutcomeIds -> OutcomeType]
           /\ nextOutcomeId' \in 1..(MaxOutcomeId + 1)
           /\ dropped' \subseteq OutcomeIds
           /\ droppedCount' \in Nat
<1>1. /\ outcomeById' \in [OutcomeIds -> OutcomeType]
       /\ nextOutcomeId' \in 1..(MaxOutcomeId + 1)
    BY SMT
    DEF ParameterAssumptions,
        RecordOutcome,
        InductiveSafety,
        Safety,
        TypeOK,
        OutcomeIds
<1>2. ASSUME Len(queue) < QueueCapacity
       PROVE  /\ queue' \in Seq(OutcomeIds)
              /\ dropped' \subseteq OutcomeIds
              /\ droppedCount' \in Nat
    <2>1. /\ Len(queue) < QueueCapacity
           /\ queue' = Append(queue, o.id)
           /\ dropped' = dropped
           /\ droppedCount' = droppedCount
        BY <1>2, EnqueueBelowCapacity, SMT
        DEF RecordOutcome
    <2>2. QED BY <2>1, AppendProperties, SMT
        DEF InductiveSafety, Safety, TypeOK, RecordOutcome
<1>3. ASSUME Len(queue) >= QueueCapacity
       PROVE  /\ queue' \in Seq(OutcomeIds)
              /\ dropped' \subseteq OutcomeIds
              /\ droppedCount' \in Nat
    <2>1. /\ Len(queue) >= QueueCapacity
           /\ queue # <<>>
           /\ Head(queue) \in OutcomeIds
           /\ Tail(queue) \in Seq(OutcomeIds)
        BY <1>3, EmptySeq, HeadTailProperties, SMT
        DEF ParameterAssumptions,
            InductiveSafety,
            Safety,
            TypeOK,
            BoundedQueue
    <2>2. /\ queue' = Append(Tail(queue), o.id)
           /\ dropped' = dropped \cup {Head(queue)}
           /\ droppedCount' = droppedCount + 1
        BY <2>1, EnqueueAtCapacity, SMT
        DEF ParameterAssumptions,
            RecordOutcome,
            InductiveSafety,
            Safety,
            TypeOK
    <2>3. queue' \in Seq(OutcomeIds)
        BY <2>1, <2>2, AppendProperties
        DEF RecordOutcome
    <2>4. /\ dropped' \subseteq OutcomeIds
           /\ droppedCount' \in Nat
        BY <2>1, <2>2, SMT
        DEF InductiveSafety, Safety, TypeOK
    <2>5. QED BY <2>3, <2>4
<1>4. /\ Len(queue) \in Nat
       /\ QueueCapacity \in Nat
    BY SMT
    DEF ParameterAssumptions, InductiveSafety, Safety, TypeOK
<1>5. QED BY <1>1, <1>2, <1>3, <1>4, SMT

THEOREM RecordOutcomePreservesInductiveSafety ==
    ASSUME ParameterAssumptions,
           NEW o,
           InductiveSafety,
           RecordOutcome(o),
           CanonicalSource(o),
           TypeOK',
           ResolutionSnapshotConsistency'
    PROVE  InductiveSafety'
<1>1. /\ nextOutcomeId \in Nat \ {0}
       /\ nextOutcomeId \notin CreatedIds
       /\ nextOutcomeId \notin Ids(queue)
       /\ nextOutcomeId \notin dropped
       /\ nextOutcomeId \notin rejected
       /\ nextOutcomeId \notin applied
       /\ CreatedIds' = CreatedIds \cup {nextOutcomeId}
       /\ outcomeById'[nextOutcomeId] = o
       /\ \A id \in CreatedIds :
             outcomeById'[id] = outcomeById[id]
    <2>1. /\ nextOutcomeId \in Nat \ {0}
           /\ nextOutcomeId \notin CreatedIds
           /\ nextOutcomeId \notin Ids(queue)
           /\ nextOutcomeId \notin dropped
           /\ nextOutcomeId \notin rejected
           /\ nextOutcomeId \notin applied
        <3>1. /\ nextOutcomeId \in Nat \ {0}
               /\ nextOutcomeId \notin CreatedIds
            BY SMT
            DEF ParameterAssumptions,
                RecordOutcome,
                CreatedIds,
                OutcomeIds
        <3>2. /\ nextOutcomeId \notin Ids(queue)
               /\ nextOutcomeId \notin dropped
               /\ nextOutcomeId \notin rejected
               /\ nextOutcomeId \notin applied
            BY <3>1, SMT
            DEF InductiveSafety,
                Safety,
                OutcomeAccounting,
                CreatedIds
        <3>3. QED BY <3>1, <3>2
    <2>2. CreatedIds' = CreatedIds \cup {nextOutcomeId}
        BY <2>1, CreatedIdsAdvance, SMT
        DEF RecordOutcome, CreatedIds
    <2>3. /\ outcomeById'[nextOutcomeId] = o
           /\ \A id \in CreatedIds :
                 outcomeById'[id] = outcomeById[id]
        <3>1. /\ outcomeById \in [OutcomeIds -> OutcomeType]
               /\ nextOutcomeId \in OutcomeIds
               /\ o \in OutcomeType
               /\ outcomeById' =
                    [outcomeById EXCEPT ![nextOutcomeId] = o]
            BY SMT
            DEF RecordOutcome, InductiveSafety, Safety, TypeOK
        <3>2. /\ outcomeById'[nextOutcomeId] = o
               /\ \A id \in OutcomeIds \ {nextOutcomeId} :
                     outcomeById'[id] = outcomeById[id]
            BY <3>1, ExceptAtAndAway
        <3>3. CreatedIds \subseteq
                 OutcomeIds \ {nextOutcomeId}
            BY <2>1, SMT
            DEF InductiveSafety,
                Safety,
                TypeOK,
                OutcomeAccounting,
                CreatedIds,
                OutcomeIds,
                Ids
        <3>4. QED BY <3>2, <3>3, SMT
    <2>4. QED BY <2>1, <2>2, <2>3
<1>2. SnapshotGenerationAgreement'
    BY SMT
    DEF RecordOutcome,
        InductiveSafety,
        Safety,
        SnapshotGenerationAgreement
<1>3. PublisherLockInvariant'
    BY SMT
    DEF RecordOutcome,
        InductiveSafety,
        Safety,
        PublisherLockInvariant,
        ValidIdsAt,
        IsValidAt,
        Ids
<1>4. CorrectCommitClassification'
    BY <1>1, SMT
    DEF RecordOutcome,
        InductiveSafety,
        Safety,
        CorrectCommitClassification,
        CreatedIds
<1>5. ForeignApplicationExclusion'
    BY <1>1, SMT
    DEF RecordOutcome,
        InductiveSafety,
        Safety,
        ForeignApplicationExclusion,
        OutcomeAccounting,
        CreatedIds,
        Ids
<1>6. InductiveStrengthening'
    BY <1>1, SMT
    DEF RecordOutcome,
        CanonicalSource,
        InductiveSafety,
        InductiveStrengthening,
        CreatedOutcomeIdentity,
        CreatedCanonicalValidity,
        CreatedIds
<1>7. ASSUME Len(queue) < QueueCapacity
       PROVE  InductiveSafety'
    <2>1. /\ Len(queue) < QueueCapacity
           /\ queue' = Append(queue, nextOutcomeId)
           /\ dropped' = dropped
           /\ droppedCount' = droppedCount
        BY <1>7, EnqueueBelowCapacity, SMT
        DEF RecordOutcome
    <2>2. Ids(queue') = Ids(queue) \cup {nextOutcomeId}
        BY <2>1, IdsAppend
        DEF InductiveSafety, Safety, TypeOK
    <2>3. UniqueOutcomeIds'
        BY <1>1, <2>1, FreshAppendPreservesInjectivity
        DEF InductiveSafety,
            Safety,
            TypeOK,
            UniqueOutcomeIds
    <2>4. BoundedQueue'
        <3>1. Len(queue') = Len(queue) + 1
            BY <2>1, AppendProperties
            DEF InductiveSafety, Safety, TypeOK
        <3>2. /\ Len(queue) < QueueCapacity
               /\ batch' = batch
               /\ Len(batch) <= QueueCapacity
               /\ Len(queue) \in Nat
               /\ QueueCapacity \in Nat
            BY <2>1, SMT
            DEF RecordOutcome,
                InductiveSafety,
                Safety,
                TypeOK,
                BoundedQueue,
                ParameterAssumptions
        <3>3. /\ Len(queue') <= QueueCapacity
               /\ Len(batch') <= QueueCapacity
            BY <3>1, <3>2, SMT
        <3>4. QED BY <3>3
            DEF BoundedQueue
    <2>5. DispositionDisjoint'
        BY <1>1, <2>1, <2>2, SMT
        DEF RecordOutcome,
            InductiveSafety,
            Safety,
            DispositionDisjoint
    <2>6. OutcomeAccounting'
        BY <1>1, <2>1, <2>2, SMT
        DEF RecordOutcome,
            InductiveSafety,
            Safety,
            OutcomeAccounting
    <2>7. CountAccuracy'
        BY <2>1, SMT
        DEF RecordOutcome,
            InductiveSafety,
            Safety,
            CountAccuracy
    <2>8. QED BY <1>2,
                  <1>3,
                  <1>4,
                  <1>5,
                  <1>6,
                  <2>3,
                  <2>4,
                  <2>5,
                  <2>6,
                  <2>7
        DEF InductiveSafety, Safety, InductiveStrengthening
<1>8. ASSUME Len(queue) >= QueueCapacity
       PROVE  InductiveSafety'
    <2>1. /\ Len(queue) >= QueueCapacity
           /\ queue' = Append(Tail(queue), nextOutcomeId)
           /\ dropped' = dropped \cup {Head(queue)}
           /\ droppedCount' = droppedCount + 1
        BY <1>8, EnqueueAtCapacity, SMT
        DEF RecordOutcome,
            InductiveSafety,
            Safety,
            TypeOK,
            ParameterAssumptions
    <2>2. /\ Len(queue) = QueueCapacity
           /\ queue # <<>>
           /\ Head(queue) \in Ids(queue)
           /\ Head(queue) \notin dropped
           /\ Head(queue) \notin rejected
           /\ Head(queue) \notin applied
           /\ IsFiniteSet(dropped)
        BY <2>1,
           OutcomeIdsFinite,
           SubsetOfOutcomeIdsFinite,
           EmptySeq,
           HeadTailProperties,
           IdsEqualsRange,
           SMT
        DEF ParameterAssumptions,
            InductiveSafety,
            Safety,
            TypeOK,
            BoundedQueue,
            DispositionDisjoint,
            OutcomeIds
    <2>3. /\ Ids(Tail(queue)) = Ids(queue) \ {Head(queue)}
           /\ SeqInjective(Tail(queue))
           /\ Tail(queue) \in Seq(OutcomeIds)
        BY <2>2,
           IdsTailOfInjectiveSequence,
           TailPreservesInjectivity,
           HeadTailProperties
        DEF InductiveSafety, Safety, TypeOK, UniqueOutcomeIds
    <2>4. Ids(queue') =
             (Ids(queue) \ {Head(queue)}) \cup {nextOutcomeId}
        <3>1. Ids(queue') =
                 Ids(Tail(queue)) \cup {nextOutcomeId}
            BY <2>1, <2>3, IdsAppend
            DEF InductiveSafety, Safety, TypeOK
        <3>2. QED BY <2>3, <3>1
    <2>5. UniqueOutcomeIds'
        BY <1>1,
           <2>1,
           <2>2,
           <2>3,
           FreshAppendPreservesInjectivity,
           SMT
        DEF InductiveSafety,
            Safety,
            TypeOK,
            UniqueOutcomeIds
    <2>6. BoundedQueue'
        BY <2>1,
           <2>2,
           HeadTailProperties,
           AppendProperties,
           SMT
        DEF RecordOutcome,
            InductiveSafety,
            Safety,
            TypeOK,
            BoundedQueue,
            ParameterAssumptions
    <2>7. Cardinality(dropped') = Cardinality(dropped) + 1
        BY <2>1, <2>2, FS_AddElement, SMT
    <2>8. DispositionDisjoint'
        BY <1>1, <2>1, <2>2, <2>4, SMT
        DEF RecordOutcome,
            InductiveSafety,
            Safety,
            DispositionDisjoint
    <2>9. OutcomeAccounting'
        BY <1>1, <2>1, <2>2, <2>4, SMT
        DEF RecordOutcome,
            InductiveSafety,
            Safety,
            OutcomeAccounting
    <2>10. CountAccuracy'
        BY <2>1, <2>2, <2>7, SMT
        DEF RecordOutcome,
            InductiveSafety,
            Safety,
            CountAccuracy
    <2>11. QED BY <1>2,
                   <1>3,
                   <1>4,
                   <1>5,
                   <1>6,
                   <2>5,
                   <2>6,
                   <2>8,
                   <2>9,
                   <2>10
        DEF InductiveSafety, Safety, InductiveStrengthening
<1>9. /\ Len(queue) \in Nat
       /\ QueueCapacity \in Nat
    BY SMT
    DEF ParameterAssumptions,
        InductiveSafety,
        Safety,
        TypeOK
<1>10. QED BY <1>7, <1>8, <1>9, SMT

THEOREM ObservePreservesInductiveSafety ==
    ASSUME ParameterAssumptions,
           NEW r,
           InductiveSafety,
           Observe(r)
    PROVE  InductiveSafety'
<1>1. RecordOutcome(LocalOutcome(r))
    BY SMT
    DEF Observe,
        RecordOutcome,
        LocalOutcome,
        InductiveSafety,
        Safety,
        TypeOK,
        ResolutionSnapshotConsistency,
        OutcomeType,
        OutcomeSources,
        OutcomeIds
<1>2. CanonicalSource(LocalOutcome(r))
    BY DEF CanonicalSource, LocalOutcome
<1>3. TypeOK'
    <2>1. LocalOutcome(r) \in OutcomeType
        BY SMT
        DEF Observe,
            LocalOutcome,
            InductiveSafety,
            Safety,
            TypeOK,
            ResolutionSnapshotConsistency,
            OutcomeType,
            OutcomeSources,
            OutcomeIds
    <2>2. /\ queue' \in Seq(OutcomeIds)
           /\ dropped' \subseteq OutcomeIds
           /\ droppedCount' \in Nat
        BY <1>1, RecordOutcomeStorageType
    <2>3. outcomeById' \in [OutcomeIds -> OutcomeType]
        BY <1>1, RecordOutcomeStorageType
    <2>4. nextOutcomeId' \in 1..(MaxOutcomeId + 1)
        BY <1>1, RecordOutcomeStorageType
    <2>5. resolverPhase' \in [Resolvers -> ResolverPhases]
        BY SMT
        DEF Observe,
            InductiveSafety,
            Safety,
            TypeOK,
            ResolverPhases
    <2>6. QED BY <2>2, <2>3, <2>4, <2>5, SMT
        DEF Observe,
            InductiveSafety,
            Safety,
            TypeOK
<1>4. ResolutionSnapshotConsistency'
    BY SMT
    DEF Observe,
        InductiveSafety,
        Safety,
        TypeOK,
        ResolutionSnapshotConsistency,
        ResolverPhases
<1>5. QED BY <1>1,
                <1>2,
                <1>3,
                <1>4,
                RecordOutcomePreservesInductiveSafety

THEOREM ObserveForeignPreservesInductiveSafety ==
    ASSUME ParameterAssumptions,
           InductiveSafety,
           ObserveForeign
    PROVE  InductiveSafety'
<1>1. RecordOutcome(ForeignOutcome)
    BY SMT
    DEF ParameterAssumptions,
        ObserveForeign,
        RecordOutcome,
        ForeignOutcome,
        InductiveSafety,
        Safety,
        TypeOK,
        SnapshotGenerationAgreement,
        OutcomeType,
        OutcomeSources,
        OutcomeIds
<1>2. CanonicalSource(ForeignOutcome)
    <2>1. ForeignOutcome.fp # ForeignOutcome.gen
        BY SMT
        DEF ParameterAssumptions,
            ForeignOutcome,
            InductiveSafety,
            Safety,
            SnapshotGenerationAgreement,
            TypeOK
    <2>2. QED BY <2>1
        DEF CanonicalSource
<1>3. TypeOK'
    <2>1. /\ queue' \in Seq(OutcomeIds)
           /\ outcomeById' \in [OutcomeIds -> OutcomeType]
           /\ nextOutcomeId' \in 1..(MaxOutcomeId + 1)
           /\ dropped' \subseteq OutcomeIds
           /\ droppedCount' \in Nat
        BY <1>1, RecordOutcomeStorageType
    <2>2. QED BY <2>1, SMT
        DEF ObserveForeign,
            InductiveSafety,
            Safety,
            TypeOK
<1>4. ResolutionSnapshotConsistency'
    BY SMT
    DEF ObserveForeign,
        InductiveSafety,
        Safety,
        ResolutionSnapshotConsistency
<1>5. QED BY <1>1,
                <1>2,
                <1>3,
                <1>4,
                RecordOutcomePreservesInductiveSafety

THEOREM ObserveInvalidPreservesInductiveSafety ==
    ASSUME ParameterAssumptions,
           InductiveSafety,
           ObserveInvalid
    PROVE  InductiveSafety'
<1>1. RecordOutcome(InvalidOutcome)
    BY SMT
    DEF ObserveInvalid,
        RecordOutcome,
        InvalidOutcome,
        InductiveSafety,
        Safety,
        TypeOK,
        OutcomeType,
        OutcomeSources,
        OutcomeIds
<1>2. CanonicalSource(InvalidOutcome)
    BY DEF CanonicalSource, InvalidOutcome
<1>3. TypeOK'
    <2>1. /\ queue' \in Seq(OutcomeIds)
           /\ outcomeById' \in [OutcomeIds -> OutcomeType]
           /\ nextOutcomeId' \in 1..(MaxOutcomeId + 1)
           /\ dropped' \subseteq OutcomeIds
           /\ droppedCount' \in Nat
        BY <1>1, RecordOutcomeStorageType
    <2>2. QED BY <2>1, SMT
        DEF ObserveInvalid,
            InductiveSafety,
            Safety,
            TypeOK
<1>4. ResolutionSnapshotConsistency'
    BY SMT
    DEF ObserveInvalid,
        InductiveSafety,
        Safety,
        ResolutionSnapshotConsistency
<1>5. QED BY <1>1,
                <1>2,
                <1>3,
                <1>4,
                RecordOutcomePreservesInductiveSafety

LEMMA ReadyBatchFacts ==
    ASSUME ParameterAssumptions,
           InductiveSafety,
           publisherPhase = "ready"
    PROVE  /\ batch = queue
           /\ batch # <<>>
           /\ generation = publisherBaseGen
           /\ fingerprint = publisherBaseFp
           /\ generation = fingerprint
           /\ pendingValid =
                ValidIdsAt(batch, publisherBaseGen, publisherBaseFp)
           /\ pendingRejected = Ids(batch) \ pendingValid
           /\ Ids(batch) \subseteq OutcomeIds
           /\ pendingValid \subseteq Ids(batch)
           /\ pendingValid \cup pendingRejected = Ids(batch)
           /\ pendingValid \cap pendingRejected = {}
           /\ applied \cap pendingValid = {}
           /\ rejected \cap pendingRejected = {}
           /\ IsFiniteSet(applied)
           /\ IsFiniteSet(rejected)
           /\ IsFiniteSet(pendingValid)
           /\ IsFiniteSet(pendingRejected)
<1>1. /\ batch = queue
       /\ batch # <<>>
       /\ generation = publisherBaseGen
       /\ fingerprint = publisherBaseFp
       /\ generation = fingerprint
       /\ pendingValid =
            ValidIdsAt(batch, publisherBaseGen, publisherBaseFp)
       /\ pendingRejected = Ids(batch) \ pendingValid
    BY SMT
    DEF InductiveSafety,
        Safety,
        PublisherLockInvariant,
        SnapshotGenerationAgreement
<1>2. Ids(batch) \subseteq OutcomeIds
    BY IdsEqualsRange, RangeOfSeq
    DEF InductiveSafety, Safety, TypeOK
<1>3. /\ pendingValid \subseteq Ids(batch)
       /\ pendingValid \cup pendingRejected = Ids(batch)
       /\ pendingValid \cap pendingRejected = {}
    BY <1>1, ValidIdsAtSubset, SMT
<1>4. /\ applied \cap pendingValid = {}
       /\ rejected \cap pendingRejected = {}
    BY <1>1, <1>3, SMT
    DEF InductiveSafety, Safety, DispositionDisjoint
<1>5. /\ IsFiniteSet(applied)
       /\ IsFiniteSet(rejected)
       /\ IsFiniteSet(pendingValid)
       /\ IsFiniteSet(pendingRejected)
    BY <1>2,
       <1>3,
       SubsetOfOutcomeIdsFinite,
       SMT
    DEF InductiveSafety, Safety, TypeOK
<1>6. QED BY <1>1, <1>2, <1>3, <1>4, <1>5

THEOREM PublishCommitPreservesInductiveSafety ==
    ASSUME ParameterAssumptions,
           InductiveSafety,
           PublishCommit
    PROVE  InductiveSafety'
<1>1. /\ batch = queue
       /\ batch # <<>>
       /\ generation = publisherBaseGen
       /\ fingerprint = publisherBaseFp
       /\ generation = fingerprint
       /\ pendingValid =
            ValidIdsAt(batch, publisherBaseGen, publisherBaseFp)
       /\ pendingRejected = Ids(batch) \ pendingValid
       /\ Ids(batch) \subseteq OutcomeIds
       /\ pendingValid \subseteq Ids(batch)
       /\ pendingValid \cup pendingRejected = Ids(batch)
       /\ pendingValid \cap pendingRejected = {}
       /\ applied \cap pendingValid = {}
       /\ rejected \cap pendingRejected = {}
       /\ IsFiniteSet(applied)
       /\ IsFiniteSet(rejected)
       /\ IsFiniteSet(pendingValid)
       /\ IsFiniteSet(pendingRejected)
    BY ReadyBatchFacts
    DEF PublishCommit
<1>2. /\ queue' = <<>>
       /\ applied' = applied \cup pendingValid
       /\ rejected' = rejected \cup pendingRejected
       /\ appliedCount' =
            appliedCount + Cardinality(pendingValid)
       /\ rejectedCount' =
            rejectedCount + Cardinality(pendingRejected)
       /\ commitLog' = Append(commitLog, CurrentCommitEvent)
       /\ publisherPhase' = "idle"
       /\ batch' = <<>>
       /\ pendingValid' = {}
       /\ pendingRejected' = {}
       /\ nextOutcomeId' = nextOutcomeId
       /\ outcomeById' = outcomeById
       /\ dropped' = dropped
       /\ droppedCount' = droppedCount
    BY DEF PublishCommit
<1>3. /\ Cardinality(applied') =
             Cardinality(applied) + Cardinality(pendingValid)
       /\ Cardinality(rejected') =
             Cardinality(rejected) + Cardinality(pendingRejected)
    <2>1. Cardinality(applied \cup pendingValid) =
               Cardinality(applied) + Cardinality(pendingValid)
        <3>1. Cardinality(applied \cup pendingValid) =
                   Cardinality(applied)
                     + Cardinality(pendingValid)
                     - Cardinality(applied \cap pendingValid)
            BY <1>1, FS_Union
        <3>2. Cardinality(applied \cap pendingValid) = 0
            BY <1>1, FS_EmptySet
        <3>3. QED BY <1>1,
                       <3>1,
                       <3>2,
                       FS_CardinalityType,
                       SMT
    <2>2. Cardinality(rejected \cup pendingRejected) =
               Cardinality(rejected) + Cardinality(pendingRejected)
        <3>1. Cardinality(rejected \cup pendingRejected) =
                   Cardinality(rejected)
                     + Cardinality(pendingRejected)
                     - Cardinality(rejected \cap pendingRejected)
            BY <1>1, FS_Union
        <3>2. Cardinality(rejected \cap pendingRejected) = 0
            BY <1>1, FS_EmptySet
        <3>3. QED BY <1>1,
                       <3>1,
                       <3>2,
                       FS_CardinalityType,
                       SMT
    <2>3. QED BY <1>2, <2>1, <2>2
<1>4. TypeOK'
    <2>1. CurrentCommitEvent \in CommitEventType
        BY <1>1, SMT
        DEF CurrentCommitEvent,
            CommitEventType,
            InductiveSafety,
            Safety,
            TypeOK
    <2>2. commitLog' \in Seq(CommitEventType)
        BY <1>2, <2>1, AppendProperties
        DEF InductiveSafety, Safety, TypeOK
    <2>3. /\ applied' \subseteq OutcomeIds
           /\ rejected' \subseteq OutcomeIds
           /\ appliedCount' \in Nat
           /\ rejectedCount' \in Nat
        BY <1>1,
           <1>2,
           FS_CardinalityType,
           SMT
        DEF InductiveSafety, Safety, TypeOK
    <2>4. /\ generation' \in Nat
           /\ fingerprint' \in Nat
           /\ updateCount' \in Nat
        BY SMT
        DEF PublishCommit, InductiveSafety, Safety, TypeOK
    <2>5. QED BY <1>2, <2>2, <2>3, <2>4, EmptySeq, SMT
        DEF PublishCommit,
            InductiveSafety,
            Safety,
            TypeOK,
            PublisherPhases
<1>5. BoundedQueue'
    BY <1>2, EmptySeq, SMT
    DEF BoundedQueue, ParameterAssumptions
<1>6. UniqueOutcomeIds'
    BY <1>2, SMT
    DEF UniqueOutcomeIds, SeqInjective
<1>7. DispositionDisjoint'
    BY <1>1, <1>2, EmptySeq, SMT
    DEF InductiveSafety,
        Safety,
        DispositionDisjoint,
        Ids
<1>8. OutcomeAccounting'
    BY <1>1, <1>2, EmptySeq, SMT
    DEF InductiveSafety,
        Safety,
        OutcomeAccounting,
        CreatedIds,
        Ids
<1>9. CountAccuracy'
    BY <1>2, <1>3, SMT
    DEF InductiveSafety, Safety, CountAccuracy
<1>10. SnapshotGenerationAgreement'
    BY SMT
    DEF PublishCommit,
        InductiveSafety,
        Safety,
        SnapshotGenerationAgreement
<1>11. PublisherLockInvariant'
    BY <1>2, SMT
    DEF PublisherLockInvariant
<1>12. ResolutionSnapshotConsistency'
    BY SMT
    DEF PublishCommit,
        InductiveSafety,
        Safety,
        TypeOK,
        ResolutionSnapshotConsistency
<1>13. CorrectCommitClassification'
    <2>1. /\ Len(commitLog') = Len(commitLog) + 1
           /\ \A k \in 1..Len(commitLog) :
                 commitLog'[k] = commitLog[k]
           /\ commitLog'[Len(commitLog) + 1] =
                 CurrentCommitEvent
        BY <1>2, AppendProperties
        DEF InductiveSafety, Safety, TypeOK
    <2>1a. commitLog'[Len(commitLog) + 1] =
                 CurrentCommitEvent
        BY <1>2, AppendProperties
        DEF InductiveSafety, Safety, TypeOK
    <2>2. \A id \in Ids(batch) :
             /\ id \in CreatedIds
             /\ (id \in pendingValid)
                   = IsValidAt(outcomeById[id],
                               publisherBaseGen,
                               publisherBaseFp)
        BY <1>1, ValidIdsAtCharacterization, SMT
        DEF InductiveSafety,
            Safety,
            OutcomeAccounting
    <2>3. /\ CurrentCommitEvent.cut =
                 CurrentCommitEvent.applied
                   \cup CurrentCommitEvent.rejected
           /\ CurrentCommitEvent.applied
                 \cap CurrentCommitEvent.rejected = {}
           /\ \A id \in CurrentCommitEvent.cut :
                 /\ id \in CreatedIds
                 /\ (id \in CurrentCommitEvent.applied)
                       = IsValidAt(outcomeById[id],
                                   CurrentCommitEvent.baseGen,
                                   CurrentCommitEvent.baseFp)
        BY <1>1, <2>2
        DEF CurrentCommitEvent
    <2>4. /\ CreatedIds' = CreatedIds
           /\ outcomeById' = outcomeById
        BY <1>2
        DEF CreatedIds
    <2>5. SUFFICES
             ASSUME NEW k \in 1..Len(commitLog')
             PROVE  LET e == commitLog'[k]
                    IN /\ e.cut = e.applied \cup e.rejected
                       /\ e.applied \cap e.rejected = {}
                       /\ \A id \in e.cut :
                             /\ id \in CreatedIds'
                             /\ (id \in e.applied) =
                                  IsValidAt(outcomeById'[id],
                                            e.baseGen,
                                            e.baseFp)
        BY DEF CorrectCommitClassification
    <2>6. \/ k \in 1..Len(commitLog)
           \/ k = Len(commitLog) + 1
        BY <2>1, <2>5, SMT
        DEF InductiveSafety, Safety, TypeOK
    <2>7. CASE k \in 1..Len(commitLog)
        BY <1>2, <2>1, <2>4, <2>7, SMT
        DEF InductiveSafety,
            Safety,
            CorrectCommitClassification
    <2>8. CASE k = Len(commitLog) + 1
        <3>1. k = Len(commitLog) + 1
            BY <2>8
        <3>2. commitLog'[k] = CurrentCommitEvent
            BY <2>1a, <3>1, SMT
        <3>3. /\ CurrentCommitEvent.cut =
                       CurrentCommitEvent.applied
                         \cup CurrentCommitEvent.rejected
               /\ CurrentCommitEvent.applied
                       \cap CurrentCommitEvent.rejected = {}
               /\ \A id \in CurrentCommitEvent.cut :
                     /\ id \in CreatedIds'
                     /\ (id \in CurrentCommitEvent.applied)
                           = IsValidAt(outcomeById'[id],
                                       CurrentCommitEvent.baseGen,
                                       CurrentCommitEvent.baseFp)
            BY <2>3, <2>4
        <3>4. QED BY <3>2, <3>3
    <2>9. QED BY <2>6, <2>7, <2>8
<1>14. ForeignApplicationExclusion'
    <2>1. \A id \in pendingValid :
             outcomeById[id].source = "local"
        BY <1>1, ValidIdsAtCharacterization, SMT
        DEF InductiveSafety,
            InductiveStrengthening,
            CreatedCanonicalValidity,
            Safety,
            OutcomeAccounting,
            IsValidAt,
            CreatedIds
    <2>2. QED BY <1>2, <2>1, SMT
        DEF InductiveSafety,
            Safety,
            ForeignApplicationExclusion
<1>15. InductiveStrengthening'
    BY <1>2
    DEF InductiveSafety,
        InductiveStrengthening,
        CreatedOutcomeIdentity,
        CreatedCanonicalValidity,
        CreatedIds
<1>16. QED BY <1>4,
                 <1>5,
                 <1>6,
                 <1>7,
                 <1>8,
                 <1>9,
                 <1>10,
                 <1>11,
                 <1>12,
                 <1>13,
                 <1>14,
                 <1>15
    DEF InductiveSafety, Safety, InductiveStrengthening

THEOREM NextPreservesInductiveSafety ==
    ASSUME ParameterAssumptions,
           InductiveSafety,
           Next
    PROVE  InductiveSafety'
<1>1. \A r \in Resolvers :
           ResolveStart(r) => InductiveSafety'
    BY ResolveStartPreservesInductiveSafety
<1>2. \A r \in Resolvers :
           ResolveFinish(r) => InductiveSafety'
    BY ResolveFinishPreservesInductiveSafety
<1>3. \A r \in Resolvers :
           Observe(r) => InductiveSafety'
    BY ObservePreservesInductiveSafety
<1>4. ObserveForeign => InductiveSafety'
    BY ObserveForeignPreservesInductiveSafety
<1>5. ObserveInvalid => InductiveSafety'
    BY ObserveInvalidPreservesInductiveSafety
<1>6. PublishBegin => InductiveSafety'
    BY PublishBeginPreservesInductiveSafety
<1>7. PublishCut => InductiveSafety'
    BY PublishCutPreservesInductiveSafety
<1>8. PublishEmpty => InductiveSafety'
    BY PublishEmptyPreservesInductiveSafety
<1>9. PublishBuild => InductiveSafety'
    BY PublishBuildPreservesInductiveSafety
<1>10. PublishFail => InductiveSafety'
    BY PublishFailPreservesInductiveSafety
<1>11. PublishCommit => InductiveSafety'
    BY PublishCommitPreservesInductiveSafety
<1>12. QED BY <1>1,
                 <1>2,
                 <1>3,
                 <1>4,
                 <1>5,
                 <1>6,
                 <1>7,
                 <1>8,
                 <1>9,
                 <1>10,
                 <1>11
    DEF Next

THEOREM StutteringPreservesInductiveSafety ==
    ASSUME InductiveSafety,
           UNCHANGED vars
    PROVE  InductiveSafety'
<1>1. PublicationState' = PublicationState
    BY SMT
    DEF vars, PublicationState
<1>2. TypeOK'
    BY SMT
    DEF vars, InductiveSafety, Safety, TypeOK
<1>3. ResolutionSnapshotConsistency'
    BY SMT
    DEF vars,
        InductiveSafety,
        Safety,
        ResolutionSnapshotConsistency
<1>4. QED BY <1>1,
                <1>2,
                <1>3,
                EqualPublicationStatePreservesInductiveSafety

THEOREM BoxedNextPreservesInductiveSafety ==
    ASSUME ParameterAssumptions,
           InductiveSafety,
           [Next]_vars
    PROVE  InductiveSafety'
<1>1. CASE Next
    <2>1. QED BY <1>1, NextPreservesInductiveSafety
<1>2. CASE UNCHANGED vars
    <2>1. QED BY <1>2, StutteringPreservesInductiveSafety
<1>3. QED BY <1>1, <1>2

THEOREM SafetySpecImpliesAlwaysSafety ==
    ASSUME ParameterAssumptions
    PROVE  SafetySpec => []Safety
<1>1. Init => InductiveSafety
    BY InitEstablishesInductiveSafety
<1>2. InductiveSafety /\ [Next]_vars => InductiveSafety'
    BY BoxedNextPreservesInductiveSafety
<1>3. InductiveSafety => Safety
    BY DEF InductiveSafety
<1>4. QED BY PTL, <1>1, <1>2, <1>3
    DEF SafetySpec

=============================================================================
