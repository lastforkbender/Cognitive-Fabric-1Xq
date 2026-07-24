---------------------- MODULE CognitiveFabricPublication ----------------------
\* CF-PUB-1.0
\*
\* Safety abstraction for Cognitive Fabric snapshot publication.
\*
\* The model deliberately exposes the two-lock protocol as a four-phase
\* publication protocol:
\*
\*   idle -> publish_locked -> building -> ready -> idle
\*
\* A resolver may finish evaluation while a publisher is active because it
\* evaluates an immutable snapshot captured earlier.  An observation may append
\* while only the publication lock is held, but not after the publisher has
\* acquired the outcome lock and cut the queue.
\*
\* Ghost state (outcomeById, disposition sets, commitLog, updateCount)
\* records proof-relevant history that the Python implementation need not store.

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Resolvers, QueueCapacity, MaxOutcomeId

ParameterAssumptions ==
    /\ IsFiniteSet(Resolvers)
    /\ Resolvers # {}
    /\ QueueCapacity \in Nat \ {0}
    /\ MaxOutcomeId \in Nat \ {0}

ASSUME ParameterAssumptions

ResolverPhases == {"idle", "evaluating", "ready"}
PublisherPhases == {"idle", "publish_locked", "building", "ready"}
OutcomeSources == {"local", "foreign", "invalid"}
OutcomeIds == 1..MaxOutcomeId

OutcomeType ==
    [id : OutcomeIds,
     gen : Nat,
     fp : Nat,
     bindingOK : BOOLEAN,
     source : OutcomeSources]

CommitEventType ==
    [baseGen : Nat,
     baseFp : Nat,
     cut : SUBSET OutcomeIds,
     applied : SUBSET OutcomeIds,
     rejected : SUBSET OutcomeIds]

VARIABLES
    generation,
    fingerprint,
    queue,
    resolverPhase,
    capturedGen,
    capturedFp,
    decisionGen,
    decisionFp,
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
    commitLog

vars ==
    <<generation,
      fingerprint,
      queue,
      resolverPhase,
      capturedGen,
      capturedFp,
      decisionGen,
      decisionFp,
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

Ids(s) == {s[i] : i \in 1..Len(s)}

SeqInjective(s) ==
    \A i, j \in 1..Len(s) :
        s[i] = s[j] => i = j

CreatedIds == 1..(nextOutcomeId - 1)

IsValidAt(o, gen, fp) == /\ o.bindingOK
                             /\ o.gen = gen
                             /\ o.fp = fp

LocalOutcome(r) ==
    [id |-> nextOutcomeId,
     gen |-> decisionGen[r],
     fp |-> decisionFp[r],
     bindingOK |-> TRUE,
     source |-> "local"]

ForeignOutcome ==
    [id |-> nextOutcomeId,
     gen |-> generation,
     fp |-> fingerprint + MaxOutcomeId + 1,
     bindingOK |-> TRUE,
     source |-> "foreign"]

InvalidOutcome ==
    [id |-> nextOutcomeId,
     gen |-> generation,
     fp |-> fingerprint,
     bindingOK |-> FALSE,
     source |-> "invalid"]

CurrentCommitEvent ==
    [baseGen |-> publisherBaseGen,
     baseFp |-> publisherBaseFp,
     cut |-> Ids(batch),
     applied |-> pendingValid,
     rejected |-> pendingRejected]

ValidIdsAt(s, gen, fp) ==
    {s[i] :
        i \in {j \in 1..Len(s) :
            IsValidAt(outcomeById[s[j]], gen, fp)}}

Enqueue(id) ==
    IF Len(queue) < QueueCapacity
    THEN /\ queue' = Append(queue, id)
         /\ dropped' = dropped
         /\ droppedCount' = droppedCount
    ELSE /\ queue' = Append(Tail(queue), id)
         /\ dropped' = dropped \cup {Head(queue)}
         /\ droppedCount' = droppedCount + 1

Init ==
    /\ generation = 0
    /\ fingerprint = 0
    /\ queue = <<>>
    /\ resolverPhase = [r \in Resolvers |-> "idle"]
    /\ capturedGen = [r \in Resolvers |-> 0]
    /\ capturedFp = [r \in Resolvers |-> 0]
    /\ decisionGen = [r \in Resolvers |-> 0]
    /\ decisionFp = [r \in Resolvers |-> 0]
    /\ publisherPhase = "idle"
    /\ publisherBaseGen = 0
    /\ publisherBaseFp = 0
    /\ batch = <<>>
    /\ pendingValid = {}
    /\ pendingRejected = {}
    /\ nextOutcomeId = 1
    /\ outcomeById =
         [id \in OutcomeIds |->
            [id |-> id,
             gen |-> 0,
             fp |-> 0,
             bindingOK |-> FALSE,
             source |-> "invalid"]]
    /\ dropped = {}
    /\ rejected = {}
    /\ applied = {}
    /\ droppedCount = 0
    /\ rejectedCount = 0
    /\ appliedCount = 0
    /\ updateCount = 0
    /\ failureSeen = FALSE
    /\ commitLog = <<>>

ResolveStart(r) ==
    /\ r \in Resolvers
    /\ resolverPhase[r] = "idle"
    /\ publisherPhase = "idle"
    /\ resolverPhase' = [resolverPhase EXCEPT ![r] = "evaluating"]
    /\ capturedGen' = [capturedGen EXCEPT ![r] = generation]
    /\ capturedFp' = [capturedFp EXCEPT ![r] = fingerprint]
    /\ UNCHANGED <<generation,
                   fingerprint,
                   queue,
                   decisionGen,
                   decisionFp,
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

ResolveFinish(r) ==
    /\ r \in Resolvers
    /\ resolverPhase[r] = "evaluating"
    /\ resolverPhase' = [resolverPhase EXCEPT ![r] = "ready"]
    /\ decisionGen' = [decisionGen EXCEPT ![r] = capturedGen[r]]
    /\ decisionFp' = [decisionFp EXCEPT ![r] = capturedFp[r]]
    /\ UNCHANGED <<generation,
                   fingerprint,
                   queue,
                   capturedGen,
                   capturedFp,
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

Observe(r) ==
    /\ r \in Resolvers
    /\ resolverPhase[r] = "ready"
    /\ publisherPhase \in {"idle", "publish_locked"}
    /\ nextOutcomeId \in OutcomeIds
    /\ LET o == LocalOutcome(r)
       IN /\ Enqueue(o.id)
          /\ outcomeById' = [outcomeById EXCEPT ![o.id] = o]
    /\ resolverPhase' = [resolverPhase EXCEPT ![r] = "idle"]
    /\ nextOutcomeId' = nextOutcomeId + 1
    /\ UNCHANGED <<generation,
                   fingerprint,
                   capturedGen,
                   capturedFp,
                   decisionGen,
                   decisionFp,
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

\* Models a DecisionDecomposition supplied from another same-numbered branch.
\* The concrete observe method accepts such a value, but publication must reject
\* it because its fingerprint does not equal the current snapshot fingerprint.
ObserveForeign ==
    /\ publisherPhase \in {"idle", "publish_locked"}
    /\ nextOutcomeId \in OutcomeIds
    /\ LET o == ForeignOutcome
       IN /\ Enqueue(o.id)
          /\ outcomeById' = [outcomeById EXCEPT ![o.id] = o]
    /\ nextOutcomeId' = nextOutcomeId + 1
    /\ UNCHANGED <<generation,
                   fingerprint,
                   resolverPhase,
                   capturedGen,
                   capturedFp,
                   decisionGen,
                   decisionFp,
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

\* Models an outcome whose redundant decision/model binding is inconsistent
\* even though its generation and fingerprint equal the current snapshot.
ObserveInvalid ==
    /\ publisherPhase \in {"idle", "publish_locked"}
    /\ nextOutcomeId \in OutcomeIds
    /\ LET o == InvalidOutcome
       IN /\ Enqueue(o.id)
          /\ outcomeById' = [outcomeById EXCEPT ![o.id] = o]
    /\ nextOutcomeId' = nextOutcomeId + 1
    /\ UNCHANGED <<generation,
                   fingerprint,
                   resolverPhase,
                   capturedGen,
                   capturedFp,
                   decisionGen,
                   decisionFp,
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

PublishBegin ==
    /\ publisherPhase = "idle"
    /\ publisherPhase' = "publish_locked"
    /\ publisherBaseGen' = generation
    /\ publisherBaseFp' = fingerprint
    /\ batch' = <<>>
    /\ pendingValid' = {}
    /\ pendingRejected' = {}
    /\ UNCHANGED <<generation,
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
                   failureSeen,
                   commitLog>>

PublishCut ==
    /\ publisherPhase = "publish_locked"
    /\ queue # <<>>
    /\ publisherPhase' = "building"
    /\ batch' = queue
    /\ pendingValid' = {}
    /\ pendingRejected' = {}
    /\ UNCHANGED <<generation,
                   fingerprint,
                   queue,
                   resolverPhase,
                   capturedGen,
                   capturedFp,
                   decisionGen,
                   decisionFp,
                   publisherBaseGen,
                   publisherBaseFp,
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

\* The concrete empty-queue call returns while still holding both locks and
\* does not invoke successor construction.
PublishEmpty ==
    /\ publisherPhase = "publish_locked"
    /\ queue = <<>>
    /\ publisherPhase' = "idle"
    /\ batch' = <<>>
    /\ pendingValid' = {}
    /\ pendingRejected' = {}
    /\ UNCHANGED <<generation,
                   fingerprint,
                   queue,
                   resolverPhase,
                   capturedGen,
                   capturedFp,
                   decisionGen,
                   decisionFp,
                   publisherBaseGen,
                   publisherBaseFp,
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

\* Successful construction changes only publisher-local proof state.  The
\* externally visible snapshot and queue still denote the pre-call state.
PublishBuild ==
    /\ publisherPhase = "building"
    /\ publisherPhase' = "ready"
    /\ LET valid ==
             ValidIdsAt(batch, publisherBaseGen, publisherBaseFp)
       IN /\ pendingValid' = valid
          /\ pendingRejected' = Ids(batch) \ valid
    /\ UNCHANGED <<generation,
                   fingerprint,
                   queue,
                   resolverPhase,
                   capturedGen,
                   capturedFp,
                   decisionGen,
                   decisionFp,
                   publisherBaseGen,
                   publisherBaseFp,
                   batch,
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

\* Abstracts an exception raised by successor construction.  The queue,
\* snapshot, disposition accounting, and resolution state are unchanged.
PublishFail ==
    /\ publisherPhase = "building"
    /\ ValidIdsAt(batch, publisherBaseGen, publisherBaseFp) # {}
    /\ publisherPhase' = "idle"
    /\ batch' = <<>>
    /\ pendingValid' = {}
    /\ pendingRejected' = {}
    /\ failureSeen' = TRUE
    /\ UNCHANGED <<generation,
                   fingerprint,
                   queue,
                   resolverPhase,
                   capturedGen,
                   capturedFp,
                   decisionGen,
                   decisionFp,
                   publisherBaseGen,
                   publisherBaseFp,
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

PublishCommit ==
    /\ publisherPhase = "ready"
    /\ LET event == CurrentCommitEvent
       IN /\ queue' = <<>>
          /\ applied' = applied \cup pendingValid
          /\ rejected' = rejected \cup pendingRejected
          /\ appliedCount' =
                appliedCount + Cardinality(pendingValid)
          /\ rejectedCount' =
                rejectedCount + Cardinality(pendingRejected)
          /\ commitLog' = Append(commitLog, event)
          /\ IF pendingValid # {}
             THEN /\ generation' = generation + 1
                  /\ fingerprint' = fingerprint + 1
                  /\ updateCount' = updateCount + 1
             ELSE /\ UNCHANGED <<generation, fingerprint, updateCount>>
    /\ publisherPhase' = "idle"
    /\ batch' = <<>>
    /\ pendingValid' = {}
    /\ pendingRejected' = {}
    /\ UNCHANGED <<resolverPhase,
                   capturedGen,
                   capturedFp,
                   decisionGen,
                   decisionFp,
                   publisherBaseGen,
                   publisherBaseFp,
                   nextOutcomeId,
                   outcomeById,
                   dropped,
                   droppedCount,
                   failureSeen>>

Next ==
    \/ \E r \in Resolvers : ResolveStart(r)
    \/ \E r \in Resolvers : ResolveFinish(r)
    \/ \E r \in Resolvers : Observe(r)
    \/ ObserveForeign
    \/ ObserveInvalid
    \/ PublishBegin
    \/ PublishCut
    \/ PublishEmpty
    \/ PublishBuild
    \/ PublishFail
    \/ PublishCommit

SafetySpec == Init /\ [][Next]_vars

PublishAdvance ==
    \/ PublishCut
    \/ PublishEmpty
    \/ PublishBuild
    \/ PublishFail
    \/ PublishCommit

\* Fairness is limited to a publisher that has already acquired a lock.
\* It does not assert that clients eventually resolve, observe, or publish.
LiveSpec ==
    SafetySpec
    /\ WF_vars(PublishAdvance)

TypeOK ==
    /\ generation \in Nat
    /\ fingerprint \in Nat
    /\ queue \in Seq(OutcomeIds)
    /\ resolverPhase \in [Resolvers -> ResolverPhases]
    /\ capturedGen \in [Resolvers -> Nat]
    /\ capturedFp \in [Resolvers -> Nat]
    /\ decisionGen \in [Resolvers -> Nat]
    /\ decisionFp \in [Resolvers -> Nat]
    /\ publisherPhase \in PublisherPhases
    /\ publisherBaseGen \in Nat
    /\ publisherBaseFp \in Nat
    /\ batch \in Seq(OutcomeIds)
    /\ pendingValid \subseteq OutcomeIds
    /\ pendingRejected \subseteq OutcomeIds
    /\ nextOutcomeId \in 1..(MaxOutcomeId + 1)
    /\ outcomeById \in [OutcomeIds -> OutcomeType]
    /\ dropped \subseteq OutcomeIds
    /\ rejected \subseteq OutcomeIds
    /\ applied \subseteq OutcomeIds
    /\ droppedCount \in Nat
    /\ rejectedCount \in Nat
    /\ appliedCount \in Nat
    /\ updateCount \in Nat
    /\ failureSeen \in BOOLEAN
    /\ commitLog \in Seq(CommitEventType)

BoundedQueue ==
    /\ Len(queue) <= QueueCapacity
    /\ Len(batch) <= QueueCapacity

UniqueOutcomeIds ==
    SeqInjective(queue)

DispositionDisjoint ==
    /\ dropped \cap rejected = {}
    /\ dropped \cap applied = {}
    /\ rejected \cap applied = {}
    /\ Ids(queue) \cap dropped = {}
    /\ Ids(queue) \cap rejected = {}
    /\ Ids(queue) \cap applied = {}

OutcomeAccounting ==
    CreatedIds = Ids(queue) \cup dropped \cup rejected \cup applied

CountAccuracy ==
    /\ droppedCount = Cardinality(dropped)
    /\ rejectedCount = Cardinality(rejected)
    /\ appliedCount = Cardinality(applied)

SnapshotGenerationAgreement ==
    /\ generation = fingerprint
    /\ generation = updateCount

PublisherLockInvariant ==
    /\ (publisherPhase = "idle"
          => /\ batch = <<>>
             /\ pendingValid = {}
             /\ pendingRejected = {})
    /\ (publisherPhase \in {"publish_locked", "building", "ready"}
          => /\ generation = publisherBaseGen
             /\ fingerprint = publisherBaseFp)
    /\ (publisherPhase = "publish_locked"
          => /\ batch = <<>>
             /\ pendingValid = {}
             /\ pendingRejected = {})
    /\ (publisherPhase = "building"
          => /\ batch = queue
             /\ batch # <<>>
             /\ pendingValid = {}
             /\ pendingRejected = {})
    /\ (publisherPhase = "ready"
          => /\ batch = queue
             /\ batch # <<>>
             /\ pendingValid =
                  ValidIdsAt(batch, publisherBaseGen, publisherBaseFp)
             /\ pendingRejected = Ids(batch) \ pendingValid)

ResolutionSnapshotConsistency ==
    \A r \in Resolvers :
        /\ (resolverPhase[r] \in {"evaluating", "ready"}
              => /\ capturedGen[r] <= generation
                 /\ capturedFp[r] = capturedGen[r])
        /\ (resolverPhase[r] = "ready"
              => /\ decisionGen[r] = capturedGen[r]
                 /\ decisionFp[r] = capturedFp[r])

CorrectCommitClassification ==
    \A k \in 1..Len(commitLog) :
        LET e == commitLog[k]
        IN /\ e.cut = e.applied \cup e.rejected
           /\ e.applied \cap e.rejected = {}
           /\ \A id \in e.cut :
                 /\ id \in CreatedIds
                 /\ (id \in e.applied)
                       = IsValidAt(outcomeById[id], e.baseGen, e.baseFp)

ForeignApplicationExclusion ==
    \A id \in applied :
        outcomeById[id].source = "local"

\* Explicit occurrence history makes identity total and deterministic.  The
\* source-shape fact is the exact strengthening used for foreign application
\* exclusion.
CreatedOutcomeIdentity ==
    \A id \in CreatedIds :
        outcomeById[id].id = id

CreatedCanonicalValidity ==
    \A id \in CreatedIds :
        (outcomeById[id].bindingOK
          /\ outcomeById[id].fp = outcomeById[id].gen)
            => outcomeById[id].source = "local"

Safety ==
    /\ TypeOK
    /\ BoundedQueue
    /\ UniqueOutcomeIds
    /\ DispositionDisjoint
    /\ OutcomeAccounting
    /\ CountAccuracy
    /\ SnapshotGenerationAgreement
    /\ PublisherLockInvariant
    /\ ResolutionSnapshotConsistency
    /\ CorrectCommitClassification
    /\ ForeignApplicationExclusion

InductiveStrengthening ==
    /\ CreatedOutcomeIdentity
    /\ CreatedCanonicalValidity

InductiveSafety ==
    /\ Safety
    /\ InductiveStrengthening

PublicationEventuallyReturns ==
    [](publisherPhase # "idle" => <> (publisherPhase = "idle"))

=============================================================================
