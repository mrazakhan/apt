# 00 — Project Vision

## North-star objective

> **Understand why a post-training experiment failed to improve held-out accuracy and actively determine what experiment or intervention should happen next.**

The important word is **accuracy**. Training reward, KL, entropy, gradient norms, GRPO group statistics, verifier outputs, and trajectories are evidence. They are not the final objective.

## Diagnostic philosophy

APT should not jump directly from a suspicious metric to a diagnosis. The intended loop is:

1. Observe that trusted held-out accuracy did not improve.
2. Identify where useful learning may first have broken down.
3. Form competing hypotheses.
4. Gather supporting and contradicting evidence.
5. Design a diagnostic probe whose outcomes differ under those hypotheses.
6. Run the probe and update the diagnosis.
7. Choose an intervention.
8. Measure whether held-out accuracy improves.

## Failure stages

APT should reason across environment/task → reward/verifier → GRPO learning signal → policy loss → gradients → optimizer/update → fresh rollouts → behavior/capability execution → held-out generalization.

## Diagnostic probes

A **Diagnostic Probe** is a deliberately modified experiment designed to isolate one causal bottleneck.

Examples include reducing task difficulty while preserving capability, replacing the verifier with a trusted oracle, providing a relevant rule explicitly, using a stronger starting checkpoint, replaying a fixed rollout batch through optimization, or restoring the original environment/harness.

Each probe should eventually specify the hypothesis tested, competing hypothesis, controlled variables, changed variable, predicted result under each hypothesis, observed result, and conclusion.

## Research memory

Research notes live under [research_notes](research_notes/README.md). Interesting paper results do not automatically become implementation commitments; durable decisions should later be promoted into the relevant architecture/tutorial chapter.