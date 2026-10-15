# Repo2RLEnv as the Verifiable-Coding Wedge

**Status: feasibility and interface design only.**

## Roles
- Repo2RLEnv supplies Harbor-compatible coding tasks and quality controls.
- Harbor runs actual coding-agent tasks and reports verifier outcomes.
- TRL SFTTrainer learns from *verified demonstrations*, not task instructions alone.
- TRL GRPOTrainer supplies the initial teaching vehicle for simple reward-graded completions. Multi-step agent rollouts require a deliberate compatible training bridge.
- APT ingests lineage, verifies evidence, localizes failures and selects diagnostic probes.

## Three distinct data contracts
**SFT example**: prompt, validated completion, provenance and train/eval split. Obtain correct responses from validated demonstrations/oracle-derived traces with leakage checks.

**Single-turn GRPO example**: prompt, multiple model completions, deterministic safe reward and group metadata.

**Coding-agent rollout**: task ID, environment/harness revision, policy version, actions/tool calls, response token masks, verifier version, result and execution metadata.

## Integration gates
1. Inspect a small published Repo2RLEnv task set and document task layout, licenses, evaluation labels and private artifacts.
2. Validate positive (oracle) and negative (no-op) verifier controls; ensure grading can distinguish behavior.
3. Generate baseline coding-agent rollouts with a fixed checkpoint, budget and deterministic evaluation harness.
4. Validate the RL training interface, policy token accounting and model synchronization before connecting to real GRPO.
5. Introduce controlled defects (difficulty, verifier, masking or optimizer) and compare APT recommendations against known faults and trusted held-out success.

## Measurement policy
Report reward informativeness, group tied rates, KL definitions, clipping, gradient/update norms when available, task success by repository/capability slice, and cost. Missing values are unknown, not zero. A verified oracle task is not automatically easy enough for current policy to learn.

## Why this wedge
A coding-task benchmark provides an external success criterion against which APT's diagnostic quality can be tested. APT adds value by determining *why* an agent+trainer did not improve and proposing the next controlled experiment, rather than repeating task generation or merely plotting metrics.

## References
- https://github.com/huggingface/Repo2RLEnv
- https://huggingface.github.io/Repo2RLEnv/quickstart/
- https://huggingface.co/docs/trl/sft_trainer
- https://huggingface.co/docs/trl/grpo_trainer
