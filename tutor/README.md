# APT tutor in VS Code

This is **repository-backed session continuity** and a reusable custom agent, not a model running permanently in the background. You open VS Code, select the agent, and it reads the current checked-out files. Progress changes are saved **only when you approve them**.

## Start
1. Check out the branch containing this directory (`feat/vscode-apt-tutor` until merged) and open the **APT repository root** in VS Code.
2. Install/sign in to a VS Code chat harness that supports workspace custom agents, such as GitHub Copilot; open Chat, select **APT Tutor** in the agents dropdown (refresh/reload window if needed).
3. Ask: **"Resume my lab using tutor/progress.yaml. Ask one question at a time and don't write the TODOs for me."**
4. Keep the chat beside the notebook/code. You can ask for a code review, hint, explanation, test design, or diagnostic plotting scaffold without copying files into another chat.
5. End with **"Propose a checkpoint update and wait for my approval before writing it."**

## Files
- `AGENTS.md`: shared repo tutoring and safety boundaries.
- `.github/copilot-instructions.md`: Copilot repository-wide instructions.
- `.github/agents/apt-tutor.agent.md`: VS Code workspace custom agent.
- `tutor/progress.yaml`: explicit learner checkpoint. It starts as **not assessed**; being committed to GitHub does not imply mastery.
- `tutor/curriculum.yaml`: minimal current prerequisite graph; use `docs/03_connected_curriculum.md` as the authoritative narrative.
- `tutor/sessions/`: optional, learner-approved session notes.

## How to ask
- "Explain clipped ratios vs norm clipping, then quiz me before showing code."
- "Review this TODO implementation, but don't solve it. Give me one failing test to write."
- "Provide a Matplotlib render scaffold; I will compute the group tie rates and choose plots."
- "Inspect my training logs and ask me which competing hypotheses they distinguish."
- "Propose a 15-minute practice exercise for the current notebook."

## Limitations
- GitHub Copilot/custom agents require an available and authenticated provider in VS Code. This repo configuration does not install it for you.
- It does not automatically watch Colab, import remote outputs, or run long experiments. Save results back to the repo or supply them in the current chat.
- Automatic cross-session conversational memory is **not guaranteed**. The repo checkpoint provides auditable continuity.
- Until PR #1 is merged, this branch also includes its still-unvalidated skeleton notebooks. Verify with `uv sync --extra dev && uv run pytest -q`.
