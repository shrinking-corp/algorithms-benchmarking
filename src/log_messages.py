"""
Centralized log message templates for logging.
"""

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_SIZES = (
    "[DATASET]       train={train}  val={val}"
)

# ── Client level ──────────────────────────────────────────────────────────────
CLIENT_START = (
    "[CLIENT START]  llm={llm}"
)
CLIENT_END = (
    "[CLIENT END]    llm={llm}"
)

# ── Model level ───────────────────────────────────────────────────────────────
MODEL_START = (
    "[MODEL START]   llm={llm}  model={model}"
)
MODEL_END = (
    "[MODEL END]     llm={llm}  model={model}"
)
BASELINE_DONE = (
    "[BASELINE]      llm={llm}  model={model}  avg_codebleu={score:.4f}"
)

# ── Algorithm level ───────────────────────────────────────────────────────────
ALG_START = (
    "[ALG START]     llm={llm}  model={model}  alg={alg}"
)
ALG_SKIPPED = (
    "[ALG SKIPPED]   llm={llm}  model={model}  alg={alg}  (already completed - .done marker found)"
)
ALG_END = (
    "[ALG END]       llm={llm}  model={model}  alg={alg}"
)

# ── PSO: checkpoint resume ─────────────────────────────────────────────────────
PSO_RESUMED = (
    "[PSO RESUMED]   llm={llm}  model={model}  alg={alg}"
    "  from_iter={from_iter}/{n_iter}  gbest={gbest:.4f}"
)

# ── PSO: per-particle evaluation ──────────────────────────────────────────────
# Fired for every particle that does NOT improve the global best.
PSO_PARTICLE = (
    "[PSO PARTICLE]  llm={llm}  model={model}  alg={alg}"
    "  iter={iter}/{n_iter}  agent={agent}/{n_agents}  fitness={fitness:.4f}"
)

# Fired when a particle improves the global best.
PSO_NEW_BEST = (
    "[PSO NEW BEST]  llm={llm}  model={model}  alg={alg}"
    "  iter={iter}/{n_iter}  agent={agent}/{n_agents}"
    "  fitness={fitness:.4f}  params={params}"
)

# ── PSO: end of iteration ─────────────────────────────────────────────────────
PSO_ITER_DONE = (
    "[PSO ITER DONE] llm={llm}  model={model}  alg={alg}"
    "  iter={iter}/{n_iter}  gbest={gbest:.4f}"
)

# ── PSO: final result ─────────────────────────────────────────────────────────
PSO_RESULT = (
    "[PSO RESULT]    llm={llm}  model={model}  alg={alg}"
    "  best_fitness={fitness:.4f}  params={params}"
)

# ── Final validation score ─────────────────────────────────────────────────────
FINAL_RESULT = (
    "[FINAL RESULT]  llm={llm}  model={model}  alg={alg}"
    "  baseline={baseline:.4f}  final={final:.4f}  retention={retention:.4f}"
)
