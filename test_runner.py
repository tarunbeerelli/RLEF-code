from dotenv import load_dotenv

load_dotenv()
from collections import Counter

from rlef.data import load_apps_split
from rlef.reward import execution_reward

# 1. Load the dataset cleanly
problems = load_apps_split(
    "data/raw/APPS", split="train", difficulties=["introductory"]
)
distribution = Counter()

problems_passed = 0
total_problems_evaluated = 0

# 2. Force a clean, hard-limited slice to exactly 500 problems
target_slice = problems[:500]

print(f"Evaluating exactly {len(target_slice)} problems sequentially...")

for p in target_slice:
    # Ensure there is an actual code solution string available to evaluate
    if not p.solutions or not isinstance(p.solutions, list) or len(p.solutions) == 0:
        continue

    result = execution_reward(
        code=p.solutions[0],  # <--- FIXED: Safely extract the first string solution
        inputs=p.inputs,
        outputs=p.outputs,
        fn_name=p.fn_name,
        reward_type="continuous",
        shaped=False,
        difficulty=p.difficulty,
    )

    # Track the problem as a SUCCESS only if it passes all its test case boundaries
    if result.passed == result.total and result.total > 0:
        problems_passed += 1

    total_problems_evaluated += 1

    # Track error distribution strings
    for err in result.error_types:
        distribution[err] += 1

print("\n=================== PER-PROBLEM RESULTS ======================")
print(f"Problems Evaluated: {total_problems_evaluated}")
print(
    f"Problems fully passed: {problems_passed}/{total_problems_evaluated} ({problems_passed/total_problems_evaluated:.1%})"
)
print("Error Distribution Breakdown:", dict(distribution))
print("==============================================================")
