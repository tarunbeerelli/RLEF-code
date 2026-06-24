from dotenv import load_dotenv

load_dotenv()
from rlef.data import load_apps_split
from rlef.reward import execution_reward
from rlef.tools import execute

problems = load_apps_split(
    "data/raw/APPS", split="train", difficulties=["introductory"]
)
printed = 0

for p in problems[:200]:
    if not p.solutions:
        continue

    result = execution_reward(
        code=p.solutions[0],
        inputs=p.inputs,
        outputs=p.outputs,
        fn_name=p.fn_name,
        reward_type="continuous",
        shaped=False,
        difficulty=p.difficulty,
    )

    if result.passed != result.total and printed < 3:
        print("\n====================================")
        print(f"FAILING PROBLEM ID: {getattr(p, 'problem_id', 'Unknown')}")
        print(f"Function Name (fn_name): {p.fn_name}")
        print(f"Passed test cases: {result.passed}/{result.total}")
        print(f"Error Types logged: {result.error_types}")
        print("====================================")

        # Let's manually run the exact first test case to see the raw error or stdout
        inp = p.inputs[0]
        expected_out = p.outputs[0]
        code = p.solutions[0]

        if p.fn_name:
            test_code = (
                f"import json\n"
                f"{code}\n"
                f"raw_inp = json.loads({repr(inp)})\n"
                f"args = raw_inp if isinstance(raw_inp, list) else [raw_inp]\n"
                f"print('ACTUAL_OUTPUT:', {p.fn_name}(*args))\n"
            )
        else:
            test_code = (
                f"import builtins, sys, io\n"
                f"sys.stdin = io.StringIO({repr(inp)} + '\\n')\n"
                f"_real_readline = sys.stdin.readline\n"
                f"sys.stdin.readline = lambda *a, **kw: _real_readline(*a, **kw) or '\\n'\n"
                f"builtins.input = lambda *a, **kw: sys.stdin.readline().rstrip()\n\n"
                f"{code}\n"
            )

        res = execute(test_code, timeout=5)
        print("--- RAW EXECUTOR ERROR ---")
        print(res.error)
        print("--- RAW EXECUTOR STDOUT ---")
        print(res.stdout)
        print("--- EXPECTED OUTPUT STR ---")
        print(repr(expected_out))
        printed += 1
