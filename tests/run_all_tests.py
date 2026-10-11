import subprocess
import sys
from pathlib import Path

test_dir = Path(__file__).resolve().parent
test_files = sorted([f for f in test_dir.glob("test_*.py") if f.name != "run_all_tests.py"])

print("=" * 70)
print(f"RUNNING ALL TEST SUITES ({len(test_files)} test files)")
print("=" * 70)

all_passed = True
results = []

for tf in test_files:
    print(f"\n[RUNNING] {tf.name}...", flush=True)
    res = subprocess.run([sys.executable, str(tf)], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[PASS] {tf.name}", flush=True)
        results.append((tf.name, "PASS", res.stdout.strip()))
    else:
        print(f"[FAIL] {tf.name}", flush=True)
        print(res.stderr, flush=True)
        all_passed = False
        results.append((tf.name, "FAIL", res.stderr.strip()))

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
for name, status, _ in results:
    print(f"  [{status}] {name}")
print("=" * 70)

if not all_passed:
    sys.exit(1)
print("\nALL TEST SUITES PASSED CLEANLY!")
