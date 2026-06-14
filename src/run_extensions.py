"""Run the methodological extensions end-to-end.

  uv run python src/run_extensions.py

Steps: build extended dataset -> competing-risks models -> MICE pooled Cox ->
ML survival + interaction discovery. See docs/EXTENSIONS.md for the writeup.
"""
import build_extended
import competing_risks
import imputation
import ml_survival

if __name__ == "__main__":
    print("#" * 70, "\n# 1. BUILD EXTENDED DATASET (competing risks + team capital)\n", "#" * 70)
    build_extended.main()
    print("\n" + "#" * 70, "\n# 2a. COMPETING-RISKS MODELS\n", "#" * 70)
    competing_risks.main()
    print("\n" + "#" * 70, "\n# 2b. MULTIPLE IMPUTATION (MICE) + POOLED COX\n", "#" * 70)
    imputation.main()
    print("\n" + "#" * 70, "\n# 4. ML SURVIVAL + INTERACTION DISCOVERY\n", "#" * 70)
    ml_survival.main()
    print("\nDone. See docs/EXTENSIONS.md and output/tables/.")
