"""Run the full reproduction pipeline end-to-end.

  uv run python src/run_all.py [paper|competing_risks]

Steps: build dataset -> Cox survival models -> LIML revenue/employment models.
"""
import sys
import build_dataset
import survival_cox
import liml_models

if __name__ == "__main__":
    definition = sys.argv[1] if len(sys.argv) > 1 else "paper"
    print("#" * 70, f"\n# 1. BUILD DATASET (survival definition: {definition})\n", "#" * 70)
    build_dataset.main(definition)
    print("\n" + "#" * 70, "\n# 2. COX PROPORTIONAL-HAZARDS MODELS (Tables II-III)\n", "#" * 70)
    survival_cox.main()
    print("\n" + "#" * 70, "\n# 3. LIML-IV REVENUE & EMPLOYMENT MODELS (Tables IV-VII)\n", "#" * 70)
    liml_models.main()
    print("\nDone. Result tables written to output/tables/.")
