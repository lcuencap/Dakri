import sys
import re
from pathlib import Path

import pandas as pd


def main() -> int:
    workbook_path = Path("SUMMARY PAPER CONSO 2016-2025.xlsx")
    print(f"Workbook: {workbook_path.resolve()}")
    if not workbook_path.exists():
        print("File not found.")
        return 1

    try:
        xls = pd.ExcelFile(workbook_path)
    except Exception as exc:
        print("Failed to open workbook:", repr(exc))
        return 2

    print("Sheets (count=", len(xls.sheet_names), "):", xls.sheet_names)

    for sheet_name in xls.sheet_names:
        print("\n===== Sheet:", sheet_name, "=====")
        try:
            # Read a small preview with headers inferred
            df = pd.read_excel(workbook_path, sheet_name=sheet_name, header=0, nrows=10)
        except Exception as exc:
            print("Read error:", repr(exc))
            continue

        print("Columns:", list(df.columns))
        print("DTypes:\n", df.dtypes.to_string())

        # Candidate total columns (English-only)
        candidate_cols = [
            c for c in df.columns
            if isinstance(c, str) and re.search(r"(total|ttl|sum)", c.lower())
        ]
        if not candidate_cols:
            # Fallback: last numeric column
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                candidate_cols = [numeric_cols[-1]]
        print("Candidate total columns:", candidate_cols)

        # Show a small value sample from the candidate total column (if any)
        if candidate_cols:
            col = candidate_cols[0]
            sample = df[col].head(8)
            print(f"\nSample values for '{col}':\n", sample.to_string(index=False))

        # Show first 5 rows compactly
        print("\nHead (first 5 rows):")
        print(df.head(5).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
