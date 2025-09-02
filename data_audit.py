# data_audit.py
import os, re, pandas as pd, numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Reuse configuration/file name
INPUT_FILE = "SUMMARY PAPER CONSO 2016-2025.xlsx"
OUTPUT_FOLDER = "./reports"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Import helper logic from the main script by reading it dynamically
# to avoid tight coupling we duplicate small utilities here if import fails.
try:
    from dakri_weekly_report import parse_month_year, normalize_type
except Exception:
    def parse_month_year(sheet_name: str):
        s = sheet_name.lower()
        month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"june":6,"jul":7,"july":7,
                     "aug":8,"august":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12}
        month = None; year = None
        for k,v in month_map.items():
            if k in s: month=v; break
        m4 = re.search(r"20\d{2}", s)
        if m4:
            year = int(m4.group(0))
        else:
            m2 = re.search(r"(\d{2})(?!.*\d)", s)
            if m2:
                y = int(m2.group(1))
                year = 2000 + y
        return year, month

    def normalize_type(text: Any):
        if not isinstance(text, str):
            return None
        s = str(text).lower().replace("-"," ").replace("_"," ")
        # Minimal heuristic
        if "white top" in s or "mepwhite" in s or "propack" in s or "ks" in s:
            return "WHITE TOP"
        if "test" in s:
            return "TESTLINER"
        if "liner" in s:
            if any(x in s for x in ["heavy","220","235","250"]):
                return "HEAVY LINER"
            return "LINER"
        if "flute" in s or "fluting" in s:
            if any(x in s for x in ["heavy","ultra","140","150"]):
                return "HEAVY FLUTE"
            if "super" in s:
                return "SUPER FLUTE"
            return "FLUTE"
        return None


def find_ttl_column(df: pd.DataFrame) -> str:
    header_probe_rows = df.head(3)
    ttl_positions = np.argwhere(header_probe_rows.applymap(lambda x: isinstance(x, str) and x.strip().lower()=="ttl").values)
    if ttl_positions.size > 0:
        ttl_marker_row, ttl_marker_col = ttl_positions[0]
        if ttl_marker_col + 1 < len(df.columns):
            return df.columns[ttl_marker_col + 1]
    # Fallback by keywords
    candidate_cols = [
        c for c in df.columns
        if isinstance(c, str) and re.search(r"\b(total|ttl|sum)\b", c.lower())
    ]
    if candidate_cols:
        return candidate_cols[0]
    # Last numeric or last column
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols[-1] if numeric_cols else df.columns[-1]


def coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d,.-]", "", regex=True)
    )
    coerced = pd.to_numeric(
        cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    if coerced.isna().all():
        coerced = pd.to_numeric(cleaned, errors="coerce")
    return coerced


def main() -> int:
    xls = pd.ExcelFile(INPUT_FILE)

    skip_rows: List[Dict[str, Any]] = []
    accepted_rows: List[Dict[str, Any]] = []

    for sh in xls.sheet_names:
        year, month = parse_month_year(sh)
        if not year or not month:
            continue
        df = pd.read_excel(INPUT_FILE, sheet_name=sh, header=0)

        descr_col_guess = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        if df[descr_col_guess].dtype != object:
            string_cols = [c for c in df.columns if df[c].dtype == object]
            if string_cols:
                descr_col_guess = string_cols[0]

        ttl_col = find_ttl_column(df)
        ttl_nums = coerce_numeric(df[ttl_col])
        # Fallback: row-wise max of numeric-looking columns to detect hidden totals
        num_all = pd.DataFrame({c: coerce_numeric(df[c]) for c in df.columns})
        row_max = num_all.max(axis=1, skipna=True)

        for i, row in df.iterrows():
            descr = row.get(descr_col_guess, None)
            num = ttl_nums.iloc[i]
            # prefer row_max if larger (and not nan)
            try:
                if pd.notna(row_max.iloc[i]) and (pd.isna(num) or row_max.iloc[i] > num):
                    num = row_max.iloc[i]
            except Exception:
                pass
            reason = None
            ptype = normalize_type(descr)
            if pd.isna(num) or float(num) == 0:
                reason = "no_numeric_total"
            elif not ptype:
                reason = "unmapped_description"

            rec_base = {
                "sheet": sh,
                "year": year,
                "month": month,
                "raw_description": descr,
                "raw_total": row.get(ttl_col, None),
                "total_numeric": float(num) if pd.notna(num) else None,
                "detected_type": ptype,
            }
            if reason:
                rec = {**rec_base, "skip_reason": reason}
                skip_rows.append(rec)
            else:
                rec = {**rec_base, "paper_type": ptype}
                accepted_rows.append(rec)

    skipped_df = pd.DataFrame(skip_rows)
    accepted_df = pd.DataFrame(accepted_rows)

    # Missing months per type based on accepted data
    gaps_df = pd.DataFrame()
    if not accepted_df.empty:
        accepted_df["date"] = pd.to_datetime(dict(year=accepted_df.year, month=accepted_df.month, day=1))
        monthly = accepted_df.groupby(["date","paper_type"])['total_numeric'].sum().reset_index()
        all_types = sorted(monthly['paper_type'].unique())
        full_range = pd.date_range(monthly['date'].min(), monthly['date'].max(), freq='MS')
        grid = (
            pd.MultiIndex.from_product([full_range, all_types], names=["date","paper_type"])
            .to_frame(index=False)
        )
        joined = grid.merge(monthly, how='left', on=["date","paper_type"])
        gaps_df = joined[joined['total_numeric'].isna()].copy()

    # Summaries
    report_paths = []
    if not skipped_df.empty:
        p = Path(OUTPUT_FOLDER) / "audit_skipped_rows.csv"
        skipped_df.to_csv(p, index=False)
        report_paths.append(str(p))
        p2 = Path(OUTPUT_FOLDER) / "audit_skip_reasons_summary.csv"
        skipped_df.groupby(["skip_reason"]).size().reset_index(name="count").to_csv(p2, index=False)
        report_paths.append(str(p2))
        p3 = Path(OUTPUT_FOLDER) / "audit_unmapped_descriptions_top.csv"
        (skipped_df[skipped_df['skip_reason']=="unmapped_description"]
         .groupby(["raw_description"]).size().reset_index(name="count")
         .sort_values("count", ascending=False).head(100).to_csv(p3, index=False))
        report_paths.append(str(p3))
    if not accepted_df.empty:
        p4 = Path(OUTPUT_FOLDER) / "audit_accepted_rows_head.csv"
        accepted_df.head(200).to_csv(p4, index=False)
        report_paths.append(str(p4))
    if gaps_df is not None and not gaps_df.empty:
        p5 = Path(OUTPUT_FOLDER) / "audit_missing_months_by_type.csv"
        gaps_df.to_csv(p5, index=False)
        report_paths.append(str(p5))

    print("Audit files written:")
    for r in report_paths:
        print(" -", r)
    if not report_paths:
        print("No audit issues detected; nothing to write.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
