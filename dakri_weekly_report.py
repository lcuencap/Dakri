# dakri_weekly_report.py
import os, re, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime

# =========================
# CONFIG
# =========================
INPUT_FILE = "SUMMARY PAPER CONSO 2016-2025.xlsx"
OUTPUT_FOLDER = "./reports"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Output unit for the purchase plan: 'kg' or 't' (metric tons)
OUTPUT_UNIT = "kg"
UNIT_FACTOR = 0.001 if OUTPUT_UNIT == "t" else 1.0
CAP_MULTIPLIER = 1.25  # coarse cap vs historical max; dynamic caps are also applied

# Data cleaning configuration
# Exclude COVID period (inclusive) from modeling/reporting if set
EXCLUDE_COVID = True
COVID_START = pd.Timestamp("2020-03-01")
COVID_END   = pd.Timestamp("2021-06-01")

# Anomaly removal using robust z-score (median/MAD). Points with |z|>THRESHOLD are removed per type
REMOVE_ANOMALIES = True
ROBUST_Z_THRESHOLD = 4.0

def clamp_forecast_to_recent_behavior(history: pd.Series, forecast: pd.Series) -> pd.Series:
    history = history.dropna()
    forecast = forecast.copy()
    if history.empty:
        return forecast.clip(lower=0)
    recent = history.tail(48)
    recent = recent[recent > 0]
    if recent.empty:
        return forecast.clip(lower=0)
    recent_median = float(np.median(recent.values))
    recent_std = float(np.std(recent.values))
    # Dynamic cap: allow up to median + 3*std, and at least 2x median
    dyn_cap = max(recent_median * 2.0, recent_median + 3.0 * recent_std)
    if np.isnan(dyn_cap) or dyn_cap <= 0:
        dyn_cap = recent_median * 2.0
    # Also respect global historical cap
    hist_cap = float(np.nanmax(history.values)) * CAP_MULTIPLIER if len(history) else dyn_cap
    cap_value = max(0.0, min(dyn_cap, hist_cap))
    return forecast.clip(lower=0, upper=cap_value)

# =========================
# MAPEO DE TIPOS
# =========================
TYPE_RULES = {
    "FLUTE": [
        "siam 105",
        "bayflute 112",
        "waraq 100",
        "wecycle 110",
    ],
    "SUPER FLUTE": [
        "siam cs 110",
        "siam (cs) 110",
        "bayflute 125",
    ],
    "HEAVY FLUTE": [
        "sappi ultraflute 140",
        "ultraflute 140",
        "waraq flute 150",
        "mpact flute 150",
        "bayplex 150",
    ],
    "LINER": [
        "naturkraft 135",
        "hipack 125",
        "naturkraft 125",
    ],
    "HEAVY LINER": [
        "sappi 220",
        "aryan 250",
        "hipack 235",
        "aryan 180",
        "sappi 250",
        "provantage 250",
    ],
    "WHITE TOP": [
        "mepwhite 140",
        "siam ks 140",
        "siam (ks) 140",
        "propack 140",
        "ik white 140",
        "ks white 140",
        "l&m white",
        "l &m white",
        "lee white",
        "mepco white 140",
        "mondi white 135",
        "norpac 125",
        "propack white 140",
    ],
    "TESTLINER": [
        "wecycle test 125",
        "wecycle test 140",
        "mpact test 140",
        "waraq test 120",
        "waraq test 125",
        "enstra 120",
        "interpac test 125",
        "keryas test 125",
        "kipas test 120",
        "mafta test 125",
        "mafta test 150",
        "mepco test 125",
        "mepco test 150",
        "rahul 120",
        "saim test ky 125",
        "saim test ky 150",
        "siam test 125",
        "union test 125",
        "union test 140",
        "waraq test 140",
    ],
}

def normalize_type(text):
    if not isinstance(text, str):
        return None
    s = text.lower()
    s = s.replace("-", " ").replace("_", " ")
    for t, patterns in TYPE_RULES.items():
        for p in patterns:
            if p in s:
                return t
    # Heuristic fallback by keywords when explicit pattern not found
    if "flute" in s or "fluting" in s or "hyfl" in s:
        # Heavy flute if explicit heavy cues or high gsm
        if "heavy" in s or "ultra" in s or re.search(r"\b1(4\d|50)\b", s):
            return "HEAVY FLUTE"
        if "super" in s:
            return "SUPER FLUTE"
        return "FLUTE"
    # White-top heuristics: any descriptor that clearly says WHITE
    if "white top" in s or "white-top" in s or " white" in s or s.startswith("white") or "-white" in s:
        return "WHITE TOP"
    if "liner" in s and ("heavy" in s or re.search(r"\b(180|18\d|19\d|200|210|220|235|250)\b", s)):
        return "HEAVY LINER"
    if "liner" in s:
        return "LINER"
    if "test" in s:
        return "TESTLINER"
    # Brand-specific fallbacks
    if "kraftpride" in s or re.search(r"\bkraft\b", s) or re.search(r"\bkp\b", s):
        # Assume kraft liner unless heavy gsm is present
        if re.search(r"\b(180|18\d|19\d|200|210|220|235|250)\b", s) or "heavy" in s:
            return "HEAVY LINER"
        return "LINER"
    if "provantage" in s:
        if re.search(r"\b(180|18\d|19\d|200|210|220|235|250)\b", s):
            return "HEAVY LINER"
        return "LINER"
    if "sappi" in s:
        # Disambiguate SAPPI variants
        if "ultraflute" in s or "hyfl" in s or re.search(r"\b1(4\d|50)\b", s):
            return "HEAVY FLUTE"
        if re.search(r"\b(180|18\d|19\d|200|210|220|235|250)\b", s) or "heavy" in s:
            return "HEAVY LINER"
        return "LINER"
    return None

# =========================
# CLEANING HELPERS
# =========================
def filter_covid_and_anomalies(monthly: pd.DataFrame) -> pd.DataFrame:
    cleaned = monthly.copy()
    # Exclude COVID period if enabled
    if EXCLUDE_COVID:
        cleaned = cleaned[(cleaned["date"] < COVID_START) | (cleaned["date"] > COVID_END)]
    # Remove anomalies per type using robust z-score
    if REMOVE_ANOMALIES and not cleaned.empty:
        def _remove_outliers(group: pd.DataFrame) -> pd.DataFrame:
            x = group["consumption"].astype(float)
            median = np.median(x)
            mad = np.median(np.abs(x - median))
            # Avoid division by zero; if constant series, keep as-is
            if mad == 0:
                return group
            robust_z = 0.6745 * (x - median) / mad
            mask = np.abs(robust_z) <= ROBUST_Z_THRESHOLD
            return group[mask]
        cleaned = cleaned.groupby("paper_type", group_keys=False).apply(_remove_outliers)
    return cleaned.reset_index(drop=True)

# =========================
# LECTURA DE EXCEL Y CONSOLIDACIÓN
# =========================
def parse_month_year(sheet_name):
    s = sheet_name.lower()
    month_map = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"june":6,"jul":7,"july":7,
                 "aug":8,"august":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12}
    month = None; year = None
    for k,v in month_map.items():
        if k in s: month=v; break
    # Prefer 4-digit years first
    m4 = re.search(r"20\d{2}", s)
    if m4:
        year = int(m4.group(0))
    else:
        # Fallback: last 2-digit number in the name (e.g., 'aug 23')
        m2 = re.search(r"(\d{2})(?!.*\d)", s)
        if m2:
            y = int(m2.group(1))
            year = 2000 + y
    return year, month

def load_data(file):
    xls = pd.ExcelFile(file)
    records = []
    for sh in xls.sheet_names:
        year, month = parse_month_year(sh)
        if not year or not month:
            continue
        df = pd.read_excel(file, sheet_name=sh, header=0)

        # Choose likely total column (English keywords only)
        # Prefer column immediately to the right of a cell labeled 'TTL' in the first header rows
        ttl_col = None
        header_probe_rows = df.head(3)
        ttl_positions = np.argwhere(header_probe_rows.applymap(lambda x: isinstance(x, str) and x.strip().lower()=="ttl").values)
        if ttl_positions.size > 0:
            # ttl_positions rows are [row_idx, col_idx]; pick the first and shift +1 col
            ttl_marker_row, ttl_marker_col = ttl_positions[0]
            if ttl_marker_col + 1 < len(df.columns):
                ttl_col = df.columns[ttl_marker_col + 1]
        if ttl_col is None:
            # Avoid false match on 'consumption' (contains 'sum'). Use word boundaries.
            candidate_cols = [
                c for c in df.columns
                if isinstance(c, str) and re.search(r"\b(total|ttl|sum)\b", c.lower())
            ]
            if candidate_cols:
                ttl_col = candidate_cols[0]
        if ttl_col is None:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            ttl_col = numeric_cols[-1] if numeric_cols else df.columns[-1]

        # Robust numeric coercion (handles thousands separators and commas)
        cleaned = (
            df[ttl_col]
            .astype(str)
            .str.replace(r"[^\d,.-]", "", regex=True)
        )
        coerced = pd.to_numeric(
            cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if coerced.isna().all():
            coerced = pd.to_numeric(cleaned, errors="coerce")
        df["_consumption_"] = coerced

        # Pick description column: first object dtype column
        # Description column is typically the second column (e.g., 'SIAM-FLUTE'). Fallback to first string column.
        descr_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        if df[descr_col].dtype != object:
            string_cols = [c for c in df.columns if c != "_consumption_" and df[c].dtype == object]
            if string_cols:
                descr_col = string_cols[0]

        for _, row in df.iterrows():
            descr = str(row[descr_col]) if pd.notna(row[descr_col]) else ""
            total_val = row["_consumption_"]
            if not pd.notna(total_val) or float(total_val) == 0:
                continue
            ptype = normalize_type(descr)
            if ptype:
                records.append({
                    "date": pd.Timestamp(year=year, month=month, day=1),
                    "paper_type": ptype,
                    "consumption": float(total_val),
                })
    return pd.DataFrame(records)

# =========================
# FORECASTING
# =========================
def forecast_series(ts, periods=12):
    ts = ts.asfreq("MS").fillna(0)
    best_model, best_fc, best_err = None, None, np.inf
    try:
        hw = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=12).fit()
        fc = hw.forecast(periods)
        err = mean_absolute_percentage_error(ts[-24:], hw.fittedvalues[-24:])
        if err < best_err: best_model, best_fc, best_err = "Holt-Winters", fc, err
    except: pass
    try:
        sar = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        fc = sar.forecast(periods)
        err = mean_absolute_percentage_error(ts[-24:], sar.fittedvalues[-24:])
        if err < best_err: best_model, best_fc, best_err = "SARIMA", fc, err
    except: pass
    return best_model, best_fc

# =========================
# REPORT PDF
# =========================
def generate_report(monthly, output_file, purchase_plan=None):
    with PdfPages(output_file) as pdf:
        # ----- Guide / Explanations -----
        plt.figure(figsize=(10,6))
        plt.axis("off")
        guide_lines = [
            "Report guide:",
            "1) Monthly Consumption by Type: Lines showing historical monthly totals per type.",
            "2) Share of Each Type: Area chart of each type's share of total consumption per month.",
            "3) Distribution by Type: Boxplot of monthly distributions (median, quartiles, outliers).",
            "4) Correlation Between Types: Heatmap of correlations across type time series.",
            "5) Descriptive Stats by Type: Count, mean, std, min, quartiles, max of monthly totals.",
            "6) Seasonal Decomposition: Trend/seasonal/residual components (when 2+ years available).",
            "7) Forecast 12 Months: Model-based forecast for the next 12 months per type (bounded by recent behavior).",
            f"8) Purchase Plan: Recommended quantities per month by type (units: {OUTPUT_UNIT}).",
        ]
        plt.text(0.02, 0.98, "\n".join(guide_lines), va="top", ha="left", fontsize=11)
        plt.title("How to read this report", y=1.05)
        pdf.savefig(); plt.close()
        # ----- Global Section -----
        plt.figure(figsize=(10,6))
        for t,g in monthly.groupby("paper_type"):
            plt.plot(g["date"], g["consumption"], label=t)
        plt.legend(); plt.title("Monthly Consumption by Type"); pdf.savefig(); plt.close()

        pivot = monthly.pivot(index="date", columns="paper_type", values="consumption").fillna(0)
        (pivot.div(pivot.sum(1), axis=0)).plot.area(figsize=(10,6))
        plt.title("Share of Each Type"); pdf.savefig(); plt.close()

        plt.figure(figsize=(10,6))
        sns.boxplot(x="paper_type", y="consumption", data=monthly)
        plt.title("Distribution by Type"); plt.xticks(rotation=45); pdf.savefig(); plt.close()

        corr = pivot.corr()
        plt.figure(figsize=(8,6)); sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0)
        plt.title("Correlation Between Types"); pdf.savefig(); plt.close()

        stats = monthly.groupby("paper_type")["consumption"].describe()
        plt.figure(figsize=(8,4)); plt.axis("off")
        plt.table(cellText=np.round(stats.values,2), colLabels=stats.columns, rowLabels=stats.index, loc="center")
        plt.title("Descriptive Stats by Type"); pdf.savefig(); plt.close()

        # ----- Individual Sections -----
        for t,g in monthly.groupby("paper_type"):
            ts = g.set_index("date")["consumption"].asfreq("MS")
            if len(ts)>24:
                try:
                    decomp = seasonal_decompose(ts, model="additive", period=12)
                    decomp.plot(); plt.suptitle(f"Seasonal Decomposition – {t}")
                    pdf.savefig(); plt.close()
                except: pass
            model, fc = forecast_series(ts)
            if fc is not None:
                plt.figure(figsize=(10,6))
                plt.plot(ts.index, ts, label="Historical")
                plt.plot(fc.index, fc, label=f"Forecast ({model})", linestyle="--")
                plt.title(f"Forecast 12 Months – {t}")
                plt.legend(); pdf.savefig(); plt.close()

        # ----- Purchase Plan Table -----
        if purchase_plan is not None and not purchase_plan.empty:
            # Show per-type in batches to fit the page
            for t, g in purchase_plan.groupby("paper_type"):
                tbl = g.copy()
                tbl["date"] = tbl["date"].dt.strftime("%Y-%m")
                values = np.round(tbl["quantity"].values, 0).astype(int)
                col_labels = tbl["date"].tolist()
                plt.figure(figsize=(11, 3 + max(1, len(values)/6)))
                plt.axis("off")
                plt.table(
                    cellText=[values.tolist()],
                    colLabels=col_labels,
                    rowLabels=[t],
                    loc="center"
                )
                plt.title(f"Purchase Plan – next 12 months: {t} ({OUTPUT_UNIT})")
                pdf.savefig(); plt.close()

            # Consolidated table (all types x months)
            try:
                pivot = purchase_plan.copy()
                pivot["date"] = pivot["date"].dt.strftime("%Y-%m")
                pivot_tbl = pivot.pivot(index="paper_type", columns="date", values="quantity").fillna(0)
                # Split across multiple pages if needed
                cols = list(pivot_tbl.columns)
                chunk_size = 8
                for i in range(0, len(cols), chunk_size):
                    sub = pivot_tbl[cols[i:i+chunk_size]]
                    plt.figure(figsize=(11, 3 + 0.3*len(sub)))
                    plt.axis("off")
                    plt.table(
                        cellText=np.round(sub.values,0).astype(int),
                        colLabels=sub.columns.tolist(),
                        rowLabels=sub.index.tolist(),
                        loc="center"
                    )
                    plt.title(f"Purchase Plan – consolidated ({OUTPUT_UNIT})")
                    pdf.savefig(); plt.close()
            except:
                pass

# =========================
# MAIN
# =========================
if __name__=="__main__":
    df = load_data(INPUT_FILE)
    if df.empty:
        print("No data parsed.")
        exit()
    monthly = df.groupby(["date","paper_type"])["consumption"].sum().reset_index()
    monthly = filter_covid_and_anomalies(monthly)
    today = datetime.today().strftime("%Y-%m-%d")
    out_file = os.path.join(OUTPUT_FOLDER, f"dakri_consumption_report_{today}.pdf")
    # Build 12-month purchase plan (forecast per type)
    future_months = pd.date_range(start=(monthly["date"].max() + pd.offsets.MonthBegin(1)), periods=12, freq="MS")
    plan_rows = []
    for t, g in monthly.groupby("paper_type"):
        ts = g.set_index("date")["consumption"].asfreq("MS")
        _, fc = forecast_series(ts)
        if fc is None:
            continue
        # Align to exactly the next 12 months
        fc = fc.asfreq("MS")
        fc = fc.reindex(future_months)
        # Clamp to recent behavior and cap spikes
        fc = clamp_forecast_to_recent_behavior(ts, fc)
        for dt, qty in fc.items():
            if pd.isna(qty):
                continue
            plan_rows.append({"paper_type": t, "date": dt, "quantity": float(qty)})
    purchase_plan = pd.DataFrame(plan_rows).sort_values(["paper_type","date"]).reset_index(drop=True)

    # Convert units for output
    if not purchase_plan.empty and UNIT_FACTOR != 1.0:
        purchase_plan["quantity"] = purchase_plan["quantity"] * UNIT_FACTOR
    purchase_plan["unit"] = OUTPUT_UNIT

    # Save and print plan
    plan_csv = os.path.join(OUTPUT_FOLDER, f"purchase_plan_{today}.csv")
    if not purchase_plan.empty:
        purchase_plan.to_csv(plan_csv, index=False)
        print("Purchase plan (next 12 months) saved:", plan_csv)
        # Console preview
        print("\nPurchase plan preview:")
        preview = purchase_plan.copy()
        preview["date"] = preview["date"].dt.strftime("%Y-%m")
        print(preview.head(24).to_string(index=False))

    generate_report(monthly, out_file, purchase_plan=purchase_plan)
    print("Report generated:", out_file)
