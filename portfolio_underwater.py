import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Load table: auto-detect CSV/Excel and encoding ----------
def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            continue
    # Fallback
    return pd.read_csv(path)

# ---------- Plot Underwater (Drawdown) chart based on Open→Open cumulative movement ----------
def plot_underwater(path: str):
    df = load_table(path)

    # 1) Use 'Index' as the date column directly
    if "Index" not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: No 'Index' column found.")
    date_col = "Index"

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # 2) Compute Open→Open cumulative movement (starting from 0)
    if "Open" not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: No 'Open' column found.")
    y = pd.to_numeric(df["Open"], errors="coerce").diff().fillna(0).cumsum()
    y = y - y.iloc[0]

    # 3) Compute Drawdown (Underwater)
    roll_max = np.maximum.accumulate(y)
    dd = y - roll_max  # ≤ 0

    # 4) Plot
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.fill_between(df[date_col], dd, 0, color='C0', alpha=0.28)
    ax.plot(df[date_col], dd, color='C0', lw=1.6)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.set_title("Portfolio Underwater (Drawdown)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # 5) Save figure
    stem, _ = os.path.splitext(path)
    out = f"{stem}_underwater.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] {os.path.basename(path)} → {os.path.basename(out)}")

# ---------- Batch processing for files 01..10 ----------
def plot_folder(dirpath: str):
    # Match 01..10 
    pats = re.compile(r"^(0?\d{1,3})\.(csv|xlsx|xls)$", re.I)
    files = []
    for name in os.listdir(dirpath):
        if pats.match(name) or re.fullmatch(r"\d{2,3}", name):
            p = os.path.join(dirpath, name)
            if os.path.isfile(p):
                files.append(p)
            else:
                for ext in (".csv", ".xlsx", ".xls"):
                    if os.path.isfile(p + ext):
                        files.append(p + ext)
                        break

    if not files:
        # Fallback: collect all CSV/Excel files and sort numerically
        cand = [os.path.join(dirpath, f) for f in os.listdir(dirpath)
                if os.path.splitext(f)[1].lower() in (".csv", ".xlsx", ".xls")]
        files = sorted(cand, key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))

    if not files:
        raise FileNotFoundError("No data files (01..10) found in the directory.")

    for p in files:
        try:
            plot_underwater(p)
        except Exception as e:
            print(f"[SKIP] {os.path.basename(p)}: {e}")

if __name__ == "__main__":
    # Modify this path to your own data directory before running
    plot_folder(r"D:\桌面\PART1")
