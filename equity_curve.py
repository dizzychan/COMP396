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

# ---------- Compute Maximum Drawdown ----------
def compute_maxdd(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    roll_max = np.maximum.accumulate(y)
    dd = roll_max - y
    trough = int(dd.argmax())
    peak = int(y[:trough+1].argmax()) if trough >= 0 else 0
    return float(dd[trough]), peak, trough

# ---------- Plot Open→Open cumulative movement ----------
def plot_open2open(path: str, date_col_hint="index"):
    df = load_table(path)

    # 1) Detect date column
    if "Index" not in df.columns:
        raise ValueError(f"{os.path.basename(path)}: No 'Index' column found.")
    date_col = "Index"


    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    # 2) Compute Open→Open cumulative changes (starting from 0)
    open_col = [c for c in df.columns if c.lower() == "open"]
    if not open_col:
        raise ValueError("No 'Open' column found.")
    y = pd.to_numeric(df[open_col[0]], errors="coerce").diff().fillna(0).cumsum()
    y = y - y.iloc[0]

    # 3) Compute MaxDD and locate Peak/Trough
    maxdd, peak_idx, trough_idx = compute_maxdd(y.values)

    # 4) Plot 
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(df[date_col], y, color='b', lw=2, label='Portfolio CumPnL')
    ax.axhline(0, color='gray', ls='--', lw=1.2, label='Break-even')

    if trough_idx > peak_idx:
        ax.axvspan(df[date_col].iloc[peak_idx], df[date_col].iloc[trough_idx],
                   color='C0', alpha=0.12, label='Max DD window')

    ax.scatter(df[date_col].iloc[peak_idx], y.iloc[peak_idx], color='red', s=36)
    ax.text(df[date_col].iloc[peak_idx], y.iloc[peak_idx],
            f" Peak\n {y.iloc[peak_idx]:.2f}", va='bottom')
    ax.scatter(df[date_col].iloc[trough_idx], y.iloc[trough_idx], color='red', s=36)
    ax.text(df[date_col].iloc[trough_idx], y.iloc[trough_idx],
            f" Trough\n {y.iloc[trough_idx]:.2f}", va='top')

    title = f"Portfolio Open→Open CumPnL | PD=  | Activity=  | MaxDD={maxdd:.2f}"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("CumPnL")
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # 5) Save figure
    out = os.path.splitext(path)[0] + ".png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] {os.path.basename(path)} → {os.path.basename(out)}")

# ---------- Batch process for all 01..10 files ----------
def plot_folder(dirpath: str):
    files = [os.path.join(dirpath, f) for f in os.listdir(dirpath)
             if re.fullmatch(r"\d{2,3}(\.(csv|xlsx|xls))?", f, re.I)]
    if not files:  # fallback: match all CSV/Excel files
        exts = (".csv", ".xlsx", ".xls")
        files = [os.path.join(dirpath, f)
                 for f in os.listdir(dirpath)
                 if os.path.splitext(f)[1].lower() in exts]
    files = sorted(files, key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
    if not files:
        raise FileNotFoundError("No 01..10 data files found in the directory.")

    for p in files:
        try:
            plot_open2open(p, date_col_hint="index")  
        except Exception as e:
            print(f"[SKIP] {os.path.basename(p)}: {e}")

if __name__ == "__main__":
    # Modify this path to your own data directory before running
    plot_folder(r"D:\桌面\PART1")
