import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

sns.set_theme(style="whitegrid", context="talk", palette="deep")

ROOT = Path("..")
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "eda" / "figures"
REPORT = ROOT / "reports" / "eda" / "EDA_report.md"
PROC.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

FLAT_THRESHOLD = 0.005

# ─── 1. Загрузка данных ──────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW / "ohlcv_all.csv", parse_dates=["TRADEDATE"])
df = df.rename(
    columns={
        "TRADEDATE": "date",
        "SECID": "ticker",
        "SHORTNAME": "name",
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "VOLUME": "volume",
        "VALUE": "value",
        "NUMTRADES": "ntrades",
    }
)
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

raw_rows = len(df)
n_tickers = df["ticker"].nunique()
dup_rows = df.duplicated(subset=["ticker", "date"]).sum()
date_min, date_max = df["date"].min(), df["date"].max()
ohcv_na = df[["open", "high", "low", "close", "volume"]].isna().sum().to_dict()
zero_vol = (df["volume"].fillna(0) == 0).sum()

# ─── 2. Саммари по тикетам ───────────────────────────────────────────────────
g = df.groupby("ticker")
summary = pd.DataFrame(
    {
        "name": g["name"].first(),
        "n_rows": g.size(),
        "date_min": g["date"].min(),
        "date_max": g["date"].max(),
        "n_missing_close": g["close"].apply(lambda s: s.isna().sum()),
        "n_zero_volume": g["volume"].apply(lambda s: (s.fillna(0) == 0).sum()),
    }
).reset_index()
summary.to_csv(PROC / "summary_stats.csv", index=False)

# ─── 3. Returns ───────────────────────────────────────────────────────────────
# Убираем строки, где close = NaN (дни, когда тикер номинально торгуется, но
# сделок не было — например, переходный период консолидации VTBR 2024-07-09..12).
# Это позволяет корректно детектировать корпоративных действи как большое процентное изменение
# между соседними днями торгов.
df = df.dropna(subset=["close"]).reset_index(drop=True)

# Считаем "сырой" ret для детекции corporate actions.
df["ret_raw"] = df.groupby("ticker")["close"].pct_change(fill_method=None)

# Корпоративные действия: |ret| > 50% между соседними торговыми днями — это
# сплит или консолидация. MOEX ISS не делает коррекцию цены,
# поэтому нужно обработать вручную: forward-adjust — умножить на все цены до события на
# ratio = close_after / close_before, чтобы весь ценовой ряд был в после разбиения.
CORP_ACT_THRESHOLD = 0.50
corp_idx = df.index[df["ret_raw"].abs() > CORP_ACT_THRESHOLD]
corp_actions = df.loc[corp_idx, ["ticker", "date", "close", "ret_raw"]].copy()
corp_actions["event"] = corp_actions["ret_raw"].apply(
    lambda r: "split/consolidation" if r < -0.5 else "reverse-split/aggregation"
)
corp_actions.to_csv(PROC / "corporate_actions.csv", index=False)

# Forward-adjust цен каждого тикера, у которого есть corporate actions.
# Обрабатываем события по очереди (от раннего к позднему), чтобы ratio корректно накапливался.
# Для каждого события:
#   close_before = close в строке перед событием
#   close_after  = close в строке события
#   ratio = close_after / close_before
#   Все цены (open/high/low/close) до события умножаются на ratio.
df_adj = df.copy()
price_cols = ["open", "high", "low", "close"]
for _, ev in corp_actions.sort_values("date").iterrows():
    tk = ev["ticker"]
    ev_date = ev["date"]
    sub = df_adj[df_adj["ticker"] == tk].sort_values("date").reset_index()
    ev_pos = sub.index[sub["date"] == ev_date]
    if len(ev_pos) == 0 or ev_pos[0] == 0:
        continue
    pos = int(ev_pos[0])
    close_before = float(sub.loc[pos - 1, "close"])
    close_after = float(sub.loc[pos, "close"])
    if close_before <= 0:
        continue
    ratio = close_after / close_before
    # Маска для строк до события этого тикера
    pre_mask = (df_adj["ticker"] == tk) & (df_adj["date"] < ev_date)
    for col in price_cols:
        df_adj.loc[pre_mask, col] = df_adj.loc[pre_mask, col] * ratio

# Считаем returns на adjusted close
df["ret"] = df_adj.groupby("ticker")["close"].pct_change(fill_method=None)
df["ret_fwd_1d"] = df_adj.groupby("ticker")["close"].shift(-1) / df_adj["close"] - 1
df["ret_fwd_5d"] = df_adj.groupby("ticker")["close"].shift(-5) / df_adj["close"] - 1
df["ret_fwd_20d"] = df_adj.groupby("ticker")["close"].shift(-20) / df_adj["close"] - 1
df["close_adj"] = df_adj["close"]


def classify_dir(r: float, thr: float = FLAT_THRESHOLD) -> str:
    if pd.isna(r):
        return np.nan
    if r > thr:
        return "up"
    if r < -thr:
        return "down"
    return "flat"


df["dir_1d"] = df["ret_fwd_1d"].apply(classify_dir)

# Up/down/flat per ticker
udf = df.dropna(subset=["dir_1d"]).groupby(["ticker", "dir_1d"]).size().unstack(fill_value=0)
for c in ["up", "down", "flat"]:
    if c not in udf.columns:
        udf[c] = 0
udf = udf[["up", "flat", "down"]]
udf_pct = udf.div(udf.sum(axis=1), axis=0) * 100
udf_pct.to_csv(PROC / "up_down_flat.csv")

# ─── 4. Заготовки для будущих фичей ─────────────────────────────────────────
ret_long = df[
    [
        "date",
        "ticker",
        "close",
        "ret",
        "ret_fwd_1d",
        "ret_fwd_5d",
        "ret_fwd_20d",
        "dir_1d",
        "volume",
        "value",
    ]
].copy()
ret_long.to_parquet(PROC / "returns_long.parquet", index=False)

# Матрица корреляций по доходностям
ret_wide = df.pivot(index="date", columns="ticker", values="ret")
corr = ret_wide.corr()
corr.to_csv(PROC / "corr_returns.csv")

# ─── 5. Визуализации ────────────────────────────────────────────────────────
PALETTE = sns.color_palette("deep", n_colors=15)

# 5.1 Cumulative returns, на старте нормируем до 100.
cum = (1 + ret_wide).cumprod() * 100
fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
for i, tk in enumerate(cum.columns):
    ax.plot(cum.index, cum[tk], label=tk, linewidth=1.6, color=PALETTE[i])
ax.set_title(
    "Кумулятивная доходность тикеров MOEX (база = 100, 2023-01-03)", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Дата")
ax.set_ylabel("Кум. доходность (база 100)")
ax.legend(ncol=5, fontsize=10, loc="upper left", bbox_to_anchor=(0, -0.12))
ax.axhline(100, color="grey", linestyle="--", linewidth=0.8)
fig.savefig(FIG / "01_cumulative_returns.png", dpi=120)
plt.close(fig)

# 5.2 Распределение дневных доходностей (overall + per-ticker boxplot)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
rets = df["ret"].dropna()
ax1.hist(rets.clip(-0.12, 0.12), bins=80, color="#376FA3", alpha=0.85, edgecolor="white")
ax1.axvline(0, color="black", linewidth=0.8)
ax1.set_title("Распределение дневных доходностей (все тикеры)", fontsize=13, fontweight="bold")
ax1.set_xlabel("Дневная доходность")
ax1.set_ylabel("Частота")
ax1.text(
    0.02,
    0.95,
    f"mean={rets.mean() * 100:.3f}%\n"
    f"std={rets.std() * 100:.3f}%\n"
    f"skew={rets.skew():.2f}\n"
    f"kurt={rets.kurtosis():.1f}",
    transform=ax1.transAxes,
    va="top",
    fontsize=11,
    bbox=dict(boxstyle="round", fc="white", alpha=0.85),
)
sns.boxplot(
    data=df.dropna(subset=["ret"]), x="ticker", y="ret", showfliers=False, palette=PALETTE, ax=ax2
)
ax2.set_title("Boxplot доходностей по тикерам", fontsize=13, fontweight="bold")
ax2.set_xlabel("Тикер")
ax2.set_ylabel("Дневная доходность")
ax2.tick_params(axis="x", rotation=45)
fig.savefig(FIG / "02_returns_distribution.png", dpi=120)
plt.close(fig)

# 5.3 Корреляционная матрица
fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=-0.3,
    vmax=1.0,
    square=True,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Корреляция Пирсона"},
)
ax.set_title(
    "Корреляция дневных доходностей (15 тикеров MOEX, 2023–2025)", fontsize=13, fontweight="bold"
)
fig.savefig(FIG / "03_correlation_heatmap.png", dpi=120)
plt.close(fig)

# 5.4 Up / Down / Flat распределение
fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
udf_pct_sorted = udf_pct.sort_values("up", ascending=False)
udf_pct_sorted.plot(
    kind="bar", stacked=True, color=["#3DDBB5", "#888888", "#D45A5A"], ax=ax, width=0.75
)
ax.set_title(
    f"Структура направления дневной доходности (порог flat = ±{FLAT_THRESHOLD * 100}%)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Тикер")
ax.set_ylabel("Доля, %")
ax.set_ylim(0, 100)
ax.legend(title="Направление", loc="upper left", bbox_to_anchor=(1.01, 1))
ax.tick_params(axis="x", rotation=45)
for i, tk in enumerate(udf_pct_sorted.index):
    up_v = udf_pct_sorted.loc[tk, "up"]
    dn_v = udf_pct_sorted.loc[tk, "down"]
    ax.text(
        i,
        50,
        f"{up_v:.0f}/{dn_v:.0f}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
    )
fig.savefig(FIG / "04_up_down_flat.png", dpi=120)
plt.close(fig)

# 5.5 Оборот по времени (сумма value по всем тикерам в день)
daily_value = df.groupby("date")["value"].sum() / 1e9  # млрд ₽
fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
daily_value.plot(ax=ax, color="#1A1A1A", linewidth=1.0)
daily_value.rolling(20).mean().plot(
    ax=ax, color="#C9A84C", linewidth=2.0, label="20д скользящее среднее"
)
ax.set_title("Суммарный оборот MOEX по 15 тикерам (млрд ₽/день)", fontsize=13, fontweight="bold")
ax.set_xlabel("Дата")
ax.set_ylabel("Оборот, млрд ₽")
ax.legend()
fig.savefig(FIG / "05_daily_turnover.png", dpi=120)
plt.close(fig)

# 5.6 Карта пропусков (по фактическим торговым дням MOEX)
all_dates = pd.date_range(date_min, date_max, freq="D")
presence = pd.DataFrame(0, index=all_dates, columns=ret_wide.columns, dtype=int)
for tk in ret_wide.columns:
    s_dates = ret_wide[tk].dropna().index
    common = s_dates.intersection(all_dates)
    presence.loc[common, tk] = 1

# Сожмём до дней, где хотя бы один тикер торговался
traded_days = presence.sum(axis=1)
presence = presence.loc[traded_days > 0]
fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
sns.heatmap(presence.T, cmap=["#FFFFFF", "#376FA3"], cbar=False, ax=ax, yticklabels=True)
ax.set_title(
    "Присутствие тикеров в торговых днях MOEX (1 = есть данные, 0 = пропуск)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Торговые дни MOEX (сжато)")
ax.set_ylabel("Тикер")
ax.tick_params(axis="x", labelsize=7)
fig.savefig(FIG / "06_presence_heatmap.png", dpi=120)
plt.close(fig)

# 5.7 Волатильность окном в 30 дней по тикерам
roll_vol = ret_wide.rolling(30).std() * np.sqrt(252) * 100
fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
for i, tk in enumerate(roll_vol.columns):
    ax.plot(roll_vol.index, roll_vol[tk], label=tk, linewidth=1.2, color=PALETTE[i], alpha=0.85)
ax.set_title("Скользящая годовая волатильность (окно 30д, ×√252)", fontsize=13, fontweight="bold")
ax.set_xlabel("Дата")
ax.set_ylabel("Волатильность, % годовых")
ax.legend(ncol=5, fontsize=10, loc="upper left", bbox_to_anchor=(0, -0.12))
fig.savefig(FIG / "07_rolling_volatility.png", dpi=120)
plt.close(fig)

# 5.8 Распределение объема торгов по тикетам
fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
volumes_log = np.log10(df["volume"].replace(0, np.nan).dropna() + 1)
df_plot = df.assign(log_vol=volumes_log)
sns.boxplot(
    data=df_plot.dropna(subset=["log_vol"]),
    x="ticker",
    y="log_vol",
    showfliers=False,
    palette=PALETTE,
    ax=ax,
)
ax.set_title("Распределение объёма торгов (log10) по тикерам", fontsize=13, fontweight="bold")
ax.set_xlabel("Тикер")
ax.set_ylabel("log10(Volume)")
ax.tick_params(axis="x", rotation=45)
fig.savefig(FIG / "08_volume_distribution.png", dpi=120)
plt.close(fig)

# ─── 6. Агрегаты ───────────────────────────────────────────────────
overall_ret_stats = {
    "mean_daily_ret_pct": round(rets.mean() * 100, 4),
    "std_daily_ret_pct": round(rets.std() * 100, 4),
    "annualized_vol_pct": round(rets.std() * np.sqrt(252) * 100, 2),
    "skew": round(rets.skew(), 3),
    "kurtosis": round(rets.kurtosis(), 3),
    "min_ret_pct": round(rets.min() * 100, 3),
    "max_ret_pct": round(rets.max() * 100, 3),
    "n_observations": int(len(rets)),
    "share_up_pct": round((rets > FLAT_THRESHOLD).mean() * 100, 2),
    "share_down_pct": round((rets < -FLAT_THRESHOLD).mean() * 100, 2),
    "share_flat_pct": round(rets.between(-FLAT_THRESHOLD, FLAT_THRESHOLD).mean() * 100, 2),
}
overall_dir = (df["dir_1d"].value_counts(normalize=True, dropna=True) * 100).round(2).to_dict()

avg_corr = float(corr.where(~np.eye(len(corr), dtype=bool)).stack().mean())
median_corr = float(np.median(corr.where(~np.eye(len(corr), dtype=bool)).stack().dropna()))
max_corr_pair = corr.where(~np.eye(len(corr), dtype=bool)).stack().idxmax()
max_corr_val = float(corr.loc[max_corr_pair[0], max_corr_pair[1]])
min_corr_pair = corr.where(~np.eye(len(corr), dtype=bool)).stack().idxmin()
min_corr_val = float(corr.loc[min_corr_pair[0], min_corr_pair[1]])

# Найдём большие gaps по дням
per_ticker_dates = df.groupby("ticker")["date"].agg(["min", "max", "count", "nunique"])


print(f"Визуализации: {len(list(FIG.glob('*.png')))} файлов в {FIG}")
print(f"Агрегаты: {len(list(PROC.glob('*')))} файлов в {PROC}")
print(f"\n Средняя корреляция: {avg_corr:.3f}")
print(
    f"up/down/flat overall: "
    f"{overall_ret_stats['share_up_pct']:.1f}/{overall_ret_stats['share_flat_pct']:.1f}/{overall_ret_stats['share_down_pct']:.1f}%"
)
print(f"annualized vol: {overall_ret_stats['annualized_vol_pct']}%")
