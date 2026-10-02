"""
Script to generate publication-grade EDA visualizations and statistics
for the Loan Default Prediction Engine.
Saves high-res figures to reports/figures/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set global visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "loans_cleaned.parquet"

COLOR_PAID = "#2ecc71"       # Clean green
COLOR_DEFAULT = "#e74c3c"    # Alert red
PALETTE_TARGET = [COLOR_PAID, COLOR_DEFAULT]
COLOR_PRIMARY = "#2c3e50"    # Deep corporate navy


def load_data() -> pd.DataFrame:
    print(f"Loading processed dataset from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def plot_target_distribution(df: pd.DataFrame):
    """Figure 1: Target distribution and class imbalance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    counts = df["target"].value_counts().sort_index()
    labels = ["Fully Paid (0)", "Charged Off (1)"]
    pcts = (counts / len(df)) * 100

    # Bar chart with annotations
    bars = ax1.bar(labels, counts.values, color=PALETTE_TARGET, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.set_title("Loan Status Frequency Count", fontweight="bold", pad=12)
    ax1.set_ylabel("Number of Loans", fontweight="bold")
    ax1.set_ylim(0, max(counts.values) * 1.15)
    
    for bar, count, pct in zip(bars, counts.values, pcts.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 800,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11
        )

    # Donut chart
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=labels,
        autopct="%1.2f%%",
        startangle=140,
        colors=PALETTE_TARGET,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        textprops=dict(fontweight="bold")
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_color("black")
    
    ax2.set_title("Portfolio Default Rate Breakdown", fontweight="bold", pad=12)

    fig.suptitle("Credit Risk Class Distribution (Target Imbalance)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = FIGURES_DIR / "01_target_distribution.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Saved] {out_file.name}")


def plot_grade_and_subgrade_risk(df: pd.DataFrame):
    """Figure 2: Default rate monotonic trend across Grades and Sub-grades."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Grade analysis
    grade_order = sorted(df["grade"].unique())
    grade_stats = df.groupby("grade").agg(
        total=("target", "count"),
        defaults=("target", "sum"),
        default_rate=("target", "mean")
    ).loc[grade_order]

    # Bar chart for grade default rates
    ax1_twin = ax1.twinx()
    ax1_twin.bar(grade_stats.index, grade_stats["total"], color="#bdc3c7", alpha=0.45, width=0.5, label="Loan Volume (Count)")
    ax1_twin.set_ylabel("Loan Volume", color="#7f8c8d", fontweight="bold")
    ax1_twin.grid(False)

    line = ax1.plot(grade_stats.index, grade_stats["default_rate"] * 100, color="#c0392b", marker="o", linewidth=3, markersize=8, label="Default Rate (%)")
    ax1.set_title("Default Rate & Loan Volume Across Lending Club Credit Grades", fontweight="bold", pad=12)
    ax1.set_xlabel("Credit Grade (A = Lowest Risk, G = Highest Risk)", fontweight="bold")
    ax1.set_ylabel("Default Rate (%)", color="#c0392b", fontweight="bold")
    ax1.set_ylim(0, max(grade_stats["default_rate"] * 100) * 1.25)
    
    for x, y in zip(grade_stats.index, grade_stats["default_rate"] * 100):
        ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold", color="#962d22")

    # Sub-grade analysis
    subgrade_order = sorted(df["sub_grade"].unique())
    subgrade_stats = df.groupby("sub_grade").agg(
        default_rate=("target", "mean")
    ).loc[subgrade_order]

    cmap = plt.colormaps.get_cmap("RdYlGn_r")
    norm_rates = (subgrade_stats["default_rate"] - subgrade_stats["default_rate"].min()) / (subgrade_stats["default_rate"].max() - subgrade_stats["default_rate"].min())
    bar_colors = [cmap(v) for v in norm_rates]

    ax2.bar(subgrade_stats.index, subgrade_stats["default_rate"] * 100, color=bar_colors, edgecolor="black", linewidth=0.7)
    ax2.set_title("Monotonic Risk Escalation Across 35 Credit Sub-Grades (A1 to G5)", fontweight="bold", pad=12)
    ax2.set_xlabel("Credit Sub-Grade", fontweight="bold")
    ax2.set_ylabel("Default Rate (%)", fontweight="bold")
    ax2.set_xticklabels(subgrade_stats.index, rotation=45, ha="right", fontsize=9)
    ax2.set_ylim(0, max(subgrade_stats["default_rate"] * 100) * 1.15)

    plt.tight_layout()
    out_file = FIGURES_DIR / "02_default_by_grade_and_subgrade.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Saved] {out_file.name}")


def plot_numerical_risk_distributions(df: pd.DataFrame):
    """Figure 3: Distributions of Interest Rate, DTI, Revolving Util, Annual Income."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Interest Rate
    sns.kdeplot(data=df, x="int_rate", hue="target", common_norm=False, palette=PALETTE_TARGET, fill=True, alpha=0.35, linewidth=2, ax=axes[0, 0])
    axes[0, 0].set_title("Interest Rate Distribution by Status", fontweight="bold")
    axes[0, 0].set_xlabel("Interest Rate (%)", fontweight="bold")
    axes[0, 0].set_ylabel("Density", fontweight="bold")
    axes[0, 0].legend(title="Status", labels=["Default (1)", "Fully Paid (0)"])

    # 2. Debt-to-Income (DTI)
    sns.kdeplot(data=df, x="dti", hue="target", common_norm=False, palette=PALETTE_TARGET, fill=True, alpha=0.35, linewidth=2, ax=axes[0, 1])
    axes[0, 1].set_title("Debt-to-Income (DTI) Ratio Distribution", fontweight="bold")
    axes[0, 1].set_xlabel("DTI (%)", fontweight="bold")
    axes[0, 1].set_ylabel("Density", fontweight="bold")
    axes[0, 1].legend(title="Status", labels=["Default (1)", "Fully Paid (0)"])

    # 3. Revolving Utilization
    sns.kdeplot(data=df, x="revol_util", hue="target", common_norm=False, palette=PALETTE_TARGET, fill=True, alpha=0.35, linewidth=2, ax=axes[1, 0])
    axes[1, 0].set_title("Revolving Credit Line Utilization Rate", fontweight="bold")
    axes[1, 0].set_xlabel("Revolving Utilization (%)", fontweight="bold")
    axes[1, 0].set_ylabel("Density", fontweight="bold")
    axes[1, 0].legend(title="Status", labels=["Default (1)", "Fully Paid (0)"])

    # 4. Annual Income (Log-transformed for readability)
    df_temp = df.copy()
    df_temp["log_annual_inc"] = np.log10(np.clip(df_temp["annual_inc"], 1000, 1000000))
    sns.boxplot(data=df_temp, x="target", y="log_annual_inc", palette=PALETTE_TARGET, ax=axes[1, 1], width=0.45)
    axes[1, 1].set_title("Log10 Annual Income by Status", fontweight="bold")
    axes[1, 1].set_xlabel("Loan Status (0 = Paid, 1 = Default)", fontweight="bold")
    axes[1, 1].set_ylabel("Log10(Annual Income)", fontweight="bold")
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(["Fully Paid", "Charged Off"])

    fig.suptitle("Core Financial Risk Attributes: Fully Paid vs. Default", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_file = FIGURES_DIR / "03_numerical_risk_distributions.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Saved] {out_file.name}")


def plot_term_and_purpose_risk(df: pd.DataFrame):
    """Figure 4: Risk variations across Loan Term and Purpose."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # Term comparison
    term_stats = df.groupby("term").agg(
        total=("target", "count"),
        default_rate=("target", "mean")
    )
    terms = [f"{t} Months" for t in term_stats.index]
    rates = term_stats["default_rate"] * 100
    vol = term_stats["total"]

    bars = ax1.bar(terms, rates, color=["#3498db", "#e67e22"], width=0.45, edgecolor="black", linewidth=1.2)
    ax1.set_title("Default Rate by Loan Term Duration", fontweight="bold", pad=12)
    ax1.set_ylabel("Default Rate (%)", fontweight="bold")
    ax1.set_ylim(0, max(rates) * 1.3)
    
    for bar, r, v in zip(bars, rates, vol):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{r:.1f}%\n({v:,} loans)",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    # Purpose comparison
    purpose_stats = df.groupby("purpose").agg(
        total=("target", "count"),
        default_rate=("target", "mean")
    ).sort_values(by="default_rate", ascending=True)

    purp_colors = [COLOR_DEFAULT if r > df["target"].mean() else "#27ae60" for r in purpose_stats["default_rate"]]

    ax2.barh(purpose_stats.index, purpose_stats["default_rate"] * 100, color=purp_colors, edgecolor="black", linewidth=0.7)
    ax2.axvline(df["target"].mean() * 100, color="black", linestyle="--", linewidth=1.5, label=f"Portfolio Baseline ({df['target'].mean()*100:.1f}%)")
    ax2.set_title("Default Rate by Stated Loan Purpose", fontweight="bold", pad=12)
    ax2.set_xlabel("Default Rate (%)", fontweight="bold")
    ax2.legend(loc="lower right")

    for idx, (p, r) in enumerate(zip(purpose_stats.index, purpose_stats["default_rate"] * 100)):
        ax2.text(r + 0.3, idx, f"{r:.1f}%", va="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    out_file = FIGURES_DIR / "04_loan_term_and_purpose_risk.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Saved] {out_file.name}")


def plot_bureau_and_delinquency_risk(df: pd.DataFrame):
    """Figure 5: Credit bureau inquiries and prior delinquency indicators."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Inquiries in last 6 months (capped at 6+ for clean display)
    df_temp = df.copy()
    df_temp["inq_binned"] = df_temp["inq_last_6mths"].apply(lambda x: "6+" if x >= 6 else str(int(x)))
    inq_order = [str(i) for i in range(6)] + ["6+"]
    inq_stats = df_temp.groupby("inq_binned").agg(
        total=("target", "count"),
        default_rate=("target", "mean")
    ).reindex(inq_order).dropna()

    ax1.plot(inq_stats.index, inq_stats["default_rate"] * 100, marker="s", color="#8e44ad", linewidth=2.5, markersize=8)
    ax1.set_title("Default Risk vs. Inquiries in Last 6 Months (Velocity)", fontweight="bold", pad=12)
    ax1.set_xlabel("Credit Inquiries (Last 6 Months)", fontweight="bold")
    ax1.set_ylabel("Default Rate (%)", fontweight="bold")
    ax1.set_ylim(0, max(inq_stats["default_rate"] * 100) * 1.25)
    
    for x, y in zip(inq_stats.index, inq_stats["default_rate"] * 100):
        ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold")

    # Public record bankruptcies
    bankrupt_stats = df.groupby("pub_rec_bankruptcies").agg(
        total=("target", "count"),
        default_rate=("target", "mean")
    )
    b_labels = [f"{int(b)} Bankruptcies" for b in bankrupt_stats.index]
    b_rates = bankrupt_stats["default_rate"] * 100

    bars = ax2.bar(b_labels, b_rates, color=["#16a085", "#e67e22", "#c0392b"], width=0.45, edgecolor="black", linewidth=1.2)
    ax2.set_title("Default Rate by Recorded Public Bankruptcies", fontweight="bold", pad=12)
    ax2.set_ylabel("Default Rate (%)", fontweight="bold")
    ax2.set_ylim(0, max(b_rates) * 1.3)

    for bar, r, v in zip(bars, b_rates, bankrupt_stats["total"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{r:.1f}%\n({v:,} loans)",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    plt.tight_layout()
    out_file = FIGURES_DIR / "05_bureau_derogatories_and_inquiries.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Saved] {out_file.name}")


def plot_correlation_matrix(df: pd.DataFrame):
    """Figure 6: Correlation matrix of continuous features with the target."""
    num_cols = [
        "target", "loan_amnt", "term", "int_rate", "installment",
        "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
        "pub_rec_bankruptcies", "emp_length_years"
    ]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=0.6,
        center=0,
        square=True,
        linewidths=0.6,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax
    )
    ax.set_title("Correlation Matrix of Continuous Features with Target", fontsize=15, fontweight="bold", pad=15)

    plt.tight_layout()
    out_file = FIGURES_DIR / "06_correlation_matrix.png"
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Saved] {out_file.name}")


def main():
    df = load_data()
    print("Generating comprehensive publication-grade EDA visualizations...")
    plot_target_distribution(df)
    plot_grade_and_subgrade_risk(df)
    plot_numerical_risk_distributions(df)
    plot_term_and_purpose_risk(df)
    plot_bureau_and_delinquency_risk(df)
    plot_correlation_matrix(df)
    print("All 6 figures successfully saved to reports/figures/!")


if __name__ == "__main__":
    main()
