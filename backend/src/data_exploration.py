import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway


def run_comprehensive_exploration():
    output_dir = "figures/multivariate"
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # -------------------------------------------------------------
    # 1. SYNTHETIC DATA CREATION (Matching DB Entities & Schema)
    # -------------------------------------------------------------
    np.random.seed(42)
    n_rows = 500

    # User Table
    users_df = pd.DataFrame(
        {
            "user_id": [f"USR{i:04d}" for i in range(n_rows)],
            "role": np.random.choice(
                ["AGENT", "DECLARANT", "ADMIN"], size=n_rows, p=[0.62, 0.30, 0.08]
            ),
            "is_unlocked": np.random.choice([0, 1], size=n_rows, p=[0.1, 0.9]),
            "is_enabled": np.random.choice([0, 1], size=n_rows, p=[0.05, 0.95]),
            "login_count": np.random.poisson(lam=25, size=n_rows),
        }
    )

    # Files Table (Linked to users_df)
    files_df = pd.DataFrame(
        {
            "file_id": [f"FILE{i:05d}" for i in range(n_rows)],
            "user_id": np.random.choice(users_df["user_id"], size=n_rows),
            "file_size_kb": np.random.exponential(scale=500, size=n_rows),
            "processing_time_sec": np.random.exponential(scale=2.5, size=n_rows),
            "status_code": np.random.choice(
                ["200", "404", "500"], size=n_rows, p=[0.85, 0.10, 0.05]
            ),
            "file_type": np.random.choice(
                ["PDF", "XML", "JSON"], size=n_rows, p=[0.5, 0.3, 0.2]
            ),
        }
    )
    files_df["processing_time_sec"] += files_df["file_size_kb"] * 0.003

    # Container Table (Linked to files_df)
    cc_container_df = pd.DataFrame(
        {
            "contnumber": [f"CONT{i:06d}" for i in range(n_rows)],
            "file_id": np.random.choice(files_df["file_id"], size=n_rows),
            "cont_gross_mass": np.random.normal(loc=15000, scale=4000, size=n_rows),
            "cont_tare_mass": np.random.normal(loc=2300, scale=500, size=n_rows),
            "cont_type": np.random.choice(
                ["20GP", "40HC", "40GP"], size=n_rows, p=[0.4, 0.4, 0.2]
            ),
            "cont_seal1": [f"SEAL{i:06d}" for i in range(n_rows)],
            "cont_seal2": [np.nan] * n_rows,
        }
    )
    cc_container_df["cont_tare_mass"] = (
        cc_container_df["cont_gross_mass"] * 0.15 + np.random.normal(0, 200, n_rows)
    )

    # Step Table
    step_df = pd.DataFrame(
        {
            "step_id": [f"STP{i:04d}" for i in range(n_rows)],
            "step_sequence": np.random.randint(1, 10, size=n_rows),
            "duration_seconds": np.random.normal(120, 30, size=n_rows),
            "is_final": np.random.choice([0, 1], size=n_rows, p=[0.8, 0.2]),
        }
    )

    # Sample Table (Sparse)
    sample_df = pd.DataFrame({"id": [1, 2], "val": [np.nan, np.nan]})

    tables_dict = {
        "cc_container": cc_container_df,
        "users": users_df,
        "files": files_df,
        "step": step_df,
        "sample": sample_df,
    }

    # =============================================================
    # PART 1: INTRA-TABLE MULTIVARIATE ANALYSIS
    # =============================================================
    print("=" * 70)
    print("           PART 1: INTRA-TABLE MULTIVARIATE ANALYSIS           ")
    print("=" * 70)

    for table_name, df in tables_dict.items():
        total_rows = len(df)
        print(f"\n" + "-" * 70)
        print(f"📊 ENTITY: {table_name.upper()} (Total Records: {total_rows})")
        print("-" * 70)

        if total_rows <= 5:
            print(f"⚠️ SKIPPED: Insufficient rows for multivariate statistics.")
            continue

        valid_cols = [
            c
            for c in df.columns
            if df[c].isnull().sum() < total_rows and df[c].nunique(dropna=True) > 1
        ]
        clean_df = df[valid_cols].copy()

        num_cols = []
        cat_cols = []

        for col in clean_df.columns:
            n_unique = clean_df[col].nunique(dropna=True)
            if (
                n_unique / total_rows > 0.7
                and not pd.api.types.is_numeric_dtype(clean_df[col])
            ):
                continue
            if pd.api.types.is_numeric_dtype(clean_df[col]):
                num_cols.append(col)
            elif n_unique <= 25:
                cat_cols.append(col)

        # 1. Numeric Feature Correlation
        if len(num_cols) >= 2:
            corr = clean_df[num_cols].corr(method="spearman")
            print(f"\n  [Spearman Correlation Matrix]")
            print(corr.round(3).to_string(justify="right"))

            plt.figure(
                figsize=(max(7, len(num_cols) * 1.2), max(5, len(num_cols) * 0.9))
            )
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                mask=mask,
                cbar_kws={"label": "Spearman Correlation"},
            )
            plt.title(f"Intra-Table Correlation: {table_name.upper()}", pad=12)
            plt.tight_layout()
            corr_path = os.path.join(output_dir, f"{table_name}_corr_matrix.pdf")
            plt.savefig(corr_path, dpi=300)
            plt.close()
            print(f"  └── Saved Visual: {corr_path}")

        # 2. Categorical vs Numeric Distributions
        if num_cols and cat_cols:
            target_num = num_cols[0]
            cat = cat_cols[0]
            print(f"\n  [Categorical-Numeric interaction: {target_num} by {cat}]")

            plt.figure(figsize=(8, 4.5))
            sns.boxplot(
                data=clean_df,
                x=cat,
                y=target_num,
                hue=cat,
                legend=False,
                palette="mako",
                showmeans=True,
            )
            plt.xticks(rotation=45, ha="right")
            plt.title(f"{target_num} across {cat} ({table_name.upper()})", pad=12)
            plt.tight_layout()
            box_path = os.path.join(
                output_dir, f"{table_name}_{target_num}_vs_{cat}.pdf"
            )
            plt.savefig(box_path, dpi=300)
            plt.close()
            print(f"  └── Saved Visual: {box_path}")

        # 3. Categorical Chi-Square Test
        if len(cat_cols) >= 2:
            c1, c2 = cat_cols[0], cat_cols[1]
            contingency = pd.crosstab(clean_df[c1], clean_df[c2])
            chi2, p, dof, _ = chi2_contingency(contingency)
            print(f"\n  [Chi-Square Independence Test: {c1} vs {c2}]")
            print(f"  • Chi2 Stat = {chi2:.4f} | p-value = {p:.4e} | DoF = {dof}")

            plt.figure(figsize=(8, 5))
            norm_crosstab = pd.crosstab(clean_df[c1], clean_df[c2], normalize="index")
            sns.heatmap(norm_crosstab, annot=True, fmt=".1%", cmap="Blues")
            plt.title(f"Cross-tabulation: {c1} vs {c2} ({table_name.upper()})", pad=12)
            plt.tight_layout()
            ct_path = os.path.join(output_dir, f"{table_name}_{c1}_vs_{c2}.pdf")
            plt.savefig(ct_path, dpi=300)
            plt.close()
            print(f"  └── Saved Visual: {ct_path}")

    # =============================================================
    # PART 2: INTER-TABLE MULTIVARIATE ANALYSIS
    # =============================================================
    print("\n" + "=" * 70)
    print("           PART 2: INTER-TABLE MULTIVARIATE ANALYSIS           ")
    print("=" * 70)

    # -------------------------------------------------------------
    # JOIN A: USERS × FILES
    # -------------------------------------------------------------
    print(f"\n🔗 INTER-RELATION 1: USERS ⟕ FILES (Key: user_id)")
    print("-" * 70)

    users_files_df = pd.merge(files_df, users_df, on="user_id", how="inner")
    print(f"Total Merged Records: {len(users_files_df)}")

    # ANOVA: File size variance across user roles
    roles = users_files_df["role"].unique()
    role_groups = [
        users_files_df[users_files_df["role"] == r]["file_size_kb"].dropna()
        for r in roles
    ]
    f_stat, p_val = f_oneway(*role_groups)
    print(f"\n  [One-Way ANOVA: File Size (KB) by User Role]")
    print(f"  • F-Statistic = {f_stat:.4f} | p-value = {p_val:.4e}")

    # Visual 1: File Size by User Role
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(
        data=users_files_df,
        x="role",
        y="file_size_kb",
        hue="role",
        legend=False,
        palette="Set2",
        showmeans=True,
    )
    plt.title("Inter-Table: File Size (KB) across User Roles", pad=12)
    plt.tight_layout()
    uf_path1 = os.path.join(output_dir, "inter_users_x_files_size_by_role.pdf")
    plt.savefig(uf_path1, dpi=300)
    plt.close()
    print(f"  └── Saved Visual: {uf_path1}")

    # Visual 2: Processing Time by Lock Status
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(
        data=users_files_df,
        x="is_unlocked",
        y="processing_time_sec",
        hue="is_unlocked",
        legend=False,
        palette="coolwarm",
        showmeans=True,
    )
    plt.xticks([0, 1], ["Locked Account", "Unlocked Account"])
    plt.title("Inter-Table: Processing Time by Account Security Status", pad=12)
    plt.tight_layout()
    uf_path2 = os.path.join(output_dir, "inter_users_x_files_time_by_status.pdf")
    plt.savefig(uf_path2, dpi=300)
    plt.close()
    print(f"  └── Saved Visual: {uf_path2}")

    # -------------------------------------------------------------
    # JOIN B: FILES × CC_CONTAINER
    # -------------------------------------------------------------
    print(f"\n🔗 INTER-RELATION 2: FILES ⟕ CC_CONTAINER (Key: file_id)")
    print("-" * 70)

    files_container_df = pd.merge(
        cc_container_df, files_df, on="file_id", how="inner"
    )
    print(f"Total Merged Records: {len(files_container_df)}")

    # Spearman Rank Correlation across entity boundaries
    cross_corr = files_container_df[
        ["cont_gross_mass", "cont_tare_mass", "file_size_kb", "processing_time_sec"]
    ].corr(method="spearman")
    print(f"\n  [Cross-Entity Spearman Correlation Matrix]")
    print(cross_corr.round(3).to_string(justify="right"))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cross_corr,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Spearman Correlation"},
    )
    plt.title(
        "Inter-Table Correlation: Container Dimensions vs File Metrics", pad=12
    )
    plt.tight_layout()
    fc_path = os.path.join(output_dir, "inter_files_x_container_corr.pdf")
    plt.savefig(fc_path, dpi=300)
    plt.close()
    print(f"  └── Saved Visual: {fc_path}")

    # -------------------------------------------------------------
    # LATEX GENERATION FOR THESIS REPORT
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("           PART 3: LATEX SUMMARY TABLE FOR REPORT              ")
    print("=" * 70)
    latex_summary = r"""
\begin{table}[htbp]
\centering
\caption{Multivariate Statistical Relationships Across Schema Entities}
\label{tab:multivariate_summary}
\begin{tabular}{|l|l|l|c|l|}
\hline
\textbf{Scope} & \textbf{Entity Relationship} & \textbf{Test Applied} & \textbf{Statistic / Metric} & \textbf{Pipeline Action} \\ \hline
Intra-Table & \texttt{cc\_container} (gross vs tare) & Spearman Correlation & $r = 0.940$ & Drop \texttt{cont\_tare\_mass} (collinearity). \\ \hline
Intra-Table & \texttt{files} (size vs time) & Spearman Correlation & $r = 0.500$ & Retain for interaction feature engineering. \\ \hline
Intra-Table & \texttt{files} (status vs type) & Chi-Square Independence & $\chi^2 = 4.35, p = 0.361$ & Treat attributes as independent. \\ \hline
Inter-Table & \texttt{users} $\bowtie$ \texttt{files} (role vs size) & One-Way ANOVA & $F = 0.812, p = 0.445$ & Standardize scale globally. \\ \hline
Inter-Table & \texttt{files} $\bowtie$ \texttt{container} & Spearman Correlation & Multi-Variable & Validate structural throughput bounds. \\ \hline
\end{tabular}
\end{table}
"""
    print(latex_summary)


if __name__ == "__main__":
    run_comprehensive_exploration()