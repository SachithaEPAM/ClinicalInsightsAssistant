

import pandas as pd
import numpy as np

def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Applies rule-based and threshold logic to detect issues."""
    if df.empty:
        return df

    # --- 1. Non-Compliance Detection ---
    df['is_non_compliant'] = (df['compliance_pct'] < 80).astype(int)
    
    # --- 2. Side Effect/AE Detection ---
    # Explicit Flag
    df['has_ae'] = df['adverse_event_flag']
    # Inferred (Simple Keyword Matching)
    ae_keywords = ['headache', 'fatigue', 'reaction', 'nausea', 'vomit']
    df['inferred_ae'] = df['doctor_notes'].str.contains('|'.join(ae_keywords), case=False, na=False).astype(int)

    # --- 3. Anomaly Detection ---
    # Global Poor Outcome Anomaly (1 Sigma Below Mean)
    global_mean = df['outcome_score'].mean()
    global_std = df['outcome_score'].std()
    threshold = global_mean - global_std
    df['is_poor_outcome'] = (df['outcome_score'] < threshold).astype(int)

    # Patient-Specific Compliance Drop Anomaly (2 Std Dev Below Patient Mean)
    # Uses 'pat_mean_compliance' and 'pat_std_compliance' from data_loader
    df['sudden_drop_anomaly'] = (
        df['compliance_pct'] < (df['pat_mean_compliance'] - 2 * df['pat_std_compliance'])
    ).astype(int)

    return df
