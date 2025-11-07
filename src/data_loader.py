import pandas as pd

import numpy as np

# --- Data Generation Logic ---
np.random.seed(42)

# Adjusted for ~1000 rows (60 patients * 100 days = 6000 rows)
num_patients = 60 
days_per_patient = 100
patient_ids = [f"P{str(i).zfill(3)}" for i in range(1, num_patients + 1)]
cohorts = ['A', 'B']

records = []
for pid in patient_ids:
    cohort = np.random.choice(cohorts)
    for day in range(1, days_per_patient + 1):
        dosage = np.random.choice([50, 75, 100])
        compliance = np.clip(np.random.normal(90, 10), 50, 100)
        adverse_event = np.random.choice([0, 1], p=[0.9, 0.1])
        
        # Simulate outcome score: better with higher compliance, worse if adverse event
        base_score = 80 + (dosage - 50) * 0.2 + (compliance - 90) * 0.3 - adverse_event * 15
        outcome = np.clip(np.random.normal(base_score, 5), 40, 100)

        notes_templates = [
            "Patient stable, no complaints.",
            "Mild headache reported, advised rest.",
            "Fatigue noted, monitoring ongoing.",
            "Symptoms improving with current dosage.",
            "Adverse reaction observed, dosage adjustment needed."
        ]
        notes = np.random.choice(notes_templates, p=[0.5, 0.2, 0.15, 0.1, 0.05])

        visit_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=day-1)
        records.append([pid, day, dosage, compliance, adverse_event, notes, outcome, cohort, visit_date.strftime('%Y-%m-%d')])

df = pd.DataFrame(records, columns=[
    'patient_id', 'trial_day', 'dosage_mg', 'compliance_pct',
    'adverse_event_flag', 'doctor_notes', 'outcome_score', 'cohort', 'visit_date'
])
df.to_csv('clinical_trial_data.csv', index=False)
def load_data(file_path: str = 'clinical_trial_data.csv') -> pd.DataFrame:
    """Loads the CSV data and performs initial type conversion."""
    try:
        df = pd.read_csv(file_path)
        # Convert date column
        df['visit_date'] = pd.to_datetime(df['visit_date'])
        # Clean/Fill missing doctor notes
        df['doctor_notes'].fillna('No Note Recorded', inplace=True)
        return df
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {file_path}")
        return pd.DataFrame()

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates patient-specific stats needed for anomaly detection."""
    if df.empty:
        return df

    # Calculate Patient-Specific Mean and Std Dev for Compliance
    patient_stats = df.groupby('patient_id')['compliance_pct'].agg(['mean', 'std']).rename(
        columns={'mean': 'pat_mean_compliance', 'std': 'pat_std_compliance'}
    )
    df = df.merge(patient_stats, on='patient_id', how='left')
    return df
