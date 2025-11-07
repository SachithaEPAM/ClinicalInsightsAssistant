import pandas as pd
from src.data_loader import load_data, prepare_data
from src.issue_detection import apply_rules
from src.genai_interface import llm_summarize_notes, generate_regulatory_summary

def main_pipeline():
    print("--- Starting Clinical Insights Assistant Pipeline ---")

    # 1. Load & Prepare Data
    df = load_data()
    if df.empty:
        return
    df = prepare_data(df)
    print("Step 1/4: Data loaded and prepared.")

    # 2. Rule-Based Analysis (Detect Issues)
    df_analyzed = apply_rules(df)
    print("Step 2/4: Rule-based detection (Non-Compliance, AE, Anomalies) completed.")

    # 3. LLM Integration (Unstructured Data Analysis)
    df_llm_processed = llm_summarize_notes(df_analyzed)
    print("Step 3/4: Doctor notes summarized and key findings extracted via LLM.")

    # 4. Agentic AI Simulation/Summary (Conceptually, this is where the Agent would run)
    # MOCK: Simulate a scenario analysis result (e.g., from an Agent)
    mock_scenario_result = "Simulated 10mg dosage reduction: Predicted 5% drop in efficacy, 15% drop in AE rate."
    print("Simulated scenario: ", mock_scenario_result)

    # 5. Regulatory Summary Generation
    final_summary = generate_regulatory_summary(df_llm_processed)
    
    print("\n" + "="*70)
    print("                    FINAL REGULATORY SUMMARY")
    print("="*70)
    print(final_summary)
    print("="*70)

    # OPTIONAL: Save the detailed processed data
    # df_llm_processed.to_csv('processed_clinical_data.csv', index=False)

if __name__ == "__main__":
    main_pipeline()
