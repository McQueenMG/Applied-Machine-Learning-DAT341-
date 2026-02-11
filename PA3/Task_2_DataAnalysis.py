"""
task 2 - data exploration
analyzes the crowdsourced training set and compares it with gold standard labels.

what we check:
- label distribution (how balanced is the dataset?)
- inter-annotator agreement (how often do crowd workers agree with experts?)
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import clean_labels, load_data, plot_confusion_matrix, plot_label_distribution

def analyze_crowd_data(crowd_df):
    """
    task 2.2: check distribution of data
    """
    print("\n[analysis] checking crowdsourced data distribution...")
    
    # 1. clean
    crowd_df = clean_labels(crowd_df)
    
    # 2. stats
    print("label counts (normalized):")
    print(crowd_df['sentiment'].value_counts(normalize=True))
    
    # 3. plot
    plot_label_distribution(crowd_df, "crowdsourced label distribution", "crowdsourced")

    return crowd_df

def compare_crowd_vs_gold(crowd_df, gold_df):
    """
    task 2.3: compare crowdsourced inputs vs gold standard.
    """
    print("\n[analysis] comparison: crowdsourced vs gold...")
    
    # ensure both are clean
    crowd_df = clean_labels(crowd_df)
    gold_df = clean_labels(gold_df)
    
    # merge the two datasets on the tweet text to compare them side-by-side
    merged = pd.merge(crowd_df, gold_df, on='text', suffixes=('_crowd', '_gold'))
    
    # calculate percentage agreement (how often did they pick the same label?)
    agreement_score = np.mean(merged['sentiment_crowd'] == merged['sentiment_gold'])
    print(f"inter-annotator agreement (accuracy): {agreement_score:.2%}")
    
    # plot confusion matrix to visualize disagreements
    plot_confusion_matrix(
        y_true=merged['sentiment_gold'], 
        y_pred=merged['sentiment_crowd'], 
        title="agreement: gold (true) vs crowd (predicted)", 
        filename_suffix="agreement_gold_vs_crowd"
    )

def main():
    print("### task 2: data exploration ###")
    
    crowd_df, gold_df, _ = load_data()
    
    if crowd_df is None: 
        return

    # step 1: analyze basic distribution
    crowd_clean = analyze_crowd_data(crowd_df)
    
    # step 2: compare accuracy of crowd vs expert
    compare_crowd_vs_gold(crowd_df, gold_df)
    
    print("\ntask 2 analysis complete. graphs saved to PA3/Graphs/")

if __name__ == "__main__":
    main()
