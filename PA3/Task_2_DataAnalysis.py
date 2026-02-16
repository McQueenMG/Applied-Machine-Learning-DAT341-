import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import clean_labels, load_data, plot_confusion_matrix, plot_label_distribution

def analyze_crowd_data(crowd_df):
        
    # Check label distribution
    print("label counts (normalized):")
    print(crowd_df['sentiment'].value_counts(normalize=True))
    
    # Plot label distribution
    plot_label_distribution(crowd_df, "crowdsourced label distribution", "crowdsourced")


def compare_crowd_vs_gold(crowd_df, gold_df):
    
    # Merge the two datasets to compare them side-by-side
    merged = pd.merge(crowd_df, gold_df, on='text', suffixes=('_crowd', '_gold'))
    
    # Calculate percentage agreement (how often did they pick the same label?)
    agreement_score = np.mean(merged['sentiment_crowd'] == merged['sentiment_gold'])
    print(f"inter-annotator agreement (accuracy): {agreement_score:.2%}")
    
    # Plot confusion matrix to see where they disagree
    plot_confusion_matrix(
        y_true=merged['sentiment_gold'], 
        y_pred=merged['sentiment_crowd'], 
        title="agreement: gold (true) vs crowd (predicted)", 
        filename_suffix="agreement_gold_vs_crowd"
    )

def main():
    
    crowd_df, gold_df, _ = load_data()
    
    if crowd_df is None: 
        return
    
    # First we clean the data
    crowd_df = clean_labels(crowd_df)
    gold_df = clean_labels(gold_df)

    # Then we analyze label distribution of the crowdsourced data
    analyze_crowd_data(crowd_df)
    
    # Lastly we compare accuracy of crowd vs gold to see if they agree
    compare_crowd_vs_gold(crowd_df, gold_df)
    

if __name__ == "__main__":
    main()
