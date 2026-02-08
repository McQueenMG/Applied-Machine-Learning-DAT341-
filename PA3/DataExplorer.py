import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import clean_labels

def compare_datasets(crowd_df, gold_df):
    # compare
    print("\n--- Agreement: Crowd vs Gold ---")
    merged = pd.merge(crowd_df, gold_df, on='text', suffixes=('_crowd', '_gold'))
    
    
    # calc accuracy
    agreement = np.mean(merged['sentiment_crowd'] == merged['sentiment_gold'])
    print(f"Agreement Score (Accuracy): {agreement:.2%}")
    
    # confusion matrix
    confusion = pd.crosstab(
        merged['sentiment_gold'],
        merged['sentiment_crowd'],
        rownames=['Gold'],
        colnames=['Crowd']
    )

    save_confusion_matrix(confusion)
    
def save_confusion_matrix(confusion):
    plt.figure(figsize=(8, 6))
    plt.title('Confusion Matrix (Gold vs Crowd)')

    im = plt.imshow(
        confusion,
        interpolation='nearest',
        cmap=plt.cm.Blues,
        vmin=0,
        vmax=confusion.values.max()
    )

    plt.colorbar(im)
    plt.xlabel('Crowd')
    plt.ylabel('Gold')

    # Add text annotations
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            value = confusion.iloc[i, j]
            plt.text(
                j, i, value,
                ha='center', va='center',
                color='white' if value > confusion.values.max() / 2 else 'black'
            )

    # Set tick labels
    plt.xticks(range(len(confusion.columns)), confusion.columns)
    plt.yticks(range(len(confusion.index)), confusion.index)

    plt.tight_layout()
    plt.savefig('PA3/Graphs/confusion_matrix.png')
    plt.clf()

def main():
    # load data
    crowd_path = 'PA3/data/crowdsourced_train.csv'
    gold_path = 'PA3/data/gold_train.csv'
    crowd_df = pd.read_csv(crowd_path, sep='\t')
    gold_df = pd.read_csv(gold_path, sep='\t')
    
    print("--- Unique Labels (Crowdsourced) ---")
    crowd_df['sentiment'] = crowd_df['sentiment'].str.lower().str.strip()
    print(crowd_df['sentiment'].unique().tolist())

    # clean data
    crowd_df = clean_labels(crowd_df)

    print("\n--- Distribution (Crowdsourced) ---")
    print(crowd_df['sentiment'].value_counts(normalize=True))
    crowd_df['sentiment'].value_counts().plot(kind='bar', title='Crowdsourced Label Distribution')
    plt.savefig('PA3/Graphs/label_distribution.png') 
    plt.clf()
    
    compare_datasets(crowd_df, gold_df)


if __name__ == "__main__":
    main()