import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def clean_labels(df):
    """
    normalizes sentiment labels by fixing typos and casing.
    """
    df['sentiment'] = df['sentiment'].str.lower().str.strip()
    
    # list of common typos found in the data
    neutral_mislabels = [
        'neutral?', 'nuetral', '_x0008_neutral', 'netural', 'netutral',
        'neural', 'neutrall', 'neugral', 'neutrla', 'nutral', 'neutra l', 'neutal',
        'neutrl', 'neutra', 'neutrl ', 'nutral', 'nuetral'
    ]
    positive_mislabels = [
        'postive', 'positve', 'positve ', 'positie', 'postitive',
        'npositive', 'positve'
    ]
    negative_mislabels = [
        'nedative', 'negtaive', 'negayive', 'negativ', 'negatve',
        'negativ ', 'nedative'
    ]
    
    # create mapping dictionary
    typos = {}
    for label in neutral_mislabels: typos[label] = 'neutral'
    for label in positive_mislabels: typos[label] = 'positive'
    for label in negative_mislabels: typos[label] = 'negative'

    df['sentiment'] = df['sentiment'].replace(typos)
    
    return df

def load_data():
    """
    loads the three datasets from the data folder.
    returns none if files are missing.
    """
    # use relative path to make sure it runs on any computer
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')

    try:
        crowd = pd.read_csv(os.path.join(data_dir, 'crowdsourced_train.csv'), sep='\t')
        gold = pd.read_csv(os.path.join(data_dir, 'gold_train.csv'), sep='\t')
        test = pd.read_csv(os.path.join(data_dir, 'test.csv'), sep='\t')
        return crowd, gold, test
    except FileNotFoundError as e:
        print(f"error: could not find files. details: {e}")
        return None, None, None

def ensure_graphs_dir():
    """
    creates a 'graphs' folder if it doesn't verify exist.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    graphs_dir = os.path.join(current_dir, 'Graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    return graphs_dir

def plot_confusion_matrix(y_true, y_pred, title, filename_suffix):
    """
    helper function to draw and save a confusion matrix.
    used for both task 2 (agreement) and task 3 (model evaluation).
    """
    graphs_dir = ensure_graphs_dir()
    
    # sorting to ensure consistent order: negative, neutral, positive
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    
    if set(unique_labels) == {'negative', 'neutral', 'positive'}:
        labels = ['negative', 'neutral', 'positive']
    else:
        labels = unique_labels

    # calculate matrix using sklearn
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    # plot and save
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(title)
    
    filename = f"confusion_matrix_{filename_suffix}.png"
    save_path = os.path.join(graphs_dir, filename)
    plt.savefig(save_path)
    print(f"saved graph to: {save_path}")
    plt.close()

def plot_label_distribution(df, title, filename_suffix):
    """
    simple bar chart to show how many positives, negatives, and neutrals there are.
    """
    graphs_dir = ensure_graphs_dir()
    
    plt.figure(figsize=(8, 6))
    # get percentages
    counts = df['sentiment'].value_counts(normalize=True)
    counts.plot(kind='bar', color='skyblue')
    
    plt.title(title)
    plt.ylabel('percentage')
    plt.ylim(0, 1) # y-axis from 0% to 100%
    
    # helper text on top of bars
    for i, v in enumerate(counts):
        plt.text(i, v + 0.02, f'{v:.1%}', ha='center')
    
    plt.tight_layout()
    save_path = os.path.join(graphs_dir, f"distribution_{filename_suffix}.png")
    plt.savefig(save_path)
    print(f"saved graph to: {save_path}")
    plt.close()
