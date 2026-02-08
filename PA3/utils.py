import pandas as pd

def clean_labels(df):
    df['sentiment'] = df['sentiment'].str.lower().str.strip()
    
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
    
    typos = {}
    for label in neutral_mislabels: typos[label] = 'neutral'
    for label in positive_mislabels: typos[label] = 'positive'
    for label in negative_mislabels: typos[label] = 'negative'

    df['sentiment'] = df['sentiment'].replace(typos)
    
    return df