import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from utils import clean_labels

def train_and_evaluate(train_df, test_df, system_name):
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    print(f"\nTraining model on {system_name}...")
    model.fit(train_df['text'], train_df['sentiment'])
    
    print("Testing model...")
    predictions = model.predict(test_df['text'])
    
    accuracy = accuracy_score(test_df['sentiment'], predictions)
    print(f"Accuracy: {accuracy:.2%}")
    
    print(classification_report(test_df['sentiment'], predictions))
    return accuracy

def main():
    try:
        # Load data
        crowd_df = pd.read_csv('PA3/data/crowdsourced_train.csv', sep='\t')
        gold_df = pd.read_csv('PA3/data/gold_train.csv', sep='\t')
        test_df = pd.read_csv('PA3/data/test.csv', sep='\t')
        
        crowd_df = clean_labels(crowd_df)
        gold_df = clean_labels(gold_df)
        
        crowd_acc = train_and_evaluate(crowd_df, test_df, "Crowdsourced Training")
        gold_acc = train_and_evaluate(gold_df, test_df, "Gold Standard Training")
        
        print("\n" + "="*30)
        print(f"Crowd Accuracy: {crowd_acc:.2%}")
        print(f"Gold Accuracy:  {gold_acc:.2%}")
        print(f"Improvement:    {gold_acc - crowd_acc:.2%}")
        print("="*30)

    except FileNotFoundError:
        print("Error: Files not found. Check your paths.")

if __name__ == "__main__":
    main()