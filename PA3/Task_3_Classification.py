import warnings

from sklearn.naive_bayes import MultinomialNB
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from utils import clean_labels, load_data, plot_confusion_matrix


def train_and_evaluate(train_df, test_df, name):
    x_train, y_train = train_df['text'], train_df['sentiment']
    x_test, y_test = test_df['text'], test_df['sentiment']

    # baseline
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(x_train, y_train)
    baseline_acc = accuracy_score(y_test, dummy.predict(x_test))
    print(f"Baseline: {baseline_acc:.2%}")

    # model 1: logistic regression
    logreg = Pipeline([
        ('tfidf', TfidfVectorizer(sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=3000, random_state=42))
    ])
    logreg_grid = GridSearchCV(logreg, {
        'tfidf__ngram_range': [(1,1), (1,2)],
        'clf__C': [0.5, 1, 5, 10]
    }, cv=5, n_jobs=-1)
    logreg_grid.fit(x_train, y_train)

    # model 2: linear svm
    svm = Pipeline([
        ('tfidf', TfidfVectorizer(sublinear_tf=True)),
        ('clf', LinearSVC(random_state=42, dual='auto', max_iter=5000))
    ])
    svm_grid = GridSearchCV(svm, {
        'tfidf__ngram_range': [(1,1), (1,2)],
        'clf__C': [0.1, 0.5, 1, 5]
    }, cv=5, n_jobs=-1)
    svm_grid.fit(x_train, y_train)
    
    # model 3: naive bayes classifier
    nby = Pipeline([
        ('tfidf', TfidfVectorizer(sublinear_tf=True)),
        ('clf', MultinomialNB())
    ])
    nby_grid = GridSearchCV(nby, {
        'tfidf__ngram_range': [(1,1), (1,2)],
        'clf__alpha': [0.1, 0.5, 1,
                          5, 10]
    }, cv=5, n_jobs=-1)
    nby_grid.fit(x_train, y_train)

    # print all model scores
    results = [
        ('logistic regression', logreg_grid),
        ('linear svm', svm_grid),
    ]
    print("\nmodel comparison (5-fold cv):")
    for model_name, grid in results:
        print(f"  {model_name:20s}  cv={grid.best_score_:.2%}  {grid.best_params_}")

    # pick the best one
    best_name, best_grid = max(results, key=lambda x: x[1].best_score_)
    best = best_grid.best_estimator_
    print(f"  -> winner: {best_name}")

    # Test
    preds = best.predict(x_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc:.2%}")
    print(classification_report(y_test, preds))
    plot_confusion_matrix(y_test, preds, name, f"result_{name.lower().replace(' ', '_')}")

    return acc, baseline_acc


def main():
    crowd_df, gold_df, test_df = load_data()
    if crowd_df is None: return

    crowd_df = clean_labels(crowd_df)
    gold_df = clean_labels(gold_df)

    crowd_acc, base = train_and_evaluate(crowd_df, test_df, "crowdsourced")
    gold_acc, _ = train_and_evaluate(gold_df, test_df, "gold standard")

    print(f"Crowd: {crowd_acc:.2%}  |  Gold: {gold_acc:.2%}  |  Diff: {gold_acc - crowd_acc:+.2%}")
    print(f"baseline: {base:.2%}")


if __name__ == "__main__":
    main()
