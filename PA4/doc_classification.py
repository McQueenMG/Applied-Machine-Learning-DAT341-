import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from Task3 import Pegasos
from Task4 import LogisticRegression

def read_data(corpus_file):
    X = []
    Y = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            _, y, _, x = line.split(maxsplit=3)
            X.append(x.strip())
            Y.append(y)
    return X, Y


if __name__ == '__main__':
    X, Y = read_data('PA4/data/all_sentiment_shuffled.txt')

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,
                                                    random_state=0)
    N = 50
    lam = 1 / (N * N)

    svc_pipeline = make_pipeline(
        TfidfVectorizer(),
        SelectKBest(k=1000),
        Normalizer(),
        Pegasos(n_iter=N),
    )

    t0 = time.time()
    svc_pipeline.fit(Xtrain, Ytrain, pegasos__lambda_=lam)
    t1 = time.time()
    Yguess_svc = svc_pipeline.predict(Xtest)
    print('--- Task 3: Pegasos SVC ---')
    print('Training time: {:.2f} sec.'.format(t1 - t0))
    print('Accuracy:      {:.4f}'.format(accuracy_score(Ytest, Yguess_svc)))

    lr_pipeline = make_pipeline(
        TfidfVectorizer(),
        SelectKBest(k=1000),
        Normalizer(),
        LogisticRegression(n_iter=N),
    )

    t0 = time.time()
    lr_pipeline.fit(Xtrain, Ytrain,  logisticregression__lambda_=lam)
    t1 = time.time()
    Yguess_lr = lr_pipeline.predict(Xtest)
    print()
    print('--- Task 4: Logistic Regression ---')
    print('Training time: {:.2f} sec.'.format(t1 - t0))
    print('Accuracy:      {:.4f}'.format(accuracy_score(Ytest, Yguess_lr)))

