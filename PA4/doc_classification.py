import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from Task3 import Pegasos
from Task4 import LogisticRegression
from bonus import Pegasos_opt, Pegasos_sparse_opt, Pegasos_sparse_scale_opt

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
    N_p = 50
    lam_p = 1 /(N_p * N_p)

    Pegasos_pipeline = make_pipeline(
        TfidfVectorizer(),
        SelectKBest(k=1000),
        Normalizer(),
        Pegasos(n_iter=N_p, lambda_=lam_p),
    )

    t0 = time.time()
    Pegasos_pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    Yguess_svc = Pegasos_pipeline.predict(Xtest)
    print('--- Task 3: Pegasos ---')
    print('Training time: {:.2f} sec.'.format(t1 - t0))
    print('Accuracy:      {:.4f}'.format(accuracy_score(Ytest, Yguess_svc)))
    
    Pegasos_opt_pipeline = make_pipeline(
        TfidfVectorizer(),
        SelectKBest(k=1000),
        Normalizer(),
        Pegasos_opt(n_iter=N_p, lambda_=lam_p),
    )

    t0 = time.time()
    Pegasos_opt_pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    Yguess_svc = Pegasos_opt_pipeline.predict(Xtest)
    print('--- Bonus Task: Optimized Pegasos ---')
    print('Training time: {:.2f} sec.'.format(t1 - t0))
    print('Accuracy:      {:.4f}'.format(accuracy_score(Ytest, Yguess_svc)))
    
    Pegasos_sparse_opt_pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        Normalizer(),
        Pegasos_sparse_opt(n_iter=N_p, lambda_=lam_p),
    )

    t0 = time.time()
    Pegasos_sparse_opt_pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    Yguess_svc = Pegasos_sparse_opt_pipeline.predict(Xtest)
    print('--- Bonus Task: Optimized Pegasos (Sparse) ---')
    print('Training time: {:.2f} sec.'.format(t1 - t0))
    print('Accuracy:      {:.4f}'.format(accuracy_score(Ytest, Yguess_svc)))
    
    Pegasos_sparse_scale_opt_pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        Normalizer(),
        Pegasos_sparse_scale_opt(n_iter=N_p, lambda_=lam_p),
    )

    t0 = time.time()
    Pegasos_sparse_scale_opt_pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    Yguess_svc = Pegasos_sparse_scale_opt_pipeline.predict(Xtest)
    print('--- Bonus Task: Optimized Pegasos (Sparse Scale) ---')
    print('Training time: {:.2f} sec.'.format(t1 - t0))
    print('Accuracy:      {:.4f}'.format(accuracy_score(Ytest, Yguess_svc)))
    
    N_lr = 30
    lam_lr = 1 /  N_lr

    lr_pipeline = make_pipeline(
        TfidfVectorizer(),
        SelectKBest(k=1000),
        Normalizer(),
        LogisticRegression(n_iter=N_lr, lambda_=lam_lr),
    )

    t0 = time.time()
    lr_pipeline.fit(Xtrain, Ytrain)
    t1 = time.time()
    Yguess_lr = lr_pipeline.predict(Xtest)
    print()
    print('--- Task 4: Logistic Regression ---')
    print('Training time: {:.2f} sec.'.format(t1 - t0))
    print('Accuracy:      {:.4f}'.format(accuracy_score(Ytest, Yguess_lr)))


