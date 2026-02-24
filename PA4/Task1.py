from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

X1 = [{'city':'Gothenburg', 'month':'July'},
      {'city':'Gothenburg', 'month':'December'},
      {'city':'Paris', 'month':'July'},
      {'city':'Paris', 'month':'December'}]
Y1 = ['rain', 'rain', 'sun', 'rain']
# Month (Y)
#   ^
# 1 |  (0,1) RAIN         (1,1) RAIN
#   |         * *
#   |          \
#   |           \  <-- Decision Boundary (A straight line works!)
#   |            \
# 0 |  (0,0) RAIN * (1,0) SUN  *
#   +-------------------------------------> City (X)
#      0 (Gothenburg)     1 (Paris)

X2 = [{'city':'Sydney', 'month':'July'},
      {'city':'Sydney', 'month':'December'},
      {'city':'Paris', 'month':'July'},
      {'city':'Paris', 'month':'December'}]
Y2 = ['rain', 'sun', 'sun', 'rain']
# Month (Y)
#   ^
# 1 |  (0,1) SUN  * (1,1) RAIN *
#   |              \     /
#   |               \   / 
#   |                X   <-- No single straight line can
#   |               /   \    put both SUNs on one side 
# 0 |  (0,0) RAIN */     \(1,0) SUN  *
#   +-------------------------------------> City (X)
#      0 (Sydney)         1 (Paris)

classifier1 = make_pipeline(DictVectorizer(), Perceptron(max_iter=10))
classifier1.fit(X1, Y1)
guesses1 = classifier1.predict(X1)
print(accuracy_score(Y1, guesses1))

classifier2 = make_pipeline(DictVectorizer(), Perceptron(max_iter=10))
#classifier2 = make_pipeline(DictVectorizer(), LinearSVC())
classifier2.fit(X2, Y2)
guesses2 = classifier2.predict(X2)
print(accuracy_score(Y2, guesses2))