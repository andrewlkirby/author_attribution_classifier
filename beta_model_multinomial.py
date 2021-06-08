import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

seed = 333

val = 9000

y1 = [1] * int((val/3))
y2 = [2] * int((val/3))
y3 = [3] * int((val/3))

y = y1 + y2 + y3

np.random.seed(seed)

x1 = (np.random.randint(1, 6, size = (val, 1)))
x2 = (np.random.randint(1, 6, size = (val, 1)))
x3 = (np.random.randint(1, 6, size = (val, 1)))
x4 = (np.random.randint(1, 6, size = (val, 1)))
x5 = (np.random.randint(1, 6, size = (val, 1)))

X = np.c_[x1, x2, x3, x4, x5]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed)

clf = LogisticRegression(n_jobs=1, multi_class='multinomial', 
                         solver ='newton-cg').fit(X_train, y_train)

print("\n")
print("beta model score: ", clf.score(X_test, y_test))