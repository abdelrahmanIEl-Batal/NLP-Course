from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
import re
import sklearn
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from nltk.stem import WordNetLemmatizer


# nltk.download('stopwords')


def preprocess(data):
    processedData = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    for i in range(len(data)):
        newData = re.sub(r'\W', ' ', str(data[i]))
        newData = re.sub(r'\s+[a-zA-Z]\s+', ' ', newData)
        newData = re.sub(r'\s+', ' ', newData)
        newData = newData.split(' ')
        newData = [word for word in newData if word not in stop_words]
        newData = [lemmatizer.lemmatize(w) for w in newData]
        newData = ' '.join([str(elem) for elem in newData])
        processedData.append(newData)
    return processedData


def word_embedding(X):
    for i in range(len(X)):
        X[i] = nltk.word_tokenize(X[i])
    model = Word2Vec(X, window=5, size=128, min_count=4, iter=40)
    embedding_res = []
    for i in range(len(X)):
        word_embedding = []
        for word in X[i]:
            if word in model.wv.vocab:
                word_embedding.append(model.wv[word])
            else:
                continue
                word_embedding.append(np.zeros(80))
        average = np.mean(word_embedding, axis=0)
        embedding_res.append(average)
    return embedding_res


def printResults(model, name, X_test, y_predict):
    print("Accuracy using ", name, " Classifier is: ", model.score(X_test, y_test))
    print("Precision using ", name, " Classifier is: ",
          sklearn.metrics.precision_score(y_test, y_predict, average='binary'))
    print("Recall using ", name, " Classifier is: ",
          sklearn.metrics.recall_score(y_test, y_predict, average='binary'))
    print()

data = sklearn.datasets.load_files("txt_sentoken")
X, y = data.data, data.target
X = preprocess(X)

X = word_embedding(X)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=49)

model = MLPClassifier(solver='adam', alpha=1e-5, random_state=1, max_iter=1500)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
printResults(model, 'MLP', X_test, y_predict)

model = svm.SVC(kernel='rbf', max_iter=1000,random_state=49)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
printResults(model, 'SVM', X_test, y_predict)

model = LogisticRegression(random_state=49)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
printResults(model, 'Logistic Regression', X_test, y_predict)
