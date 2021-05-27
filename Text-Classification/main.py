import nltk
from nltk.corpus import stopwords
import re
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions

nltk.download('stopwords')


def preprocess(data):
    processedData = []
    stop_words = set(stopwords.words('english'))
    for i in range(len(data)):
        newData = re.sub(r'\W', ' ', str(data[i]))
        newData = re.sub(r'\s+[a-zA-Z]\s+', ' ', newData)
        newData = re.sub(r'\s+', ' ', newData)
        newData = newData.split(' ')
        newData = [word for word in newData if word not in stop_words]
        newData = ' '.join([str(elem) for elem in newData])
        processedData.append(newData)
    return processedData


def plot(data, labels):
    pca = PCA(n_components=2, random_state=1)
    reduced_X = pca.fit_transform(data)
    NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    NN.fit(reduced_X, labels)
    plot_decision_regions(reduced_X, labels, clf=NN, legend=2)
    plt.xlim(-0.1, 0.1)
    plt.ylim(-0.1, 0.1)

    plt.title("Decision Boundary using MLP Classifier")
    plt.show()


# uncomment to print whole array without truncation
# np.set_printoptions(threshold=np.inf)

data = sklearn.datasets.load_files("txt_sentoken")
X, y = data.data, data.target
X = preprocess(X)

tf_idf = TfidfVectorizer(use_idf=True)
X = tf_idf.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=1)

plot(X_test, y_test)

nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
nn.fit(X_train, y_train)

y_predict = nn.predict(X_test)
print("Accuracy of Model: ", nn.score(X_test, y_test) * 100, '%')

while True:
    test = [input("Enter test\n")]
    if test == 1:
        break
    result = nn.predict(tf_idf.transform(test))
    print("Positive" if result == 1 else "Negative")
