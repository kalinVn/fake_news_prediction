import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def get_dataset():
    dataset = pd.read_csv('datasets/train.csv')
    dataset = dataset.fillna('')
    dataset['content'] = dataset['author'] + ' ' + dataset['title']

    return dataset


def get_x(dataset):
    # separating the data and labels
    x = dataset.drop(colums='label', axis=1)
    return x


def get_y(dataset):
    y = dataset['label']
    return y


def get_porter_stem():
    # Stemming -> the process of reducing word to its Root  word
    # example -> actor, actress, acting --> the root word is act
    porter_stem = PorterStemmer()
    return porter_stem


def stemming(content):
    porter_stem = get_porter_stem()

    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [porter_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content


def get_stemming_dataset():
    dataset = get_dataset()
    dataset['content'] = dataset['content'].apply(stemming)

    return dataset


def predict(news_number):
    dataset = get_stemming_dataset()

    x = dataset['content'].values
    y = dataset['label'].values
    # print(x, y)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(x)
    x = vectorizer.transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    # accuracy score the training data
    x_train_prediction = model.predict(x_train)
    training_data_accuracy = accuracy_score(x_train_prediction, y_train)
    print('Accuracy score of the training data: ', training_data_accuracy)

    x_test_prediction = model.predict(x_test)
    test_data_accuracy = accuracy_score(x_test_prediction, y_test)
    print('Accuracy score of the test data: ', test_data_accuracy)

    # Predictive system
    x_new = x_test[news_number]
    prediction = model.predict(x_new)
    print(prediction)

    if(prediction[0] == 0):
        print('The news is Real')
    else:
        print('The new is fake')


predict(news_number=0)
