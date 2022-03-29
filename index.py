import csv
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import random
# Określanie jaki mam maksymalny rozmiar pola dostępny
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

		

# przygotowanie pierwszej porcji danych
def parseRawData():
	emails = [];

	#odczytywanie drugiej porcji danych
	pathToCsv = './data/raw/2/data_2.csv'
	file = open(pathToCsv, encoding='utf-8')
	csvreader = csv.reader(file)
	next(csvreader) # pomijanie headera CSV

	## odczytywanie kolejnych wierszy
	for row in csvreader:
		if row[1]:
			if row[0] == '0':
				label = 'ham'
			else:
				label = 'spam'
			emails.append([label, row[1]])
	file.close()

	#odczytywanie trzeciej porcji danych
	pathToCsv = './data/raw/3/data_3.csv'
	file = open(pathToCsv, encoding='utf-8')
	csvreader = csv.reader(file)
	next(csvreader) # pomijanie headera CSV

	## odczytywanie kolejnych wierszy
	for row in csvreader:
		emails.append(row)
	file.close()

	# zwracanie przygotowanych danych
	random.shuffle(emails)
	return emails


def saveParsedData(emails):
	pathToCsv = './data/parsed/full.csv'
	with open(pathToCsv, 'w', newline="", encoding='utf-8') as file:
		csvwriter = csv.writer(file)
		csvwriter.writerow(['label', 'email'])
		csvwriter.writerows(emails)

## Wstępne przygotowywanie mejli
# emails = parseRawData()
# saveParsedData(emails)

pathToDataForProcess = './data/parsed/full.csv'
# otwieranie danych do przetworzenia
data = pd.read_csv(pathToDataForProcess)
data.info()

# rozdzielanie danych na x i y
X = data['email'].values
y = data['label'].values

#rozdzielanie danych na treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=1234567)

# Przerabianie tekstu na liczby
cv = CountVectorizer() 
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

#uruchamianie algorytmu SVC
# print('SVC')
# svc_classifier = SVC(kernel = 'rbf', random_state = 0)
# svc_classifier.fit(X_train, y_train)
# print(svc_classifier.score(X_test,y_test))

#uruchamianie algorytmu MLP
print('MLP')
mlp_classifier = MLPClassifier(alpha=1, max_iter=3)
mlp_classifier.fit(X_train, y_train)
print(mlp_classifier.score(X_test,y_test))

#uruchamianie algorytmu KNN
# print('KNN')
# knn_classifier = KNeighborsClassifier(n_neighbors=6)
# knn_classifier.fit(X_train, y_train)
# print(knn_classifier.score(X_test,y_test))