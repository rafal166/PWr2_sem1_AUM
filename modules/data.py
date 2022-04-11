import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def getDataToLearn(file = 'full.csv', testSize = 0.2):
	pathToDataForProcess = './data/parsed/'+file
	# otwieranie danych do przetworzenia
	data = pd.read_csv(pathToDataForProcess)
	# data.info()

	# rozdzielanie danych na x i y
	X = data['email'].values
	Y = data['label'].values
	cv = CountVectorizer() 

	#rozdzielanie danych na treningowe i testowe
	X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=testSize, random_state=1)

	# Przerabianie tekstu na liczby
	X_train = cv.fit_transform(X_train)
	X_test = cv.transform(X_test)

	return X_train, X_test, Y_train, Y_test