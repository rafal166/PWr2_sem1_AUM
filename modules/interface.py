from modules.data import * 
from modules.utils import * 
import pickle

# ALGORYTMY UCZENIA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

def trainClassifier(args):
	log('Pobiernie danych do nauki z pliku "' + getDataFileName(args) + '"')
	X_train, X_test, Y_train, Y_test = getDataToLearn(getDataFileName(args))

	if args.algorithm == 'SVC':
		kernel, randomState = getAlgOptions(args)
		classifier = SVC(kernel = kernel, random_state = randomState)

	elif args.algorithm == 'MLP':
		hiddenLayerSizes, maxIter = getAlgOptions(args)
		classifier = MLPClassifier(hidden_layer_sizes=hiddenLayerSizes, max_iter=maxIter)

	elif args.algorithm == 'KNN':
		nNeighbors = getAlgOptions(args)
		classifier = KNeighborsClassifier(n_neighbors=nNeighbors)

	else:
		log('Nie znaleziono podanego algorytmu: ' + args.algorithm)
		exit()

	log('Rozpoczynam uczenie algorytmu ' + args.algorithm)
	classifier.fit(X_train, Y_train)
	log('Zakończono uczenie algorytmu ' + args.algorithm)

	saveModelToFile(args, classifier)
	log('Zakończono proces uczenia')

def statisticClassifier(args):
	# ładowanie modelu
	log('Rozpoczynam pobieranie statystyk klasyfikatora: '+ args.file)

	filePath = getOutputFilePath(getFileName(args)+'.sav')
	classifier = pickle.load(open(filePath, 'rb'))
	#pobieranie danych do testowania
	X_train, X_test, Y_train, Y_test = getDataToLearn(getDataFileName(args), 0.2)

	Y_predicted = classifier.predict(X_test)

	accuracy = accuracy_score(Y_test, Y_predicted)
	precission = precision_score(Y_test, Y_predicted, average=None)
	f1score = f1_score(Y_test, Y_predicted, average=None)
	confusionMatrix = confusion_matrix(Y_test, Y_predicted)

	log('Accuracy: '+ str(accuracy))
	log('Precission: '+ str(precission))
	log('F1Score: '+ str(f1score))
	log('ConfusionMatrix: '+ str(confusionMatrix))

	log('Koniec pobierania statystyk klasyfikatora: '+ args.file)

def runClassifier(args):
	# ładowanie modelu
	log('Rozpoczynam uruchamianie klasyfikatora: '+ args.file)

	filePath = getOutputFilePath(getFileName(args)+'.sav')
	classifier = pickle.load(open(filePath, 'rb'))
	#pobieranie danych do testowania

	with open(getInputFilePath(args.input)) as f:
		content = f.read()

	# zamiana na wartość liczbową
	cv = getCountVectorizer() 
	X_content = cv.transform([content])
 

	Y_predicted = classifier.predict(X_content)

	log('Result: '+ str(Y_predicted))

	log('Koniec uruchamiania klasyfikatora: '+ args.file)


def visualiseClassifier(args):
	log('Rozpoczynam wizualizowanie klasyfikatora: '+ args.algorithm)

	X_train, X_test, Y_train, Y_test = getDataToLearn(getDataFileName(args), 0.00000000000001)

	log('Określanie parametrów klasyfikatora')
	if args.algorithm == 'SVC':
		kernel, C = getAlgOptions(args)
		classifier = SVC(kernel = kernel, C = C)
		title = 'Krzywe uczenia algorytmu SVC (jądro "'+kernel+'")'
	elif args.algorithm == 'MLP':
		hidden_layer_sizes, maxIter = getAlgOptions(args)
		classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=maxIter)
		title = 'Krzywe uczenia algorytmu MLP (hidden_layer_sizes: '+str(hidden_layer_sizes)+', ilość iteracji: '+str(maxIter)+')'
	elif args.algorithm == 'KNN':
		nNeighbors = getAlgOptions(args)
		classifier = KNeighborsClassifier(n_neighbors=nNeighbors)
		title = 'Krzywe uczenia algorytmu KNN (ilość sąsiadów: '+str(nNeighbors)+')'
	else:
		log('Nie znaleziono podanego algorytmu: ' + args.algorithm)
		exit()

	log('Rozpoczynanie tworzenia wizualizacji procesu uczenia')
	# plot krzywych uczenia wybranego algorytmu
	plot_learning_curve(classifier, title, X_train, Y_train)
	log('Proces uczenia zakończony')

	fig1 = plt.gcf()
	fig1.set_size_inches(15, 18, forward=True)
	filePath = getOutputFilePath(getFileName(args)+'.png')
	fig1.savefig(filePath, dpi=100)
	log('Zapisano wykresy do pliku: "' + filePath + '"')
	plt.show()

	log('Koniec wizualizacji klasyfikatora')

def gridSearchClassifier(args):
	log('Rozpoczynam szukanie parametrów klasyfikatora')
	X_train, X_test, Y_train, Y_test = getDataToLearn(getDataFileName(args), 0.00001)

	if args.algorithm == 'SVC':
		log('Wybrano klasyfikator SVC')
		classifier = SVC();
		parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[0.1, 1, 20]}
	elif args.algorithm == 'MLP':
		log('Wybrano klasyfikator MLP')
		classifier = MLPClassifier();
		parameters = {
			'hidden_layer_sizes': [(1,2,3), (3,2)],
			'max_iter': [100, 200, 400],
		}
	elif args.algorithm == 'KNN':
		log('Wybrano klasyfikator KNN')
		classifier = KNeighborsClassifier();
		parameters = {'n_neighbors': [1, 2, 3, 5, 8, 13]}
	else:
		log('Nie znaleziono podanego algorytmu: ' + args.algorithm)
		exit()


	#działanie...
	clf = GridSearchCV(estimator = classifier, param_grid = parameters, n_jobs = -1, verbose = 2)
	clf.fit(X_train, Y_train)

	# zapisywanie danych do pliku
	saveGridSearchToFile(args, clf.cv_results_)
	log('Koniec szukania parametrów klasyfikatora')