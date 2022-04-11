from modules.data import * 
from modules.utils import * 
import pickle

# ALGORYTMY UCZENIA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

def trainClassifier(args):
	log('Pobiernie danych do nauki z pliku "' + getDataFileName(args) + '"')
	X_train, X_test, Y_train, Y_test = getDataToLearn(getDataFileName(args))

	if args.algorithm == 'SVC':
		kernel, randomState = getAlgOptions(args)
		classifier = SVC(kernel = kernel, random_state = randomState)
		classifier.fit(X_train, Y_train)

	elif args.algorithm == 'MLP':
		alpha, maxIter = getAlgOptions(args)
		classifier = MLPClassifier(alpha=alpha, max_iter=maxIter)

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

def runClassifier(args):
	# ładowanie modelu
	log('Rozpoczynam testowanie klasyfikatora: '+ args.algorithm)

	filePath = getOutputFilePath(getFileName(args)+'.sav')
	classifier = pickle.load(open(filePath, 'rb'))
	#pobieranie danych do testowania
	X_train, X_test, Y_train, Y_test = getDataToLearn(getDataFileName(args), 0.2)

	testScore = classifier.score(X_test, Y_test)
	log('Test score: '+ str(testScore))

	log('Koniec testu klasyfikatora: '+ args.algorithm)


def visualiseClassifier(args):
	log('Rozpoczynam wizualizowanie klasyfikatora: '+ args.algorithm)

	X_train, X_test, Y_train, Y_test = getDataToLearn(getDataFileName(args), 0.00000000000001)

	log('Określanie parametrów klasyfikatora')
	if args.algorithm == 'SVC':
		kernel, randomState = getAlgOptions(args)
		classifier = SVC(kernel = kernel, random_state = randomState)
		title = 'Krzywe uczenia algorytmu SVC (jądro "'+kernel+'")'
	elif args.algorithm == 'MLP':
		alpha, maxIter = getAlgOptions(args)
		classifier = MLPClassifier(alpha=alpha, max_iter=maxIter)
		title = 'Krzywe uczenia algorytmu MLP (alfa: '+str(alpha)+', ilość iteracji: '+str(maxIter)+')'
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
