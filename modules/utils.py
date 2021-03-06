import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import time
import codecs, json

# KRZYWE UCZENIA
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
import pickle

# KRZYWE UCZENIA
def plot_learning_curve(
    model,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 0.5, 10),
):
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(20, 5), constrained_layout=True)

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Ilość danych uczących")
    axes[0].set_ylabel("Wynik")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        model,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Wynik uczenia"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Wynik uczenia z podziałem (Walidacja krzyżowa)"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Ilość danych uczących")
    axes[1].set_ylabel("Czas uczenia")
    axes[1].set_title("Skalowalność modelu")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("Czas uczenia")
    axes[2].set_ylabel("Wynik")
    axes[2].set_title("Wydajność modelu")

    return plt

def getOutputFilePath(fileName):
	return './output/' + fileName;

def getInputFilePath(fileName):
	return './input/' + fileName;

def getAlgOptions(args):
	if args.algorithm == 'SVC':
		# jądro
		if args.option1 == '':
			kernel = 'linear'
			log('Nie podano jądra. Używam domyślnego: "' + kernel + '"')
		else:
			kernel = args.option1
		#C
		if args.option2 == '':
			C = 20
			log('Nie podano parametru C. Używam domyślnego: ' + str(C))
		else:
			C = int(args.option2)
		return kernel, C

	elif args.algorithm == 'MLP':
		# hidden_layer_sizes
		if args.option1 == '':
			hidden_layer_sizes = (3, 2)
			log('Nie podano parametru "hidden_layer_sizes". Używam domyślnego: ' + str(hidden_layer_sizes))
		else:
			hidden_layer_sizes = int(args.option1)
		#maxIter
		if args.option2 == '':
			maxIter = 100
			log('Nie podano maxIter. Używam domyślnego: ' + str(maxIter))
		else:
			randomState = int(args.option2)
		return hidden_layer_sizes, maxIter
	
	elif args.algorithm == 'KNN':
		#nNeighbors
		if args.option1 == '':
			nNeighbors = 5
			log('Nie podano ilości sąsiadów. Używam domyślnej wartości: ' + str(nNeighbors))
		else:
			randomState = int(args.option1)
		return nNeighbors

def getFileName(args):
	if args.file == '':
		return 'default_' + str(time.time())
	return args.file

def getDataFileName(args):
	if args.data == '':
		return 'full.csv'
	args.data

def log(mess):
	time = datetime.now()
	print('['+str(time.strftime("%d-%m-%y"))+' '+str(time.strftime("%H:%M:%S"))+'] \t' +mess)


def saveModelToFile(args, model):
	filepath = getOutputFilePath(getFileName(args)+'.sav')
	pickle.dump(model, open(filepath, 'wb'))
	log('Model został zapisany do "' + filepath + '"')

def saveGridSearchToFile(args, results):
	#konwertowanie danych
	for key in results.keys():
		if "list" not in type(results[key]).__name__:
			results[key] = results[key].tolist();

	filepath = getOutputFilePath(getFileName(args)+'.json')

	json.dump(results, codecs.open(filepath, 'w', encoding='utf-8'), 
		separators=(',', ':'), 
		sort_keys=True, 
		indent=4)

	log('Wynik został zapisany do "' + filepath + '"')