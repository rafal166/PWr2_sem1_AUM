from time import time
from modules.utils import * 

def getAlgOptions(args):
	if args.algorithm == 'SVC':
		# jądro
		if args.option1 == '':
			kernel = 'rbf'
			log('Nie podano jądra. Używam domyślnego: "' + kernel + '"')
		else:
			kernel = args.option1
		#randomState
		if args.option2 == '':
			randomState = 0
			log('Nie podano randomState. Używam domyślnego: ' + str(randomState))
		else:
			randomState = int(args.option2)
		return kernel, randomState

	elif args.algorithm == 'MLP':
		# alfa
		if args.option1 == '':
			alpha = 1
			log('Nie podano parametru "alfa". Używam domyślnego: ' + str(alpha))
		else:
			alpha = int(args.option1)
		#maxIter
		if args.option2 == '':
			maxIter = 3
			log('Nie podano maxIter. Używam domyślnego: ' + str(maxIter))
		else:
			randomState = int(args.option2)
		return alpha, maxIter
	
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
		return 'default_' + time()
	return args.file

def getDataFileName(args):
	if args.data == '':
		return 'full.csv'
	args.data