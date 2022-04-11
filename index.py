import argparse
from modules.utils import * 
from modules.interface import * 

parser = argparse.ArgumentParser(description='Program umożliwiający zbadanie działania algorytmów SVC, MLP i KNN na rzeczywistych danych. Stara się on rozwiązać problem rozpoznawania, czy dana wiadomośc email jest spamem czy nie.', 
epilog="Autor: \n Rafał Rzewucki - 248926\n Dodatkowe opcje dla algorytmów: \n SVC: \n \t -o1 - jądro \n \t -o2 - randomState  \n MLP: \n \t -o1 - alfa \n \t -o2 - maxIter  \n KNN: \n \t -o1 - ilość sąsiadów",
 add_help=False)
parser.add_argument('-m', "--mode", help="Określa metodę uruchoienia programu. Wartości: train|visualise|run", default='run', required=True)
parser.add_argument('-a', "--algorithm", help="Określa używany algorytm. Wartości: SVC|MLP|KNN", default='', required=True)
parser.add_argument('-f', "--file", help="Określa nazwę pliku do którego będzie zapisywane wyjście (Bez rozszerzenia!)", default='', required=False)
parser.add_argument('-d', "--data", help="Określa nazwę pliku z którego ma pobierać dane do nauki", default='', required=False)
parser.add_argument('-o1', "--option1", help="Opcja pierwsza - używana do przekazania argumentów do modelu (Opis na dole)", default='', required=False)
parser.add_argument('-o2', "--option2", help="Opcja druga - używana do przekazania argumentów do modelu (Opis na dole)", default='', required=False)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,help='Pokazuje menu pomocy i kończy działanie programu')
args = parser.parse_args()

if args.mode == 'train':
	log('Tryb uczenia')
	trainClassifier(args)
elif args.mode == 'visualise':
	log('Tryb wizualizacji')
	visualiseClassifier(args)
elif args.mode == 'run':
	log('Tryb uruchamiania algorytmu')
	runClassifier(args)
else:
	log("Nieznany tryb uruchomienia: " + args.mode)