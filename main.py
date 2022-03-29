import argparse
from modules.interface import * 
parser = argparse.ArgumentParser(description='Program umożliwiający zbadanie działania algorytmów SVC, MLP i KNN na rzeczywistych danych. Stara się on rozwiązać problem rozpoznawania, czy dana wiadomośc email jest spamem czy nie.', 
epilog="Autor: \n Rafał Rzewucki - 248926", add_help=False)
parser.add_argument('-m', "--mode", help="Określa metodę uruchoienia programu. Wartości: full|test|train|visualise-data", default='full', required=True)
parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='Pokazuje menu pomocy i kończy działanie programu')
args = parser.parse_args()

print(args.mode)
mode = args.mode

if args.mode == 'visualise-data':
	runDataVisualise(args)
