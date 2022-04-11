import csv
import sys
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
