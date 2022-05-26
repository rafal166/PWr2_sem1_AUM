# Projekt przygotowany w ramach kursu **Algorytmy Uczenia Maszynowego** na **Politechnice Wrocławskiej**

## Opis
Program został stworzony w celu zbadania działania wybranych algorytmów uczenia maszynowego. Rozpatrywanym problemem jest rozpoznawanie spamu w wiadomościach e-mail zatem jest to zagadnienie klasyfikacji.

Program posiada zaimplementowane 3 algorytmy uczenia maszynowego:
- SVC
- MLP
- KNN

## Tryby działania 
Program można uruchomić w 3 różnych trybach:
### Tryb nauki
Ten tryb pozwala nauczyć model na podstawie podanych danych. Dane podawane są w pliku CSV w formacie zdefiniowanym w przykładowym pliku __full.csv__ w katalogu __data__.
Za pomocą wbudowanych opcji programu można zdefinować parametry danego algorytmu -> patrz opcję __--help__ programu.
Nauczone modele zapisywane są w katalogu **output** w pliku o nazwie zdefiniowanej parametrem **--file**.
#### Przykładowe uruchomienie:
```bash
py index.py -m train -a MLP -f output_model
```


### Tryb uruchamiania algorytmu
Odczytuje treść mejla podaną w pliku, którego nazwa jest podawana w parametrze -i. 
#### Przykładowe uruchomienie:
```bash
py index.py -m run -f SVC -i not_spam.txt
```

### Tryb wizualizacji
Ten tryb pozwala na zwizualizowanie procesu uczenia. Dane do uczenia przekazywane są tak samo jak w **Trybie nauki**.
Wyjściem są wykresy opisujące parametry pracy algorytmu.
#### Przykładowe uruchomienie:
```bash
py index.py -m visualise -a SVC -f SVC-visualisation
```

### Tryb szukania parametrów - Grid Search
Ten tryb pozwala na uruchomienie mechanizmu poszukiwania optymalnych parametrów danego algorytmu.
#### Przykładowe uruchomienie:
```bash
py index.py -m grid -a SVC -f SVC-gird
```

### Tryb pobierania statystyk algorytmu
Ten tryb pozwala na uruchomienie algorytmu i zebranie statystyk na jego temat. Trzeba podać model w parametrze -f
#### Przykładowe uruchomienie:
```bash
py index.py -m stats -f SVC
```

## Problemy
Wykryte problemy proszę zgłaszać w ramach tego repozytorium