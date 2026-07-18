# Cat Breed Classifier 
**Autore:** Mattias Manograsso
**Data:** Luglio 2026  

Il presente repository contiene l'infrastruttura software per la classificazione supervisionata delle razze feline, implementata tramite una pipeline end-to-end in linguaggio Python. Il sistema gestisce l'intero ciclo di vita del dato, dall'ingegneria delle feature alla validazione statistica, fino all'inferenza finale.

---

## Struttura del Progetto

```text
.
├── grafici/                     # Directory degli output di diagnostica visiva
│   ├── distribuzione_razze.png  # Frequenze delle classi nel training set
│   ├── heatmap_correlazione.png # Matrice di correlazione di Pearson (feature numeriche)
│   ├── peso_per_razza.png       # Distribuzione del peso corporeo per singola razza
│   ├── mantello_per_razza.png   # Tabulazione incrociata colore mantello / razza
│   └── matrice_confusione.png   # Performance di classificazione sul Validation Set
├── cats_dataset.csv             # Dataset originario etichettato per l'addestramento (Training Set)
├── test_set.csv                 # Dataset non etichettato per la valutazione finale (Test Set)
├── gatti.py                     # Script principale contenente la pipeline computazionale
├── predictions.csv              # Output finale strutturato con le predizioni del test set
└── README.md                    # Documentazione tecnica del progetto

```

---

## Pipeline di Pre-elaborazione e Mitigazione del Data Leakage

La funzione di pulizia dei dati `clean_data` è progettata per operare in modo asimmetrico tra la fase di addestramento (*fit*) e quella di inferenza (*transform*), prevenendo fenomeni di data leakage attraverso il passaggio del dizionario `fit_params`.

1. **Normalizzazione delle Stringhe:** Le variabili categoriali subiscono una trasformazione standardizzata che prevede la conversione in caratteri minuscoli, la rimozione degli spazi bianchi marginali (`.str.strip()`), la sostituzione degli spazi interni con il carattere underscore e la conversione delle stringhe nulle in costanti `np.nan`.
2. **Mappatura e Vincoli di Dominio:**
* La feature `sesso` viene ricondotta a tre categorie atomiche (`maschio`, `femmina`, `altro`) tramite dizionario di traduzione esplicito.
* Le feature `lunghezza_pelo`, `livello_attivita` e `frequenza_miagolio` vengono vincolate ai rispettivi domini di validità mediante filtraggio esplicito; i valori non conformi o mancanti vengono imputati con la moda calcolata sul set di addestramento.
* La feature `sterilizzato` viene convertita in formato booleano intero (0/1).


3. **Trattamento degli Outlier e Imputazione Numerica:** Per le variabili `eta_anni` e `peso_kg` viene applicata una tecnica di **Clipping** vincolata al 1° e al 99° percentile (con limiti fisici rigidi stabiliti a [0, 30] per l'età e [0.3, 15] per il peso) per neutralizzare l'impatto di anomalie biologiche. I valori mancanti sono successivamente sostituiti con la mediana campionaria del training set.

---

## Risoluzione Supervisionata del Target Anomalo ("Alien")

Per non alterare la distribuzione statistica del target con rimozioni arbitrarie o imputazioni ingenue, le etichette contrassegnate dalla stringa fittizia `"Alien"` nel training set vengono corrette tramite un approccio predittivo supervisionato:

* Viene isolata una sottomatrice di record aventi razza nota.
* Su questi dati viene addestrato un modello ausiliario `RandomForestClassifier` ($300$ stimatori, `max_depth=10`).
* Il modello esegue l'inferenza sui record contrassegnati come `"Alien"`, sovrascrivendo l'etichetta originaria con la classe statisticamente più probabile.

---

## Analisi del Rumore Intrinseco (Bayes Error Rate Limit)

Il sistema include una funzione di diagnostica che analizza le collisioni deterministiche all'interno dello spazio delle feature (record identici associati a razze differenti). Calcolando la quota maggioritaria all'interno di ciascun gruppo di feature identiche, viene definito il **Tetto Teorico di Accuratezza** raggiungibile su questo dataset. Tale valore funge da benchmark assoluto per valutare l'efficienza dei classificatori.

---

## Strategia di Modellazione e Hyperparameter Tuning

Il dataset emendato viene ripartito in un sottoinsieme di **Training (80%)** e uno di **Validation (20%)**, preservando rigorosamente la stratificazione delle classi tramite `train_test_split`.

La ricerca degli iperparametri ottimali viene condotta tramite **GridSearchCV** associata a una validazione incrociata **StratifiedKFold**. Il numero di fold viene calcolato dinamicamente (da 2 a 5) in base alla numerosità della classe meno frequente nel training set per garantire la stabilità matematica del processo.

Vengono messi a confronto due modelli competitivi all'interno di specifiche pipeline (comprensive di `StandardScaler` e `OneHotEncoder` denso):

* **DecisionTreeClassifier:** Ottimizzato su parametri di profondità (`max_depth`), granularità dei nodi foglia (`min_samples_leaf`) e criteri di scissione (`gini`, `entropy`).
* **RandomForestClassifier:** Ottimizzato sul numero di stimatori (`n_estimators`), profondità degli alberi (`max_depth`) e dimensione minima delle foglie (`min_samples_leaf`).

Il modello che registra il punteggio di accuratezza media cross-validata più elevato viene eletto per la fase successiva.

---

## Performance Sperimentali (Validation Set)

I parametri metrici rilevati sul Validation Set locale prima del processo di deployment finale hanno registrato i seguenti valori:

* **Accuracy Globale:** 0.XX *(Sostituire con il valore effettivo stampato a terminale)*
* **F1-Score (Weighted):** 0.XX *(Sostituire con il valore effettivo stampato a terminale)*
* **Accuratezza CV sull'intero Training Set:** 0.XX *(Sostituire con il valore effettivo stampato a terminale)*

L'analisi degli errori di classificazione e il bilanciamento tra precision e recall per ciascuna classe sono documentati analiticamente nel grafico `matrice_confusione.png`.

---

## Generazione dell'Output e Inferenza Finale

Previamente all'applicazione sul test set, il modello ottimizzato viene clonato e **riaddestrato sul 100% dei dati di training disponibili** al fine di massimizzare la copertura informativa.

Le predizioni sul file `test_set.csv` vengono convertite nei nomi testuali delle razze mediante la funzione di inversione del `LabelEncoder`. L'output viene esportato nel file `predictions.csv`, strutturato esclusivamente con le colonne `ID` (convertito forzatamente in formato nativo intero) e `razza_prevista`, in totale conformità con i requisiti del sistema di correzione automatica.

```

```
