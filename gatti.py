#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLASSIFICAZIONE DELLA RAZZA FELINA – HackersGen / Sorint.lab
Versione finale – metodo supervisionato corretto, senza forzature dei target.
"""

import os, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
RANDOM_STATE = 42
os.makedirs("grafici", exist_ok=True)


# ---------------------------------------------------------------------
# FUNZIONI DI SUPPORTO
# ---------------------------------------------------------------------
def _moda_sicura(serie, default):
    """Restituisce la moda di una Series, o *default* se vuota."""
    m = serie.mode()
    return m.iloc[0] if len(m) > 0 else default


def clean_data(df, fit_params=None):
    """
    Pulisce un dataframe di gatti.
    I parametri appresi (clipping, mediane, moda, categorie valide) vengono
    calcolati SOLO quando fit_params è None (training set) e poi riutilizzati
    per il test set, in modo da non causare data leakage.
    """
    df = df.copy()
    fitting = fit_params is None
    if fitting:
        fit_params = {}

    def normalizza(col):
        return (
            df[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(" ", "_", regex=False)
            .replace({"nan": np.nan, "none": np.nan, "": np.nan})
        )

    # --- sesso ---
    sesso_map = {
        "m": "maschio", "maschio": "maschio",
        "f": "femmina", "femmina": "femmina",
        "trans": "altro", "femminuccia": "femmina",
    }
    df["sesso"] = normalizza("sesso").map(sesso_map).fillna("altro")

    # --- lunghezza_pelo ---
    pelo = normalizza("lunghezza_pelo")
    valid_pelo = {"corto", "semilungo", "lungo"}
    pelo = pelo.where(pelo.isin(valid_pelo))
    if fitting:
        fit_params["moda_pelo"] = _moda_sicura(pelo, "corto")
    df["lunghezza_pelo"] = pelo.fillna(fit_params["moda_pelo"])

    # --- livello_attivita ---
    attivita = normalizza("livello_attivita")
    valid_attivita = {"molto_attivo", "attivo", "sedentario"}
    attivita = attivita.where(attivita.isin(valid_attivita))
    if fitting:
        fit_params["moda_attivita"] = _moda_sicura(attivita, "attivo")
    df["livello_attivita"] = attivita.fillna(fit_params["moda_attivita"])

    # --- frequenza_miagolio ---
    miagolio = normalizza("frequenza_miagolio")
    valid_miagolio = {"basso", "medio", "alto"}
    miagolio = miagolio.where(miagolio.isin(valid_miagolio))
    if fitting:
        fit_params["moda_miagolio"] = _moda_sicura(miagolio, "medio")
    df["frequenza_miagolio"] = miagolio.fillna(fit_params["moda_miagolio"])

    # --- sterilizzato (0/1) ---
    steril_raw = normalizza("sterilizzato")
    steril = steril_raw.map({"si": 1, "sì": 1, "yes": 1, "1": 1,
                             "no": 0, "not": 0, "0": 0})
    if fitting:
        fit_params["moda_sterilizzato"] = _moda_sicura(steril, 0)
    df["sterilizzato"] = steril.fillna(fit_params["moda_sterilizzato"]).astype(int)

    # --- patologia ---
    df["patologia"] = normalizza("patologia").fillna("nessuna")

    # --- colore_mantello ---
    df["colore_mantello"] = normalizza("colore_mantello").fillna("sconosciuto")

    # --- classe (profilo comportamentale) ---
    # Le categorie valide vengono apprese dal training, non forzate a quelle
    # di livello_attivita (bug corretto).
    classe = normalizza("classe")
    if fitting:
        fit_params["categorie_classe"] = set(classe.dropna().unique())
        fit_params["moda_classe"] = _moda_sicura(classe, "sconosciuto")
    classe = classe.where(classe.isin(fit_params["categorie_classe"]))
    df["classe"] = classe.fillna(fit_params["moda_classe"])

    # --- eta_anni, peso_kg ---
    df["eta_anni"] = pd.to_numeric(df["eta_anni"], errors="coerce")
    df["peso_kg"] = pd.to_numeric(df["peso_kg"], errors="coerce")
    if fitting:
        fit_params["eta_low"] = max(0, df["eta_anni"].quantile(0.01))
        fit_params["eta_high"] = min(30, df["eta_anni"].quantile(0.99))
        fit_params["peso_low"] = max(0.3, df["peso_kg"].quantile(0.01))
        fit_params["peso_high"] = min(15, df["peso_kg"].quantile(0.99))
    df["eta_anni"] = df["eta_anni"].clip(fit_params["eta_low"], fit_params["eta_high"])
    df["peso_kg"] = df["peso_kg"].clip(fit_params["peso_low"], fit_params["peso_high"])
    if fitting:
        fit_params["mediana_eta"] = df["eta_anni"].median()
        fit_params["mediana_peso"] = df["peso_kg"].median()
    df["eta_anni"] = df["eta_anni"].fillna(fit_params["mediana_eta"])
    df["peso_kg"] = df["peso_kg"].fillna(fit_params["mediana_peso"])

    # Rimuove righe senza target (solo training)
    if "razza" in df.columns:
        df = df.dropna(subset=["razza"]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df, fit_params


def build_preprocessor(num_cols, bin_cols, cat_cols):
    """Crea il ColumnTransformer (StandardScaler + OneHot + passthrough)."""
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("bin", "passthrough", bin_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("1. Caricamento e pulizia...")
    df_train_raw = pd.read_csv("cats_dataset.csv")
    df_test_raw  = pd.read_csv("test_set.csv")
    print(f"   Training set grezzo: {df_train_raw.shape[0]} righe, {df_train_raw.shape[1]} colonne")
    print(f"   Test set grezzo:      {df_test_raw.shape[0]} righe, {df_test_raw.shape[1]} colonne")

    missing = df_train_raw.isna().sum()
    print("   Valori mancanti nel training set grezzo (per colonna):")
    print(missing[missing > 0].to_string() if missing.sum() > 0 else "      nessuno")

    df_train, fit_params = clean_data(df_train_raw)
    df_test, _ = clean_data(df_test_raw, fit_params=fit_params)

    # -----------------------------------------------------------------
    # 2. Correzione etichette "Alien" con un modello supervisionato
    # -----------------------------------------------------------------
    print("\n2. Correzione delle etichette 'Alien'...")
    features = [
        "eta_anni", "peso_kg", "sesso", "lunghezza_pelo", "colore_mantello",
        "livello_attivita", "frequenza_miagolio", "sterilizzato", "patologia", "classe"
    ]
    target = "razza"
    num_cols = ["eta_anni", "peso_kg"]
    bin_cols = ["sterilizzato"]
    cat_cols = ["sesso", "lunghezza_pelo", "colore_mantello", "livello_attivita",
                "frequenza_miagolio", "patologia", "classe"]

    n_alien = (df_train[target] == "Alien").sum()
    if n_alien > 0:
        known_mask = df_train[target] != "Alien"
        imputer = Pipeline([
            ("prep", build_preprocessor(num_cols, bin_cols, cat_cols)),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=10,
                                           random_state=RANDOM_STATE)),
        ])
        imputer.fit(df_train.loc[known_mask, features],
                    df_train.loc[known_mask, target])
        razza_stimata = imputer.predict(df_train.loc[~known_mask, features])
        df_train.loc[~known_mask, target] = razza_stimata
        print(f"   {n_alien} etichette 'Alien' sostituite con la predizione "
              f"di un Random Forest addestrato sulle altre {known_mask.sum()} righe.")
    else:
        print("   Nessuna etichetta 'Alien' trovata.")

    print("\n   Distribuzione delle razze dopo la correzione:")
    print(df_train[target].value_counts().to_string())

    # -----------------------------------------------------------------
    # 3. Analisi del rumore intrinseco (solo diagnostica)
    # -----------------------------------------------------------------
    print("\n3. Analisi del rumore intrinseco...")
    gruppi = df_train.groupby(features)[target]
    n_unique = gruppi.nunique()
    size_gruppi = gruppi.size()
    quota_magg = gruppi.apply(lambda s: s.value_counts(normalize=True).iloc[0])
    n_conflitti = (n_unique > 1).sum()
    righe_conflitto = size_gruppi[n_unique > 1].sum()
    tetto_accuracy = (quota_magg * size_gruppi).sum() / size_gruppi.sum()
    print(f"   {n_conflitti} combinazioni di feature associate a più razze "
          f"({righe_conflitto} righe).")
    print(f"   Tetto teorico di accuratezza: {100*tetto_accuracy:.1f}%")

    # -----------------------------------------------------------------
    # 4. Grafici EDA
    # -----------------------------------------------------------------
    print("\n4. Grafici esplorativi...")
    # Distribuzione razze
    plt.figure(figsize=(10, 5))
    ordine = df_train["razza"].value_counts().index
    sns.countplot(data=df_train, x="razza", order=ordine, hue="razza",
                  palette="viridis", legend=False)
    plt.title("Distribuzione delle razze nel training set")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("grafici/distribuzione_razze.png", dpi=150)
    plt.close()

    # Heatmap correlazione
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_train[["eta_anni", "peso_kg", "sterilizzato"]].corr(),
                annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlazione feature numeriche")
    plt.tight_layout()
    plt.savefig("grafici/heatmap_correlazione.png", dpi=150)
    plt.close()

    # Peso per razza
    plt.figure(figsize=(10, 5))
    ordine_peso = df_train.groupby("razza")["peso_kg"].median().sort_values().index
    sns.boxplot(data=df_train, x="razza", y="peso_kg", order=ordine_peso,
                hue="razza", palette="viridis", legend=False)
    plt.title("Peso corporeo per razza")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("grafici/peso_per_razza.png", dpi=150)
    plt.close()

    # Colore mantello per razza
    plt.figure(figsize=(11, 6))
    crosstab = pd.crosstab(df_train["colore_mantello"], df_train["razza"])
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="mako")
    plt.title("Colore del mantello per razza")
    plt.tight_layout()
    plt.savefig("grafici/mantello_per_razza.png", dpi=150)
    plt.close()

    print("   Salvati: distribuzione_razze.png, heatmap_correlazione.png, "
          "peso_per_razza.png, mantello_per_razza.png")

    # -----------------------------------------------------------------
    # 5. Preprocessing e selezione del modello
    # -----------------------------------------------------------------
    print("\n5. Preparazione dati e selezione del modello...")
    X = df_train[features]
    y = df_train[target]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE
    )

    # Fold adattivi (se una classe ha meno di 5 esempi)
    min_classe = pd.Series(y_tr).value_counts().min()
    n_fold = max(2, min(5, int(min_classe)))
    if n_fold < 5:
        print(f"   NB: uso {n_fold}-fold CV (classe minima ha {min_classe} esempi).")
    cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=RANDOM_STATE)

    candidati = {
        "DecisionTree": (
            Pipeline([("prep", build_preprocessor(num_cols, bin_cols, cat_cols)),
                      ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            {"clf__max_depth": [3, 5, 8, 12, None],
             "clf__min_samples_leaf": [1, 2, 5, 10],
             "clf__criterion": ["gini", "entropy"]}
        ),
        "RandomForest": (
            Pipeline([("prep", build_preprocessor(num_cols, bin_cols, cat_cols)),
                      ("clf", RandomForestClassifier(random_state=RANDOM_STATE))]),
            {"clf__n_estimators": [200, 400],
             "clf__max_depth": [5, 8, 12, None],
             "clf__min_samples_leaf": [1, 2, 5]}
        ),
    }

    risultati_cv = {}
    migliori_pipe = {}
    for nome, (pipe, griglia) in candidati.items():
        print(f"   Ottimizzazione {nome}...")
        gs = GridSearchCV(pipe, griglia, cv=cv, scoring="accuracy", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        risultati_cv[nome] = gs.best_score_
        migliori_pipe[nome] = gs.best_estimator_
        print(f"     Migliori iperparametri: {gs.best_params_}")
        print(f"     Accuratezza CV media: {gs.best_score_:.4f}")

    nome_migliore = max(risultati_cv, key=risultati_cv.get)
    model = migliori_pipe[nome_migliore]
    print(f"\n   Modello selezionato: {nome_migliore} "
          f"(CV accuracy = {risultati_cv[nome_migliore]:.4f})")

    # -----------------------------------------------------------------
    # 6. Valutazione sul validation set
    # -----------------------------------------------------------------
    y_pred = model.predict(X_val)
    print("\n6. Classification Report (Validation Set):")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Matrice di Confusione (Validation Set)")
    plt.ylabel("Reale"); plt.xlabel("Predetto")
    plt.tight_layout()
    plt.savefig("grafici/matrice_confusione.png", dpi=150)
    plt.close()
    print("   Salvato: matrice_confusione.png")

    # -----------------------------------------------------------------
    # 7. Cross‑validazione sull'intero training set
    # -----------------------------------------------------------------
    scores = cross_val_score(model, X, y_enc, cv=cv, scoring="accuracy")
    print(f"\n7. {n_fold}-fold CV accuracy sull'intero training set: "
          f"{scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"   Tetto teorico: {tetto_accuracy:.4f}")
    print(f"   Il modello raggiunge il {100 * scores.mean() / tetto_accuracy:.1f}% "
          f"del massimo teoricamente raggiungibile con queste feature.")

    # -----------------------------------------------------------------
    # 8. Predizioni sul test set
    # -----------------------------------------------------------------
    print("\n8. Addestramento finale e predizioni sul test set...")
    final_model = clone(model)
    final_model.fit(X, y_enc)
    test_pred = final_model.predict(df_test[features])
    test_labels = le.inverse_transform(test_pred)
    pd.DataFrame({
        "ID": df_test["ID"],
        "razza_prevista": test_labels
    }).to_csv("predictions.csv", index=False)
    print("   predictions.csv salvato.")
    print("\nFatto.")


if __name__ == "__main__":
    main()