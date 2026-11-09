"""
Application Streamlit : prédictions avec les 4 modèles des notebooks.

    Téléphone      -> état du téléphone (classification, 4 classes)
    Voiture        -> « Venant » ou « D'occasion » (classification, 2 classes)
    Prix voiture   -> prix de vente d'une voiture (régression)
    Taxi           -> prix d'une course de taxi (régression)

Pourquoi cette application ne peut pas envoyer les valeurs brutes au modèle :
les modèles ont été entraînés sur des données ENCODÉES (LabelEncoder) puis MISES À
L'ÉCHELLE (MinMaxScaler ou RobustScaler). Pour prédire, il faut donc rejouer, dans
le même ordre, exactement les mêmes étapes :

    saisie brute -> encodeurs -> variables dans l'ordre de « features » -> normaliseur -> modèle

Fichiers lus dans le dossier models/ (créés par les notebooks) :
    <préfixe>_model.joblib       le modèle (obligatoire)
    <préfixe>_features.joblib    ordre des variables
    <préfixe>_encoders_v2.joblib encodeurs {colonne: LabelEncoder}
    <préfixe>_scaler.joblib      normaliseur ajusté à l'entraînement
avec <préfixe> = telephone, voiture, prix_voiture, taxi.

Si l'un de ces trois derniers fichiers manque (c'est le cas des encodeurs du notebook
téléphone, qui ne les enregistre pas), l'application le reconstruit à partir du CSV
d'origine en rejouant le prétraitement des notebooks, puis l'enregistre dans models/.
Les CSV sont cherchés dans data/, puis à côté de ce fichier.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

DOSSIER_APP = Path(__file__).resolve().parent
DOSSIER_MODELES = DOSSIER_APP / "models"

st.set_page_config(page_title="Modèles de prédiction", page_icon="🤖", layout="wide")


# ----------------------------------------------------------------------------
# Configuration des 4 modèles
# ----------------------------------------------------------------------------
# Chaque champ « num » donne : bornes de saisie, valeur par défaut, pas, et la plage
# réellement vue à l'entraînement (« entraine ») après le traitement des valeurs
# aberrantes des notebooks. Hors de cette plage, le modèle extrapole : on prévient.
MODELES = {
    "Téléphone": {
        "prefixe": "telephone",
        "type": "classification",
        "cible": "etat",
        "titre_cible": "État du téléphone",
        "description": "Classe un téléphone en : D'occasion, Neuf, Réconditionné ou Venant.",
        "performance": "Forêt aléatoire : environ 90 % de bonnes réponses sur le jeu de test.",
        "csv": "telephone_data.csv",
        "champs": [
            {"nom": "prix", "libelle": "Prix (FCFA)", "type": "num", "min": 0, "max": 5_000_000,
             "defaut": 360_000, "pas": 5_000, "entraine": (1, 795_000)},
            {"nom": "adresse", "libelle": "Adresse (quartier)", "type": "cat", "defaut": "Plateau"},
            {"nom": "marque", "libelle": "Marque", "type": "cat", "defaut": "iPhone"},
            {"nom": "dim_ecr", "libelle": "Taille de l'écran (pouces)", "type": "num", "min": 3.0,
             "max": 12.0, "defaut": 6.0, "pas": 0.1, "entraine": (5, 8)},
            {"nom": "ram", "libelle": "RAM (Go)", "type": "num", "min": 1, "max": 64,
             "defaut": 8, "pas": 1, "entraine": (7, 16)},
            {"nom": "stockage", "libelle": "Stockage (Go)", "type": "num", "min": 0, "max": 2048,
             "defaut": 128, "pas": 16, "entraine": (1, 348)},
        ],
    },
    "Voiture (Venant / D'occasion)": {
        "prefixe": "voiture",
        "type": "classification",
        "cible": "Venant_Occasion",
        "titre_cible": "Catégorie de la voiture",
        "description": "Classe une voiture en : Venant ou D'occasion.",
        "performance": "Arbre de décision : 78,4 % de bonnes réponses sur le jeu de test.",
        "csv": "Car_Data.csv",
        "champs": [
            {"nom": "Marque", "libelle": "Marque", "type": "cat", "defaut": "Ford"},
            {"nom": "Année", "libelle": "Année", "type": "num", "min": 1980, "max": 2026,
             "defaut": 2015, "pas": 1, "entraine": (2009, 2020)},
            {"nom": "Boite_Vitesse", "libelle": "Boîte de vitesses", "type": "cat", "defaut": "Automatique"},
            {"nom": "Adresse", "libelle": "Adresse", "type": "cat", "defaut": "Almadies,Dakar"},
            {"nom": "Prix", "libelle": "Prix (FCFA)", "type": "num", "min": 0, "max": 800_000_000,
             "defaut": 7_800_000, "pas": 100_000, "entraine": (1, 24_000_000)},
        ],
    },
    "Prix d'une voiture": {
        "prefixe": "prix_voiture",
        "type": "regression",
        "cible": "Selling_Price",
        "titre_cible": "Prix de vente estimé",
        "description": "Estime le prix de vente d'une voiture. Le résultat est dans la même unité "
                       "que la colonne Selling_Price du fichier Car_dataset.csv.",
        "performance": "Forêt aléatoire : R² = 0,905 ; erreur absolue moyenne ≈ 0,84 sur le jeu de test.",
        "csv": "Car_dataset.csv",
        "champs": [
            {"nom": "Kms_Driven", "libelle": "Kilomètres parcourus", "type": "num", "min": 0,
             "max": 600_000, "defaut": 70_000, "pas": 1_000, "entraine": (3_000, 135_154)},
            {"nom": "Present_Price", "libelle": "Prix actuel (neuf), unité du jeu de données", "type": "num",
             "min": 0.1, "max": 110.0, "defaut": 15.0, "pas": 0.5, "entraine": (0.32, 32.2)},
            {"nom": "Fuel_Type", "libelle": "Carburant", "type": "cat", "defaut": "Petrol"},
            {"nom": "Seller_Type", "libelle": "Type de vendeur", "type": "cat", "defaut": "Dealer"},
            {"nom": "Transmission", "libelle": "Transmission", "type": "cat", "defaut": "Automatic"},
            {"nom": "Age", "libelle": "Âge de la voiture (ans)", "type": "num", "min": 0, "max": 25,
             "defaut": 8, "pas": 1, "entraine": (2, 16)},
        ],
    },
    "Taxi": {
        "prefixe": "taxi",
        "type": "regression",
        "cible": "Trip_Price",
        "titre_cible": "Prix de la course estimé",
        "description": "Estime le prix d'une course de taxi. Le résultat est dans la même unité "
                       "que la colonne Trip_Price du fichier taxi_trip_pricing.csv.",
        "performance": "Forêt aléatoire : R² = 0,889 ; erreur absolue moyenne ≈ 5,4 sur le jeu de test.",
        "csv": "taxi_trip_pricing.csv",
        "champs": [
            {"nom": "Trip_Distance_km", "libelle": "Distance (km)", "type": "num", "min": 0.5,
             "max": 150.0, "defaut": 25.0, "pas": 0.5, "entraine": (1.23, 49.99)},
            {"nom": "Time_of_Day", "libelle": "Moment de la journée", "type": "cat", "defaut": "Afternoon"},
            {"nom": "Day_of_Week", "libelle": "Type de jour", "type": "cat", "defaut": "Weekday"},
            {"nom": "Passenger_Count", "libelle": "Nombre de passagers", "type": "num", "min": 1,
             "max": 6, "defaut": 2, "pas": 1, "entraine": (1, 4)},
            {"nom": "Traffic_Conditions", "libelle": "Circulation", "type": "cat", "defaut": "Low"},
            {"nom": "Weather", "libelle": "Météo", "type": "cat", "defaut": "Clear"},
            {"nom": "Base_Fare", "libelle": "Prise en charge (Base_Fare)", "type": "num", "min": 1.0,
             "max": 6.0, "defaut": 3.5, "pas": 0.1, "entraine": (2.01, 5.0)},
            {"nom": "Per_Km_Rate", "libelle": "Tarif par km (Per_Km_Rate)", "type": "num", "min": 0.1,
             "max": 3.0, "defaut": 1.2, "pas": 0.05, "entraine": (0.5, 2.0)},
            {"nom": "Per_Minute_Rate", "libelle": "Tarif par minute (Per_Minute_Rate)", "type": "num",
             "min": 0.05, "max": 1.0, "defaut": 0.3, "pas": 0.01, "entraine": (0.1, 0.5)},
            {"nom": "Trip_Duration_Minutes", "libelle": "Durée (minutes)", "type": "num", "min": 1.0,
             "max": 180.0, "defaut": 60.0, "pas": 1.0, "entraine": (5.01, 119.84)},
        ],
    },
}

# Libellés français pour les petites listes de valeurs (la valeur brute reste celle du jeu de données)
LIBELLES_VALEURS = {
    "Time_of_Day": {"Morning": "Matin", "Afternoon": "Après-midi", "Evening": "Soir", "Night": "Nuit"},
    "Day_of_Week": {"Weekday": "Jour de semaine", "Weekend": "Week-end"},
    "Traffic_Conditions": {"Low": "Faible", "Medium": "Modérée", "High": "Forte"},
    "Weather": {"Clear": "Dégagé", "Rain": "Pluie", "Snow": "Neige"},
    "Fuel_Type": {"Petrol": "Essence", "Diesel": "Diesel", "CNG": "GNC (CNG)"},
    "Seller_Type": {"Dealer": "Concessionnaire", "Individual": "Particulier"},
    "Transmission": {"Manual": "Manuelle", "Automatic": "Automatique"},
}


# ----------------------------------------------------------------------------
# Reconstruction des artefacts manquants (rejeu du prétraitement des notebooks)
# ----------------------------------------------------------------------------
# aberrantes_cible : le notebook traite aussi la colonne cible (régressions)
# tout_le_jeu      : le KNNImputer est d'abord passé sur tout le tableau (régressions)
# normaliser_tout  : le normaliseur est ajusté sur tout le jeu avant le découpage (taxi)
REJEU = {
    "telephone":    {"normaliseur": "minmax", "decoupage": {"train_size": 0.8}, "aberrantes_cible": False,
                     "mode_categories": False, "tout_le_jeu": False, "normaliser_tout": False},
    "voiture":      {"normaliseur": "minmax", "decoupage": {"train_size": 0.8}, "aberrantes_cible": False,
                     "mode_categories": False, "tout_le_jeu": False, "normaliser_tout": False},
    "prix_voiture": {"normaliseur": "robust", "decoupage": {"test_size": 0.2}, "aberrantes_cible": True,
                     "mode_categories": True, "tout_le_jeu": True, "normaliser_tout": False},
    "taxi":         {"normaliseur": "robust", "decoupage": {"test_size": 0.2}, "aberrantes_cible": True,
                     "mode_categories": True, "tout_le_jeu": True, "normaliser_tout": True},
}

FICHIERS = {  # artefact -> noms de fichiers acceptés (le premier est utilisé à l'écriture)
    "features": ["{p}_features.joblib"],
    "encodeurs": ["{p}_encoders_v2.joblib", "{p}_encoders.joblib"],
    "normaliseur": ["{p}_scaler.joblib"],
}


def _traiter_valeurs_aberrantes(jeu, colonne, n_neighbors=5):
    """Même traitement que dans les notebooks : méthode IQR puis remplacement par KNNImputer."""
    q1, q3 = jeu[colonne].quantile(0.25), jeu[colonne].quantile(0.75)
    iqr = q3 - q1
    jeu.loc[(jeu[colonne] < q1 - 1.5 * iqr) | (jeu[colonne] > q3 + 1.5 * iqr), colonne] = np.nan
    jeu[:] = KNNImputer(n_neighbors=n_neighbors).fit_transform(jeu)


def _trouver_csv(nom):
    for dossier in (DOSSIER_APP / "data", DOSSIER_APP, Path.cwd() / "data", Path.cwd()):
        if (dossier / nom).exists():
            return dossier / nom
    raise FileNotFoundError(
        f"Fichier de données « {nom} » introuvable. Place-le dans le dossier data/ à côté de "
        f"{Path(__file__).name} pour que l'application puisse reconstruire les encodeurs ou le normaliseur."
    )


def _reconstruire(prefixe, cfg_modele):
    """Rejoue le prétraitement du notebook et renvoie (features, encodeurs, normaliseur)."""
    r = REJEU[prefixe]
    jeu = pd.read_csv(_trouver_csv(cfg_modele["csv"]))
    if prefixe == "voiture":  # nettoyage de la colonne Adresse fait dans le notebook
        jeu["Adresse"] = jeu["Adresse"].str.replace(r"\s*,\s*", ",", regex=True).str.strip()
        jeu["Adresse"] = jeu["Adresse"].replace({"Lac rose,Dakar": "Lac Rose,Dakar"})
    categorielles = jeu.select_dtypes(exclude="number").columns.tolist()
    numeriques = jeu.select_dtypes(include="number").columns.tolist()
    if r["mode_categories"]:
        for col in categorielles:
            jeu.fillna({col: jeu[col].mode()[0]}, inplace=True)
    encodeurs = {}
    for col in categorielles:
        encodeurs[col] = LabelEncoder()
        jeu[col] = encodeurs[col].fit_transform(jeu[col])
    if r["tout_le_jeu"]:
        jeu[:] = KNNImputer(n_neighbors=5).fit_transform(jeu)
    for col in numeriques:
        if col == cfg_modele["cible"] and not r["aberrantes_cible"]:
            continue
        _traiter_valeurs_aberrantes(jeu, col)
    features = jeu.columns[:-1].tolist()  # la cible est la dernière colonne
    x, y = jeu[features].values, jeu[cfg_modele["cible"]].values
    normaliseur = MinMaxScaler() if r["normaliseur"] == "minmax" else RobustScaler()
    if r["normaliser_tout"]:
        normaliseur.fit(x)
    else:
        x_entrainement, _, _, _ = train_test_split(x, y, random_state=42, **r["decoupage"])
        normaliseur.fit(x_entrainement)
    return features, encodeurs, normaliseur


# ----------------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------------
def _lire(prefixe, artefact):
    for modele_nom in FICHIERS[artefact]:
        chemin = DOSSIER_MODELES / modele_nom.format(p=prefixe)
        if chemin.exists():
            return joblib.load(chemin)
    return None


@st.cache_resource(show_spinner="Chargement du modèle…")
def charger(nom_modele):
    """Charge le modèle et ses artefacts ; reconstruit ceux qui manquent."""
    cfg = MODELES[nom_modele]
    p = cfg["prefixe"]
    chemin_modele = DOSSIER_MODELES / f"{p}_model.joblib"
    if not chemin_modele.exists():
        raise FileNotFoundError(f"Modèle introuvable : {chemin_modele}")
    modele = joblib.load(chemin_modele)
    art = {"features": _lire(p, "features"), "encodeurs": _lire(p, "encodeurs"),
           "normaliseur": _lire(p, "normaliseur")}
    manquants = [k for k, v in art.items() if v is None]
    if manquants:
        features, encodeurs, normaliseur = _reconstruire(p, cfg)
        reconstruits = {"features": features, "encodeurs": encodeurs, "normaliseur": normaliseur}
        for k in manquants:
            art[k] = reconstruits[k]
            try:
                DOSSIER_MODELES.mkdir(exist_ok=True)
                joblib.dump(art[k], DOSSIER_MODELES / FICHIERS[k][0].format(p=p))
            except OSError:
                pass  # dossier en lecture seule : on garde l'objet en mémoire
    attendues = [c["nom"] for c in cfg["champs"]]
    if list(art["features"]) != attendues:
        raise ValueError(f"Ordre des variables inattendu : {list(art['features'])} au lieu de {attendues}")
    if getattr(modele, "n_features_in_", len(attendues)) != len(attendues):
        raise ValueError("Le modèle n'attend pas le même nombre de variables que l'application.")
    return {"modele": modele, "reconstruits": manquants, **art}


def predire(nom_modele, art, saisies):
    """saisie brute -> encodeurs -> ordre des features -> normaliseur -> modèle."""
    cfg = MODELES[nom_modele]
    ligne = []
    for nom in art["features"]:
        if nom in art["encodeurs"]:
            ligne.append(float(art["encodeurs"][nom].transform([saisies[nom]])[0]))
        else:
            ligne.append(float(saisies[nom]))
    x = art["normaliseur"].transform(np.array([ligne], dtype=float))
    resultat = {"brut": art["modele"].predict(x)[0]}
    if cfg["type"] == "classification":
        encodeur_cible = art["encodeurs"][cfg["cible"]]
        resultat["libelle"] = encodeur_cible.inverse_transform([int(resultat["brut"])])[0]
        if hasattr(art["modele"], "predict_proba"):
            probas = art["modele"].predict_proba(x)[0]
            classes = encodeur_cible.inverse_transform(art["modele"].classes_.astype(int))
            resultat["probas"] = pd.DataFrame({"Classe": classes, "Probabilité (%)": (probas * 100).round(1)}) \
                .sort_values("Probabilité (%)", ascending=False)
    return resultat


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.title("🤖 Application de prédiction")
st.markdown("Choisis un modèle dans la barre latérale, renseigne les caractéristiques, puis lance la prédiction.")

st.sidebar.title("📑 Sélection du modèle")
nom_modele = st.sidebar.radio("Choisis un modèle :", list(MODELES))
cfg = MODELES[nom_modele]

st.header(f"Modèle : {nom_modele}")
st.info(f"**Description :** {cfg['description']}")
st.caption(f"Performance mesurée dans le notebook : {cfg['performance']}")

try:
    art = charger(nom_modele)
except Exception as e:  # fichier absent, version de scikit-learn différente, etc.
    st.error(f"Impossible de charger ce modèle : {e}")
    st.stop()

if art["reconstruits"]:
    noms = {"features": "l'ordre des variables", "encodeurs": "les encodeurs", "normaliseur": "le normaliseur"}
    st.sidebar.info("Reconstruit depuis le CSV (fichier absent de models/) : "
                    + ", ".join(noms[k] for k in art["reconstruits"]) + ".")

st.subheader("🔢 Caractéristiques")
saisies = {}
with st.form(f"formulaire_{cfg['prefixe']}"):
    colonnes = st.columns(3)
    for i, champ in enumerate(cfg["champs"]):
        nom = champ["nom"]
        cle = f"{cfg['prefixe']}__{nom}"
        with colonnes[i % 3]:
            if champ["type"] == "cat":
                options = art["encodeurs"][nom].classes_.tolist()
                defaut = options.index(champ["defaut"]) if champ["defaut"] in options else 0
                traduction = LIBELLES_VALEURS.get(nom, {})
                saisies[nom] = st.selectbox(champ["libelle"], options, index=defaut, key=cle,
                                            format_func=lambda v, t=traduction: t.get(v, v))
            else:
                entier = all(isinstance(champ[k], int) for k in ("min", "max", "defaut", "pas"))
                saisies[nom] = st.number_input(
                    champ["libelle"], min_value=champ["min"], max_value=champ["max"], value=champ["defaut"],
                    step=champ["pas"], key=cle, format="%d" if entier else "%.2f")
    lancer = st.form_submit_button("🔍 Faire une prédiction", type="primary")

if lancer:
    hors_plage = []
    for champ in cfg["champs"]:
        if champ["type"] == "num":
            bas, haut = champ["entraine"]
            valeur = saisies[champ["nom"]]
            if valeur < bas or valeur > haut:
                hors_plage.append(f"{champ['libelle']} = {valeur:g} (valeurs vues à l'entraînement : {bas:g} à {haut:g})")
    try:
        res = predire(nom_modele, art, saisies)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        st.stop()

    st.success("✅ Prédiction effectuée")
    gauche, droite = st.columns([2, 1])
    with gauche:
        st.markdown("### 📊 Résultat")
        if cfg["type"] == "classification":
            st.metric(cfg["titre_cible"], res["libelle"])
            if "probas" in res:
                st.dataframe(res["probas"], hide_index=True)
        else:
            st.metric(cfg["titre_cible"], f"{res['brut']:.2f}")
    with droite:
        st.markdown("### 📝 Résumé des entrées")
        resume = pd.DataFrame({"Variable": [c["libelle"] for c in cfg["champs"]],
                               "Valeur": [str(saisies[c["nom"]]) for c in cfg["champs"]]})
        st.dataframe(resume, hide_index=True)
    if hors_plage:
        st.warning("Certaines valeurs sortent de la plage vue à l'entraînement : le résultat est une "
                   "extrapolation peu fiable.\n\n" + "\n".join(f"- {t}" for t in hors_plage))

st.markdown("---")
st.caption("Application basée sur les notebooks du projet Mbaye. Les modèles ne remplacent pas un avis d'expert.")
