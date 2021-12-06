# Utilisation de modèles classiques (XGBoost) et étude de l'interprétabilité

- WA_Fn-UseC\_-Telco-Customer-Churn.csv

Dataset d'origine sur le churn.

- df_churn_all_tenures.csv

On a conservé seulement les personnes ayant churné, et créé une ligne par mois par client avec le temps depuis lequel il est abonné.

- Churn_Classif_Regression

Notebook exposant les résultats obtenus.
Récap:

|      | df d'origine | df modifié |
| ---- | ------------ | ---------- |
| MAE  | 10.4         | **8.5**    |
| RMSE | 13.4         | **11**     |
