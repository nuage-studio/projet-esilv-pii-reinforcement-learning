# Utilisation de modèles classiques (XGBoost) et étude de l'interprétabilité

-   WA_Fn-UseC\_-Telco-Customer-Churn.csv

Dataset d'origine sur le churn.

-   df_churn_all_tenures.csv

On a conservé seulement les personnes ayant churné, et créé une ligne par mois par client avec le temps depuis lequel il est abonné.

-   Churn_Classif_Regression

Notebook exposant les résultats obtenus.
Récapitulatif des meilleurs résultats en Machine Learning (utilisation de l'[Explainable Boosting Regressor](https://interpret.ml/docs/ebm.html)):

|      | df d'origine | df modifié |
| ---- | ------------ | ---------- |
| MAE  | 11.4         | **9.9**    |
| RMSE | 17.5         | **13.5**   |
