## Notes

- Les notebooks sont réservés à l'exploration, le code src/ est importé, testé et déployé.

- DVC versionne les gros fichiers de données (CSV, modèles) que Git ne peut pas gérer. Git est fait pour versionner du texte (code). Si tu commites un CSV de 500 Mo, ton repo explose. DVC résout ça avec une mécanique simple : il stocke le fichier lourd ailleurs (ton disque local, S3, Google Drive…) et ne garde dans Git qu'un petit fichier pointeur .dvc de quelques octets.

- Un __init__.py (même vide) dit à Python : "ce dossier est un package, pas juste un dossier". C'est ce qui permettra à tes notebooks et à ton API FastAPI d'importer le code de src/ comme un vrai package Python installé.

- Le fichier pyproject.toml centralise la configuration des outils de qualité de code : formatage, linting, tests. C'est le standard moderne Python (PEP 518).

- mlflow — c'est ton journal de bord automatique. Chaque fois que tu entraînes un modèle, MLflow enregistre les paramètres (n_clusters=5), les métriques (silhouette_score=0.42) et l'artefact (le modèle .pkl). Tu pourras comparer tous tes runs dans une interface web.

- umap-learn — UMAP est un algorithme de réduction de dimension comme PCA, mais bien supérieur pour visualiser des clusters en 2D. Il préserve mieux la structure locale des données. Tu l'utiliseras pour produire des graphiques comme celui-ci :

- pydantic — valide automatiquement les données qui entrent dans ton API. Si l'API reçoit un client avec un champ manquant ou un mauvais type, Pydantic lève une erreur claire avant même que ton modèle soit appelé

- Pourquoi compute_recency() prend customer_unique_id et pas customer_id ? Si on groupait par customer_id, on aurait 3 clients distincts au lieu d'un seul. Le RFM serait faux — recency, frequency et monetary calculés sur des fragments de client.

- Pourquoi impute-t-on review_score_mean avec la médiane et pas la moyenne ? Notre distribution des scores est bimodale (pics en 1 et en 5). La moyenne (4.09) ne représente quasiment aucun client réel — personne ne donne 4.09 étoiles. La médiane (5.0) correspond à un vrai comportement majoritaire. Pour les valeurs manquantes, imputer avec la médiane introduit moins de biais dans un clustering.

- Pourquoi apply_log=True crée frequency_log et monetary_log comme nouvelles colonnes au lieu de remplacer frequency et monetary ? 
    Raison 1 — Traçabilité : dans MLflow on loggue les stats des features brutes ET transformées — si on écrase, on perd l'information originale.
    Raison 2 — Interprétabilité : après le clustering, pour décrire les segments au jury et à l'équipe marketing on utilisera les valeurs brutes ("Ce segment dépense en moyenne 250 BRL"), pas les valeurs log ("Ce segment a un monetary_log de 5.52").
    Raison 3 — Flexibilité : certains algorithmes comme DBSCAN sont moins sensibles à l'échelle — on pourra tester avec ou sans log-transformation sans recalculer tout le pipeline.

- 