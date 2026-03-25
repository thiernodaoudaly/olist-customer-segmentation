## Notes

- Les notebooks sont réservés à l'exploration, le code src/ est importé, testé et déployé.

- DVC versionne les gros fichiers de données (CSV, modèles) que Git ne peut pas gérer. Git est fait pour versionner du texte (code). Si tu commites un CSV de 500 Mo, ton repo explose. DVC résout ça avec une mécanique simple : il stocke le fichier lourd ailleurs (ton disque local, S3, Google Drive…) et ne garde dans Git qu'un petit fichier pointeur .dvc de quelques octets.

- Un __init__.py (même vide) dit à Python : "ce dossier est un package, pas juste un dossier". C'est ce qui permettra à tes notebooks et à ton API FastAPI d'importer le code de src/ comme un vrai package Python installé.

- Le fichier pyproject.toml centralise la configuration des outils de qualité de code : formatage, linting, tests. C'est le standard moderne Python (PEP 518).

- mlflow — c'est ton journal de bord automatique. Chaque fois que tu entraînes un modèle, MLflow enregistre les paramètres (n_clusters=5), les métriques (silhouette_score=0.42) et l'artefact (le modèle .pkl). Tu pourras comparer tous tes runs dans une interface web.

- umap-learn — UMAP est un algorithme de réduction de dimension comme PCA, mais bien supérieur pour visualiser des clusters en 2D. Il préserve mieux la structure locale des données. Tu l'utiliseras pour produire des graphiques comme celui-ci :

- pydantic — valide automatiquement les données qui entrent dans ton API. Si l'API reçoit un client avec un champ manquant ou un mauvais type, Pydantic lève une erreur claire avant même que ton modèle soit appelé

