Pour classer des verbatims, il faut exécuter le script python "livrable.py"

Pour cela un certain nombre de librairies sont prérequises :


-nltk
-keras
-numpy
-string
-sys
-time
-pickle
-xlrd
-spacy


Une fois ces librairies installées, il faut télecharger un modèle spacy en tapant ceci sur le terminal :


python -m spacy download fr_core_news_md





On peut maintenant exécuter le script en entrant dans le terminal :

python3 livrable.py "nom_du_excel.xlsx"



Assurez vous que votre excel est disposé de la même manière que le fichier "exemple.xlsx"
La page du excel que vous voulez traiter doit ABSOLUMENT être renommé "Verbatims"

Une fois le code executé, vous devriez obtenir une fichier CSV "resultat.csv"


