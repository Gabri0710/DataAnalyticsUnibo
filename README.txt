Progetto Data Analytics.
Si sono implementati i seguenti modelli:
- Linear Regression ('lr')
- Random Forest ('rf')
- Support Vector Regressor ('svr')
- K-Neighbors Regressor ('knr')
- Rete Neurale FeedForward ('ff')
- Architettura TabNet ('tb')
- Architettura TabTransformer ('tf')

I modelli risultanti sono presenti nella cartella "models". Lo script 'test.py' importerà gli oggetti necessari per le predizioni da quella cartella.

Si consiglia di testare il modello SVR per ultimo a causa della lentezza di predizione dovuta alla complessità del modello (circa 20 minuti per 25mila campioni sulla mia macchina). La scelta di non rendere il modello meno complesso per velocizzare le predizioni è stata fatta per massimizzare i risultati del modello. In scenari reali, si potrebbe preferire una via di mezzo o utilizzare modelli più veloci e performanti.