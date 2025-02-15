# wot-SolarProductionForecast-Backend-BelcuoreErroi
## Descrizione del progetto

Il progetto *Solar Production Forecast* nasce dall’esigenza di sviluppare un modello in grado di prevedere l’energia prodotta da un impianto fotovoltaico, offrendo agli utenti un supporto per una gestione più efficiente e consapevole dei consumi energetici. L'obiettivo è ottimizzare i costi legati all’utilizzo della rete elettrica, privilegiando l’uso di una fonte rinnovabile.

Poiché la radiazione solare varia nel tempo in funzione delle condizioni meteorologiche, la produzione di energia fotovoltaica risulta discontinua e soggetta a fluttuazioni non sempre prevedibili. Per affrontare questa variabilità, il progetto impiega un modello ensemble composto da *Support Vector Regression* (SVR), *Random Forest*, *Ridge Regression* e *Kernel Ridge Regression*. A supporto di questi algoritmi, vengono applicate ulteriori tecniche avanzate di machine learning e data mining per migliorare l’accuratezza delle previsioni. I dati utilizzati vengono raccolti da *open-meteo.com* e da un impianto situato nella città di Lecce.

Le previsioni sono presentate in modo chiaro e intuitivo attraverso una dashboard user-friendly, che consente agli utenti di interpretare facilmente i dati senza la necessità di consultare documentazione aggiuntiva.

## Architettura del sistema

L’architettura del sistema è composta dai seguenti elementi:

1. **Raccolta dati**: API a disposizione del backend/frontend per l'acquisizione di dati metereologici da *open-meteo.com* e di dati storici della produzione dell'impianto fotovoltaico;
2. **Backend in Python**: pre-elabora i dati e genera previsioni tramite un modello ensemble basato su *SVR*, *Random Forest*, *Ridge Regression* e *Kernel Ridge Regression*;
3. **Archiviazione in MongoDB**: memorizza le previsioni generate dal modello per analisi e consultazioni future;
4. **Dashboard in React**: offre un’interfaccia intuitiva per la visualizzazione interattiva delle previsioni.

![Architettura](proposta.png)

## Backend

Il backend del sistema è strutturato in diverse sezioni, ciascuna con un ruolo ben definito, che collaborano per ottenere un risultato preciso: fornire previsioni sulla produzione solare per i prossimi sette giorni, con un dettaglio temporale di 15 minuti.
Le principali componenti sviluppate sono le seguenti:

1. **Raccolta Dati**: Il backend interroga diverse API per raccogliere dati essenziali, sia di tipo meteorologico che relativi all’impianto fotovoltaico. Queste informazioni costituiscono la base su cui si fondano le previsioni e devono essere quanto più accurate e aggiornate possibile;
2. **Pulizia Dati**: Una volta acquisiti, i dati vengono sottoposti a un processo di preprocessing per assicurarne la coerenza e la completezza. Questa fase è cruciale per eliminare eventuali anomalie o imperfezioni che potrebbero compromettere l’affidabilità delle previsioni. Le operazioni di pulizia comprendono:
* Selezione delle feature numeriche
* Rimozione dei duplicati
* Gestione dei valori nulli
* Eliminazione degli outlier
* Codifica delle feature categoriche
* Scaling
4. **Modello Machine Learning**: Il cuore del sistema di previsione è un modello di machine learning basato su un approccio ensemble, una tecnica avanzata che combina più modelli per ottenere risultati più affidabili rispetto a un singolo algoritmo. Questo metodo permette di sfruttare i punti di forza di diversi approcci, migliorando così la precisione delle previsioni. Nel nostro caso, il modello ensemble è composto da diversi algoritmi di regressione, tra cui:
* Support Vector Regression (SVR)
* Random Forest Regressor
* Kernel Ridge
* Ridge
6. **Archiviazione**: Infine, il backend è direttamente connesso a un'istanza MongoDB, un database NoSQL che garantisce una gestione efficiente e scalabile dei dati. Qui vengono memorizzate sia le informazioni raccolte che le previsioni generate, permettendo un facile accesso e consultazione per le future elaborazioni.


## Il frontend del sistema è disponibile nel seguente repository: [wot-SolarProductionForecast-Frontend-BelcuoreErroi](https://github.com/GErroi/wot-SolarProductionForecast-FrontEnd-BelcuoreErroi)
## La presentazione del sistema è disponibile nel seguente repository: [wot-SolarProductionForecast-Presentation-BelcuoreErroi](https://github.com/GErroi/wot-SolarProductionForecast-Presentation-BelcuoreErroi)
