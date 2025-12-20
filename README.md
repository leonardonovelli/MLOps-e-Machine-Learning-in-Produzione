# MLOps-e-Machine-Learning-in-Produzione

## Descrizione

Questo progetto implementa un modello di analisi del sentiment utilizzando Hugging Face `cardiffnlp/twitter-roberta-base-sentiment-latest`. 
Il progetto è pensato per simulare un contesto di produzione reale, dove il training deve essere rapido, automatizzato e condizionato al miglioramento delle metriche, mentre l’osservabilità del modello avviene a runtime tramite metriche persistenti.

Implementazione progetto:
- **Pipeline CI/CD** per training, test, e push **condizionale** su Hugging Face Hub del modello tramite GitHub actions
- **Inference API** in FastAPI
- **Metriche salvate in tempo reale su SQL Server**
- **Visualizzazione metriche con Grafana**
- **Deploy tramite Docker Compose**
- **Integrazione con Hugging Face Hub**

Ogni chiamata di inferenza aggiorna automaticamente le metriche nel database, rendendole immediatamente disponibili in Grafana come **time series reali**.

---

## Steps x GitHub actions

Prima di procedere con il train del modello vengono effettuati jobs di **Lint with Pylint** (con una threshold di 9.0) ed esecuzione di test di unità con *pytest*

1. **Training rapido**  
   - `train.py` esegue il training su un **subset casuale** del dataset (`cardiffnlp/tweet_eval, preso anch'esso da HF`): 1000 esempi per il training e 200 per la validazione.  
   - Il campionamento è **random e riproducibile** tramite seed configurabile.
   - Questa decisione è stata presa per evitare di aumentare di troppo il tempo di addestramento del modello.

2. **Calcolo metriche**  
   - Viene calcolata l'`accuracy` sul validation set.

3. **Salvataggio metriche**  
   - Le metriche correnti vengono salvate in `metrics.json`.

4. **Controllo migliori metriche**  
   - Viene confrontata l'`accuracy` corrente con quella salvata in `best_metrics.json`.
   - Se l'accuracy è migliore, il file `best_metrics.json` viene aggiornato con il nuovo valore e il timestamp.

5. **Push su Hugging Face Hub (opzionale)**  
   - Se le metriche migliorano, il modello e il tokenizer possono essere pushati direttamente su Hugging Face Hub tramite la funzione `push_to_hf()` in `src/utils.py`.

Note: Su Github sono state inoltre aggiunte delle secrects per l'eventuale pubblicazione del modello su HF
```
HF_TOKEN: ${{ secrets.HF_TOKEN }}
HF_REPO_ID: ${{ secrets.HF_REPO_ID }}
```

### Parametri configurabili

- `TRAIN_SIZE` e `EVAL_SIZE` in `train.py` per definire il numero di esempi.
- `SEED` per rendere il campionamento casuale riproducibile.
- `HF_TOKEN` variabile d’ambiente per l’accesso a modelli privati su Hugging Face.
- `HF_REPO_ID` variabile d’ambiente che fa riferimento al modello su Hugging Face.

### Esecuzione locale

```bash
# training rapido
python src/train.py
```


---

## Architettura

1. | Client (Postman / HTTP) |

2. | FastAPI Inference | (Hugging Face) |

3. | SQL Server | metrics table |

4. | Grafana | Time Series |

## Componenti

### 1. FastAPI Inference Service
- Espone l’endpoint ` POST /predict`
  ```
  { "text": "Beautiful post!!!" }
  ```
- Usa il modello Hugging Face (`pipeline`) e risponde con
   ```
  { "label": "positive", "score": 0.9987 , "latency_ms": 163.98 }
  ```
- Registra **ad ogni predict**:
  - timestamp
  - text
  - latency
  - label
  - confidence
- Scrive le metriche direttamente su **SQL Server**

### 2. SQL Server
- Database persistente
- Tabelle create automaticamente all’avvio
- Metriche disponibili via SQL
- Fonte dati per Grafana

### 3. Grafana
- Collegato a SQL Server
- Visualizza metriche come:
  - latenza media
  - richieste nel tempo
  - confidence media
- Dashboard in tempo reale

---

## Variabili d’ambiente

Creare un file `.env` nella root del progetto:

```env
HF_REPO_ID=username/model-name
HF_TOKEN=xxxxxxxxxxxxxxxxx
```

Alcune variabili sono utilizzate solo dal docker-compose per il mapping delle porte e la configurazione dei servizi.
```

API_PORT_MAPPING=XXXX:XXXX
GRAFANA_PORT_MAPPING=XXXX:XXXX

SQL_SERVER=sqlserver
SQL_PORT=1433
SQL_USER=xxxxxxxxxxxxxxxxx
SQL_PASSWORD=xxxxxxxxxxxxxxxxx
SQL_DB=xxxxxxxxxxxxxxxxx
SQL_PORT_MAPPING=XXXX:XXXX

SA_PASSWORD="xxxxxxxxxxxxxxxxx"
ACCEPT_EULA=Y
```

---

## Comandi docker

### Costruzione build
```
docker compose up -d --build
```
