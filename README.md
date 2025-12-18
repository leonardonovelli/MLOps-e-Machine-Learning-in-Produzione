# MLOps-e-Machine-Learning-in-Produzione

## Descrizione

Questo progetto implementa un modello di analisi del sentiment utilizzando Hugging Face `cardiffnlp/twitter-roberta-base-sentiment-latest`. 
Include una pipeline CI/CD per training rapido, test, e push condizionale su Hugging Face Hub.

## Steps

1. **Training rapido**  
   - `train.py` esegue il training su un **subset casuale** del dataset (`cardiffnlp/tweet_eval, preso anch'esso da HF`): 1000 esempi per il training e 200 per la validazione.  
   - Il campionamento è **random e riproducibile** tramite seed configurabile.

2. **Calcolo metriche**  
   - Viene calcolata l'`accuracy` sul validation set.

3. **Salvataggio metriche**  
   - Le metriche correnti vengono salvate in `metrics.json`.

4. **Controllo migliori metriche**  
   - Viene confrontata l'`accuracy` corrente con quella salvata in `best_metrics.json`.
   - Se l'accuracy è migliore, il file `best_metrics.json` viene aggiornato con il nuovo valore e il timestamp.

5. **Push su Hugging Face Hub (opzionale)**  
   - Se le metriche migliorano, il modello e il tokenizer possono essere pushati direttamente su Hugging Face Hub tramite la funzione `push_to_hf()` in `src/utils.py`.

## Parametri configurabili

- `TRAIN_SIZE` e `EVAL_SIZE` in `train.py` per definire il numero di esempi.
- `SEED` per rendere il campionamento casuale riproducibile.
- `HF_TOKEN` variabile d’ambiente per l’accesso a modelli privati su Hugging Face.

## Esecuzione locale

```bash
# training rapido
python src/train.py