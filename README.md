# EmotionGIF2020

1. ./exec_bert.sh [Desire BERT model directory] (bert_model_simple or bert_model_large)

  Open bert server first.

2. python3 final.py 

  This procedure will generate parameter pickle of different embedding, so we do not need to retrain our model every time.
  
3. python3 classify.py

  This procedure will do the classification work and output the final csv for upload. (If you want to change some hyperparameter setting, see the global variable of classify.py)
  
4. result is saved as dev.json
