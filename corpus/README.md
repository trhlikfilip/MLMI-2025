# Scripts for extracting analysing and adjusting the corpora

## Metrics
- _corpus_metrics.py_: Extracts all metrics from the TextDescriptives library
- _toxicity.py_: Labels toxicity of every sentence in a given corpus
- _hatespeech.py_: Labels the probability of every sentence in a given corpus to be hate-speech
- _sentiment.p_y: Labels sentiment of every sentence in a given corpus
- _lexical_emotions.py_: Extracts the avarage presence of the 6 core emotions form the corpus

## Adjustment
- _fair.ipynb_: Code for perturbing the corpus in order to debias it
- - _detox.py_: Code for detoxifying sentences in the corpus using LLM
