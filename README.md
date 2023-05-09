# SHAP-Enhanced Syntatic Analysis for Aspect-based Sentiment Analysis based on Labeled Sentiments Only

## Overview
1. ```EASTER.py``` the codes of EASTER. To extract sentiment and shap value.

2. ```main.py``` the codes of AOSExtractor. To extact AOS base on sentiment and shap value.

3. ```config.py``` the configurations of project.

4. ```data/aos/pretrained``` the pretrained model weights file of EASTER. 

5. ```data/aos/*.pkl``` the sentiment and shap value file.

6. ```data/aos/SentiAspectExtractor-0.0.1-SNAPSHOT-jar-with-dependencies.*``` the aspect extractor.

7. ```data/aos/dictionary``` the aspect extractor's dependency files

8. ```data/aos/rq2/*``` the rq2 file


## Dependencies
1.python=3.8.10

2.tensorflow=2.7.0

3.transformers=4.15.0


## Run
1. Depress the EASTER model file and jar file

2. Download RoBERTa pretrained model file from [HuggingFace](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment/tree/main)

3. Download stanfordcorenlp tool from [CoreNLP](https://nlp.stanford.edu/software/stanford-corenlp-4.5.1.zip)

4. Download Laptop and Restaurant dataset from [ACOS](https://github.com/NUSTM/ACOS/tree/main/data)

5. Configure the config.py

5. run python3 main.py to extract AOS Triple

5. run python3 EASTER.py to re-train and re-calculate shap value
