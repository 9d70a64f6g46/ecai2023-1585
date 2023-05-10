# SHAP-Enhanced Syntatic Analysis for Aspect-based Sentiment Analysis based on Labeled Sentiments Only

  In this paper, we propose an lightweight, AOS-Triplet-Oriented, compound approach that first solely utilizes the labeled sentiments to train a classifier, then deduces both opinions and aspects based on our syntactic analysis guided by the calculated contribution values for each word in the texts from our trained classifier. Specifically, our approach that outputs the (a,o,s) triplet consists of three stages. **First**, we use a RoBERTa-incorporated TextCNN framework to train a sentiment polarity (i.e., positive, neutral, and negative) classifier. **Second**, we use the AI-Explanation framework [SHAP (SHapley-Additive-exPlanation)](https://github.com/slundberg/shap) to analyze our classification results, and select the representative words as the opinions based on the SHAP values. **Third**, we locate aspects for the extracted opinions based on our proposed syntactic rules considering the word dependencies in the texts. The schematic diagram of our approach is shown below.

![](https://github.com/9d70a64f6g46/ecai2023-1585/blob/main/img/figure2.png);

  Our evaluation is based on two widely-researched datasets, Laptop-ACOS and Restaurant-ACOS, which are newly proposed by [[Cai et al.@ACL2021]](https://github.com/NUSTM/ACOS) for evaluating the ACOS quadruple extraction task. The baselines of our evaluation are two state-of-the-art (SOTA), [Extract-Classify-ACOS](https://github.com/NUSTM/ACOS) and [Span-ASTE](https://github.com/chiayewken/Span-ASTE), which consider (a,c,o,s) quadruples and (a,o,s) triplets, respectively. When reproducing two baselines, we strictly followed their published code. The experimental results indicate that our approach is a lightweight, explainable ABSA approaches that achieves a comparable level of performance (especially for solely extracting Aspects) to SOTA learning-based approaches, while it only requires the labeled sentiments. 

  Since the benchmark dataset we use is not self generated and published, we do not provide them here. If you want to use any of them, you should complete the license for their publication and consider referencing the original paper. You can download them from the link we provided above.

## Overview
1. ```EASTER.py``` the codes of EASTER. To extract sentiment and shap value.

2. ```main.py``` the codes of AOSExtractor. To extact AOS base on sentiment and shap value.

3. ```config.py``` the configurations of project.

4. ```SentiAspectExtractorIntroduction.pdf``` the detailed introduction of the aspect extraction rules in stage three.

5. ```data/aos/pretrained``` the pretrained model weights file of EASTER. 

6. ```data/aos/*.pkl``` the sentiment and shap value file.

7. ```data/aos/SentiAspectExtractor-0.0.1-SNAPSHOT-jar-with-dependencies.*``` the aspect extractor.

8. ```data/aos/dictionary``` the aspect extractor's dependency files

9. ```data/aos/rq2/*``` the rq2 file


## Dependencies
1. python=3.8.10

2. tensorflow=2.7.0

3. transformers=4.15.0


## Run
**Preparation**:

1. Depress the EASTER model file and jar file

2. Download RoBERTa pretrained model file from [HuggingFace](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment/tree/main)

3. Download stanfordcorenlp tool from [CoreNLP](https://nlp.stanford.edu/software/stanford-corenlp-4.5.1.zip)

4. Download Laptop and Restaurant dataset from [ACOS](https://github.com/NUSTM/ACOS/tree/main/data)

5. Configure the config.py

**Running**:

1. run python3 EASTER.py to train and calculate shap value：

   Now the folder has provided the data used in our experiment. If you want to extract AOS from the data our experimental data, you can skip this step. If you want to conduct experiments on new data, this step is required.

   For more details on our Stage One classifier, please refer to [[Sun et al.@Internetware2022]](https://github.com/xiaobo-lab/EASTER)

2. run python3 main.py to extract AOS Triple 

