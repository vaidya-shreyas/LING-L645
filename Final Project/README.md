# Emotion-recognition-using-BERT

`Emotions` play a crucial role in human existence and have an impact in our day-to-day lives, our mental and physical well-being, as well as our ability to make decisions. Consequently, it is crucial to create emotion detection models since they have a wide range of uses and social narratives. `Emotion classification or recognition` is the task of identifying emotions from natural language data such as reviews, blogs, news articles, and so on. Some of the example applications of emotion recognition being Emotional-aware virtual assistants or chatbots, Social Media sentiment analysis, Consumer Analysis etc. 

This project focuses on multi-label emotion classification by using language models, mainly BERT and its variants, with the primal focus being analyzing their performance. The efficacy of the methods is demonstrated by experiments carried out on Twitter and Wikipedia (proposed) datasets.



## Table of Contents

- [Challenges](#Challenges)
- [Highlights](#Highlights)
- [Dataset](#Dataset)
- [Experimental Setup](#Experimental-Setup)
- [Future Scope](#Future-Scope)
- [References](#References)
- [End Notes](#End-Notes)


## Challenges

Numerous applications have triggered many researchers to work in the domain resulting in a myriad of NLP models and approaches. They focus mainly on positive and negative sentiment analysis, single label emotion classification and so on. This stated, however, there exist two key challenges while classifying emotions, as follows.

- `Overlapping Emotions`: Standard methods for classifying emotions tend to deal with each emotion separately. However, emotions are not autonomous; a particular emotive expression might be connected to a variety of emotions. Such methods fail to account for situations where many emotions may overlap.

- `Dearth of labeled data`: As emotions are not only subjective but also fuzzy, with multiple emotions occurring at once. As a result, the creation of resources connected to emotions, such as training data, has been constrained to a few manually annotated datasets or lexicons, a labor-intensive and expensive procedure.


## Highlights

Following are the highlights of the project:
- Multi-Label Emotion recognition on Twitter and Wikipedia datasets using BERT and its variants, namely BERTLarge and DistillBERT
- Twitter dataset from Part 5 of SemEval 2018 Task 1, with tweets and 11 labels which are one hot encoded in form of 0s and 1s, where 1 means emotion is present.
- Code template Reference from SpanEmo (Source code for the paper "SpanEmo: Casting Multi-label Emotion Classification as Span-prediction" in EACL2021. [[1]](#1))
- Code template was replicated with additional of Bert variants and inclusion of prerocessing and data loading for given mentioned datasets.
- Data Preprocessing
  - Use of pre-trained GloVe 6B 100 embeddings to create a distributed word representation.
  - TextPreProcessor module of ekphrasis preprocessor, a tool designed specifically for Twitter Data (Normalisation, Tokenization, Segmentation etc.)
- Model Training
  - Use of PyTorch and a Tesla P4 GPU with 4 cores of 16GB each on Google Cloud Vertex AI Jupyter Notebook to run all experiments.
  - BERT with feature dimension of 786, batch size of 32, dropout rate of 0.1
  - Loss function as cross-entropy with Adam optimizer.
  - An early stop patience of 10 and 20 epochs.
  - Use of transfer learning
- Analysing Model efficacy with F1-Macro, F1-Micro and Jaccard Similarity Scores.


## Dataset

`Affect in Tweets Dataset, Semeval-2018 Task 1`: This dataset consists of user IDs, around 6840 tweets, and 11 labels which are one hot encoded in form of 0s and 1s, where 1 means emotion is present.

`Jigsaw Toxic Comment Classification`: This dataset consists of over half million Wikipedia comments and have been labeled by human users for toxic behavior with 6 labels.


## Experimental Setup


### Files Structure

- `data/twitter_train.txt`: Labeled dataset of 6840 tweets for training.
- `data/twitter_dev.txt`: Validation dataset of around 900 tweets.
- `data/twitter_test.txt`: Test dataset with 2855 tweets.
- `data/wikipedia_train.csv`: Wikipedia dataset for training with around 0.5 million comments and 6 labels.
- `data/wikipedia_dev.csv`: Wikipedia dataset for validation.
- `data/wikipedia_test.csv`: Wikipedia dataset for testing.
- `scripts/data_loader.py`: Python script for all the data loading and preprocessing routines.
- `scripts/learner.py`: Python script for training process orchestration.
- `scripts/model.py`: Python script to define model architectures.
- `scripts/test.py`: Python script for test routines.
- `scripts/train.py`: Python script for train routines.
- `process_wikipedia-data.py`: Python script to divide wikipedia data into train and validation.
- `run.sh`: Shell script to execute the entire workflow
- `requirements.txt`: Listing of Python requirements

### Usage

> *Execute run.sh on terminal as follows : bash run.sh*

Please note that the current version is tested on Twitter dataset with results attached. Provisional design and setup implemented for Wikipedia Dataset


## Future Scope

- Implementation with different language models with different parameter settings to compare results.
- Execution with Wkipedia dataset to compare results with increase in data complexity. The given results are yet to be produced due to system and resource limitations.


## References

<a id="1">[1]</a> Hassan Alhuzali Sophia Ananiadou, 2017. SpanEmo: Casting Multi-label Emotion Classification as Span-prediction.

<a id="2">[2]</a> Saif M. Mohammad and Peter D. Turney. 2013. Crowdsourcing a word-emotion association lexicon. 29(3):436–465.

<a id="3">[3]</a> Soujanya Poria, Alexander Gelbukh, Erik Cambria, Amir Hussain, and Guang-Bin Huang. 2014. Emosenticspace: A novel framework for affective common-sense reasoning. Knowledge-Based Systems, 69:108–123.

<a id="4">[4]</a> Felipe Bravo-Marquez, Eibe Frank, Saif M Mohammad, and Bernhard Pfahringer. 2016. Determining word-emotion associations from tweets by multilabel classification. In 2016 IEEE WIC/ACM International Conference on Web Intelligence (WI), pages 536–539. IEEE.

<a id="5">[5]</a> Laura-Ana-Maria Bostan and Roman Klinger. 2018. An Analysis of Annotated Corpora for Emotion Classification in Text. In Proceedings ofthe 27th International Conference on Computational Linguistics, pages 2104–2119.

<a id="6">[6]</a> Elvis Saravia, Hsien-Chi Toby Liu, Yen-Hao Huang, Junlin Wu, and Yi-Shin Chen. 2018. CARER: Contextualized Affect Representations for Emotion Recognition. In Empirical Methods in Natural Language Processing, pages 3687–3697.

<a id="7">[7]</a> Wenbo Wang, Lu Chen, Krishnaprasad Thirunarayan, and Amit P Sheth. 2012. Harnessing twitter” big data” for automatic emotion identification. In Privacy, Security, Risk and Trust (PASSAT), 2012 International Conference on and 2012 International Confernece on Social Computing (SocialCom), pages 587–592. IEEE.


## End Notes

Feel free to discuss your experiences on the [discussion page](https://github.com/vaidya-shreyas/LING-L645/discussions).

[Back to Top](#Emotion-recognition-using-BERT)