# AG News Classification using 🤗 transformers

![Alt text](dataset-cover.png)

AG is a collection of more than 1 million news articles. News articles have been gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of activity. ComeToMyHead is an academic news search engine which has been running since July, 2004. The dataset is provided by the academic comunity for research purposes in data mining (clustering, classification, etc), information retrieval (ranking, search, etc), xml, data compression, data streaming, and any other non-commercial activity. For more information, please refer to the [link](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html).

You can download the dataset from [here](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

### Description

The AG's news topic classification dataset is constructed by choosing 4 largest classes from the original corpus. Each class contains 30,000 training samples and 1,900 testing samples. The total number of training samples is 120,000 and testing 7,600.

The file classes.txt contains a list of classes corresponding to each label.

The files train.csv and test.csv contain all the training samples as comma-sparated values. There are 3 columns in them, corresponding to class index (1 to 4), title and description. The title and description are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). New lines are escaped by a backslash followed with an "n" character, that is "\n".

### About the file

The class ids are numbered 1-4 where 1 represents World, 2 represents Sports, 3 represents Business and 4 represents Sci/Tech.

| Class Index | Title | Description |
| - | - | - |
| Consists of class ids 1-4 where 1-World, 2-Sports, 3-Business, 4-Sci/Tech | Contains title of the news articles | Contains description of the news articles |

## Introduction

In this dataset, we have been given the title of news articles and their respective descriptions. The news could belong to any of the 4 types mentioned above - World, Sports, Business and Sci/Tech. We will select a pre-trained model from hugging-face hub and then fine tune it based on the requirement of the dataset.

Here, I'm using the `bert-base-uncased` model.

## Pre-processing phase

There are a lot of unnecessary symbols and punctuations in the dataset like `,`, `.`, `\`, `'`, `!` which are not very useful for training. We will filter them using the regular expression `re` library.

Also for creating each and every word to token would consume a lot of memory. So, to overcome this problem, we will remove the stopwords from dataset. 
A stop word is a commonly used word (such as `the`, `a`, `an`, `in`). 
We would not want these words to take up space in memory, or taking up valuable processing time. For this, we can remove them easily, by storing a list of words that we consider to be as stop words. 

After that we will tokenize the text using the pre-built tokenizer of the pre-trained model.

## Architecture

![Alt text](architecture/The-architecture-of-the-Fine-tuned-BERT-base-classifier.jpeg)

BERT is based on the Transformer model architecture, instead of LSTMs.It is a language model which is bidirectionally trained. This means it has a deeper sense of language context and flow compared to the single-direction language models. BERT makes use of a novel technique called Masked LM (MLM): it randomly masks words in the sentence and then it tries to predict them.

A Transformer works by performing a small, constant number of steps. In each step, it applies an attention mechanism to understand relationships between all words in a sentence, regardless of their respective position.

BERT needs the input to be massaged and decorated with some extra metadata:

- Token embeddings: A [CLS] token is added to the input word tokens at the beginning of the first sentence and a [SEP] token is inserted at the end of each sentence.
- Segment embeddings: A marker indicating Sentence A or Sentence B is added to each token. This allows the encoder to distinguish between sentences.
- Positional embeddings: A positional embedding is added to each token to indicate its position in the sentence.

![Alt text](architecture/0_XET3A5BmwES3qxgF.png)

There are four types of pre-trained versions of BERT depending on the scale of the model architecture. The one used for this dataset is:

BERT-Base: 12-layer, 768-hidden-nodes, 12-attention-heads, 110M parameters.

## Evaluation Metrics

- Trained for 3 epochs
- Training Runtime: 1527 secs
- Final training loss: 0.2042
- train_samples_per_second: 70.708
- Accuracy: 93%

| Class label/Metric | Precision | Recall | F1-score | Support |
| -: | :-: | :-: | :-: | :-: |
| World | 96% | 92% | 94% | 2530 |
| Sport | 97% | 99% | 98% | 2409 |
| Business | 90% | 91% | 90% | 2477 |
| Sci/Tech | 89% | 91% | 90% | 2477 |

## Final thoughts

The fine tuned model did performed decent in the testing phase. All metrics in the table are equal or above 90% except for the precision for Sci/Tech class.

The model was trained on colab and training for 3 epochs took nearly half an hour. Some ways to achieve a higher accuracy:

- Train the model on full train dataset of size 120,000 samples.
- Train for more epochs.
- Maybe try the bert-large model or gpt models but it would require a lot more computational power and even longer to train.
