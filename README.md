# Keep Me Sane Online - Part 1

An NLP based app to keep one sane online

## Description

More often then not we come across online social forums where people get into arguments they originally don't intend to which later becomes outrageous. I believe that if we could have a pop up warning about the insincerity of message everytime one egages on social media or even go ahead and suggest a better and polite phrasing of sentences then we can alleviate some of the toxic text contents online. In the part one of the implemlentation of this idea, I tried to build a text classifier to detect inappropriate sentences and integrate it to a simple web based application to demostrate its use case.

## Application

As mentioned in the description, the motivation behind this project was to promote more polite and sincere communication through digital media. The text classification model detects the toxicity in the given text data and classifies them into one or more of the foolowing classes: toxic, severe toxic, insult, obscene, identity hate and threat.

This could have application in almost any of the social media forum where people engage in coversations or discussions. Below is the demo of its applciation.

https://user-images.githubusercontent.com/43740782/235079120-c6d290f8-6a8a-4d68-8a17-34e2d6031afd.mov

## Text Classificaion

### Method

The underlying function uses a NLP model to classify a given piece of text. A custom model using the embeddings from the [BERT-base uncased](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4) was trained for classification task. The [Toxic Comment Classification](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) dataset was taken from Kaggle which had ~160K sentence belonging to one or more of six lables i.e toxic, severe toxic, insult, obscene, identity hate and threat.

The dataset was highly imbalance with only ~10% of dataset belonging to 1 or more of the classification labels where as ~90% of data did not belong to any of the toxic labels. To handle this, I took equal number of positive and negative label data points where positive label corresponds to data point belonging to atleast one of the six labels.

With this the a shallow neural network with three dense layer was trained on the embeddings generated using the Bert model. The final model had ~110M parameters with trainable parameters of ~110K.

![](https://github.com/Ayush-Mi/Keep_Me_Sane_Online/blob/main/images/model.png)

### Training

The model was trained on Macbook M1pro chip with 32gb memory for 5 epochs and each epoch took around 30 mins to complete. Since the labels were binary and one datapoint could have more than one labels associated with it, binary crossentropy loss function was used along with adam optimizer and F1 score as metric for evaluation.

![](https://github.com/Ayush-Mi/Keep_Me_Sane_Online/blob/main/images/loss.png)

### Results

With the above training method following results were observed
| | | |
| :--: | :--: | :--: |
![](https://github.com/Ayush-Mi/Keep_Me_Sane_Online/blob/main/images/F1_score.png) | ![](https://github.com/Ayush-Mi/Keep_Me_Sane_Online/blob/main/images/precision.png) | ![](https://github.com/Ayush-Mi/Keep_Me_Sane_Online/blob/main/images/output.png)


## How to run

To replicate this application the system must be running python >3.* . Clone this repository and download the dataset from [kaggle](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) and place it in folder ./data . Run the *train.ipynb* notebook which downloads the Bert Base uncased model and trains the neural network for text classification. To run the demo shown in the above video run `python app.py` from the present directory and go to url http://127.0.0.1:5000 on your local browser.

## Future works

This is just the part one of the project *Keep Me Sane Online* where we are only classifying the text. The part two would consist of a language generation model that takes in these inappropriate sentences and transforms them into more polite version of themselves.

It would be really challenging to transform some of the harsh language that we see online today but would also be worth exploring.

## References

- Dataset used was taken from [kaggle](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge)
- BERT Base Uncased from [tensorflow hub](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4)
- HTML code was taken from this Damian Boah's [blog](https://medium.datadriveninvestor.com/train-and-deploy-an-nlp-news-classifier-web-app-to-the-cloud-for-free-82655b6b32f4)
