from flask import Flask,render_template,url_for,request
import flask
import pandas as pd 
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_hub as tf_hub
import tensorflow_text as tf_text
import os


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
model = tf.keras.models.load_model(os.getcwd() +"/model", 
				   custom_objects={'f1_m':f1_m,
 		                            'precision_m': precision_m,
 				                    'recall_m': recall_m})


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        message = flask.request.form['comment']
        data = [message]
        vect = model.predict([data])
        if  ((np.where(vect[0]>0.8)[0]).size == 0):
            my_prediction = "Looks safe to post online"
        else:
            pred = [labels[x] for x in list(np.where(vect[0]>0.5)[0])]
            my_prediction = "Seems: "+ str(pred) + " try rephrasing."
        return flask.render_template('main.html',
                                     original_input={'Given Text':message},
                                     result=my_prediction)
    else:
        return flask.render_template('main.html',
                                     original_input={'Given Text':message},
                                     result=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
        

