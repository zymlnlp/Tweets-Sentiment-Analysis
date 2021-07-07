# -*- coding: utf-8 -*-
#
# Time-stamp: <Tuesday 2021-04-20 16:01:02 AEST Graham Williams>
#
# Copyright (c) Togaware Pty Ltd. All rights reserved.
# Licensed under the MIT License.
# Author: Graham.Williams@togaware.com
#
# This demo using a pre-built model from
# https://github.com/zymlnlp/Tweets-Sentiment-Analysis

import torch
import warnings

from preprocessing import tokenizer
from nlp_model import SentimentClassifier
from mlhub.pkg import mlask, mlcat

mlcat("Zeyu Gao's Sentiment Analysis Pre-Built Model", """\
Welcome to a demo of Zeyu Gao's pre-built model for Sentiment Analysis.

Sentiment analysis is an example application of Natural Language
Processing (NLP). Sentences are assessed by the model to capture their
sentiment. See https://onepager.togaware.com.

The model load may take a few seconds.
""")

warnings.filterwarnings("ignore")

sentiment_map = {2: "neutral", 1: "positive", 0: "negative"}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_LEN = 256

# Load the model.

mfile = "./model/best_model_state.bin"

try:
    model = SentimentClassifier(n_classes=3)
    model.load_state_dict(torch.load(mfile,
                                     map_location=torch.device('cpu')))
except Exception:
    print("No model found. Bad model file or model not yet trained.")
    print(f"Tried loading '{mfile}'.")
    exit()

samples = ["Here's to having a glorious day.",
           "That was a horrible meal.",
           "The chef should be sacked.",
           "Hi there, hope you are well.",
           "The sun has already risen."]

for text in samples:

    mlask(end="\n")

    mlcat(f"{text}", """\
Passing the above text on to the pre-built model to determine
sentiment identifies the sentiment as being:
""")

    encoded_tweet = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoded_tweet['input_ids'].to(device)
    attention_mask = encoded_tweet['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    print(f'\t{sentiment_map[prediction.numpy()[0]]}\n')

