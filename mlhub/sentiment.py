# -*- coding: utf-8 -*-
#
# Time-stamp: <Tuesday 2021-04-20 16:52:27 AEST Graham Williams>
#
# Copyright (c) Togaware Pty Ltd. All rights reserved.
# Licensed under the MIT License.
# Author: Graham.Williams@togaware.com
#
# This demo using a pre-built model from
# https://github.com/zymlnlp/Tweets-Sentiment-Analysis

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

import os
import argparse

import torch
import warnings

from nlp_model import SentimentClassifier
from preprocessing import tokenizer
from mlhub.utils import get_cmd_cwd

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# Process the command line.
# -----------------------------------------------------------------------

option_parser = argparse.ArgumentParser(add_help=False)

option_parser.add_argument(
    'sentence',
    nargs='*',
    help='sentence to analyse')

option_parser.add_argument(
    '--file', '-f',
    help='path to a text file to analyse sentences')

args = option_parser.parse_args()

# ----------------------------------------------------------------------
# Read the sentences to be analysed.
# ----------------------------------------------------------------------

text = ""
if args.file:
    text = open(os.path.join(get_cmd_cwd(), args.file), "r").read()
elif args.sentence:
    text = " ".join(args.sentence)

# Split the text into a list of sentences. Each sentence is sent off
# for analysis.

text = " ".join(text.splitlines())
text = " ".join(text.splitlines())
text = text.replace(". ", "\n")
text = text.splitlines()

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

if len(text):
    for sentence in text:

        encoded_tweet = tokenizer.encode_plus(
            sentence,
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

        print(f'"{sentence}",{sentiment_map[prediction.numpy()[0]]}')
