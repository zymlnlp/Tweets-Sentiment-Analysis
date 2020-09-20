import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import *

from nlp_model import SentimentClassifier
from preprocessing import data_loader, preprocess, tokenizer

warnings.filterwarnings("ignore")

# Define Constants
EPOCHS = 3
BATCH_SIZE = 16
MAX_LEN = 256

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):

    # change to traning mode
    model = model.train()

    losses = []
    correct_predictions = 0

    for index, d in enumerate(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if index % 100 == 0:
            print("Iteration {}/{}, loss is {}".format(index, len(data_loader), loss))

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):

    # change to evaluation mode
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for index, d in enumerate(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            #if index // 20 == 0:
            #    print("Iteration {}/{}, loss is {}".format(index, len(data_loader), loss))

    return correct_predictions.double() / n_examples, np.mean(losses)



if __name__ == "__main__":

    # read train and test csv file
    train = pd.read_csv("./data/train.csv")
    # test = pd.read_csv("./data/test.csv")

    # preprocess the data
    train = preprocess(train)

    print(train.shape)

    # train validation split
    train, validation, _, _ = train_test_split(train, train, test_size=0.2, random_state=42)

    # construct dataloader
    train_data_loader = data_loader(train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = data_loader(validation, tokenizer, MAX_LEN, BATCH_SIZE)
    # test_data_loader = data_loader(test, tokenizer, MAX_LEN, BATCH_SIZE)

    # construct model
    model = SentimentClassifier(n_classes = 3)
    model = model.to(device)

    # define AdamW optimizer from the tranformers package
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

    # total steps during training process
    total_steps = len(train_data_loader) * EPOCHS

    # use a warm-up scheduler as suggested in the paper
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # define cross entropy loss for classification problem
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # this will be store "train_acc", "train_loss", "val_acc" and "val_loss"
    history = defaultdict(list)

    # best accuracy from the best model across the whole training process
    best_accuracy = 0

    # the main training for loop
    for epoch in range(EPOCHS):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        start_time = time.time()
        print("Current Epoch starts at: {}".format(time.ctime()))

        # training process
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        # validation process
        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(validation)
        )

        print(f'Val loss {val_loss} accuracy {val_acc}')

        print("Current Epoch ends at: {}".format(time.ctime()))
        print("Time used for training the current epoch:{} \n".format(time.time()-start_time))

        # put all the accuracy and loss information into the history dictionary
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        # identify and save the best model
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
