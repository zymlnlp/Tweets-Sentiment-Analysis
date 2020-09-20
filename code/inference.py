import torch
from torch import nn
import warnings
from preprocessing import tokenizer
from nlp_model import SentimentClassifier
warnings.filterwarnings("ignore")

# sentiment_map = {"neutral": 2, "positive": 1, "negative": 0}
sentiment_map = {2:"neutral", 1:"positive", 0:"negative"}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256


if __name__ == '__main__':

    # load in the model
    try:
        model = SentimentClassifier(n_classes=3)
        model.load_state_dict(torch.load("./model/best_model_state.bin", map_location=torch.device('cpu')))
    except:
        print("No model found, you need to train the model first.")
        exit()


    while True:
        input_text = str(input("Please input tweet text:"))

        if input_text == "exit program":
            break

        if len(input_text) == 0:
            print("No input text detected, will use sample text: 'I like this movie.' ")
            input_text = "I like this movie."

        encoded_tweet = tokenizer.encode_plus(
            input_text,
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

        print(f'Tweet text: {input_text}')
        print(f'Sentiment  : {sentiment_map[prediction.numpy()[0]]} \n')


def get_predictions(model, data_loader):

    model = model.eval()

    tweet_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            texts = d["tweet_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = nn.functional.softmax(outputs, dim=1)

            tweet_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return tweet_texts, predictions, prediction_probs, real_values
