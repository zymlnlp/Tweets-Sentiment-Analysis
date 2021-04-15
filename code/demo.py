import torch
import warnings
from preprocessing import tokenizer
from nlp_model import SentimentClassifier

warnings.filterwarnings("ignore")

sentiment_map = {2: "neutral", 1: "positive", 0: "negative"}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_LEN = 256

if __name__ == '__main__':

    # Load the model.

    try:
        model = SentimentClassifier(n_classes=3)
        model.load_state_dict(torch.load("./model/best_model_state.bin",
                                         map_location=torch.device('cpu')))
    except:
        print("No model found, you need to train the model first.")
        exit()

    samples = ["Here's to having a glorious day.",
               "That was a horrible meal.",
               "The chef should be sacked.",
               "Hi there, hope you are well.",
               "The sun has already risen."]

    for input_text in samples:

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

        print(f'\nSentence  : {input_text}')
        print(f'Sentiment : {sentiment_map[prediction.numpy()[0]]}')
