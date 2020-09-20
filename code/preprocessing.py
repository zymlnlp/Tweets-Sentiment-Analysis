import torch
from transformers import *

# transform sentiment into numerical label
sentiment_map = {"neutral": 2, "positive": 1, "negative": 0}

# construct tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True)

# max_input_length is 512
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

# sentence tokenization function
# pass in the sentence, tokenize it, and cut it if the length exceed max_input_length
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens

def preprocess(data):

    # drop the null data
    data.dropna(inplace=True)

    # change label to numerical representation
    sentiment_num = []

    for s in data["sentiment"]:
        sentiment_num.append(sentiment_map[s])

    data["sentiment_num"] = sentiment_num

    return data


class TweetDataset(torch.utils.data.Dataset):

    def __init__(self, tweet_texts, targets, tokenizer, max_len):
        self.tweet_texts = tweet_texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweet_texts)

    def __getitem__(self, item):
        tweet_text = str(self.tweet_texts[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            tweet_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'tweet_text': tweet_text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


# data loader
def data_loader(data, tokenizer, max_len, batch_size):
    ds = TweetDataset(
        tweet_texts=data["text"].to_numpy(),
        targets=data["sentiment_num"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        # num_workers=4
    )
