# Tweets Sentiment Analysis



Sentiment analysis is a very important topic in Natural Language Processing (NLP). Helping machine to understand the sentiment in human language may benefit other NLP research as well. I tend to include sentiment analysis as one of the process in my NLP pipeline when I tried to solve real-life language problem since including sentiment as the meta-information may increase the model performance. 



I use this repository to train and test various sentiment analysis model for NLP tasks.

This repository consists with Python file and Jupyter Notebook. You can run the Python file to preprocess the data, train the model and perform inference from the command line. The Jupyter Notebook is mainly for interactive exploratory data analysis.



## Dataset Information:

The dataset is mainly consists with a "tweet text" field and a "sentiment" field.

The sentiment field is dived into three categories: "positive", "negative" and "neutral".

The dataset for this task can be extensive and adaptive.



## Requirements:

For the model building part, I use PyTorch as the main machine learning library. 





## Training 

You can train the model by yourself, or use your own sentiment analysis dataset to train the model for your task.
```shell
python ./code/train.py
```




## Inference

You can interact with the sentiment analysis model through the below command line in your terminal

```shell
python ./code/inference.py
```

 



