# Tweets Sentiment Analysis



Sentiment analysis is a very important topic in Natural Language Processing (NLP). Helping machine to understand the sentiment in human language may benefit other NLP research as well. I tend to include sentiment analysis as one of the process in my NLP pipeline when I tried to solve real-life language problem since including sentiment as the meta-information may increase the model performance. 



I use this repository to train and test various sentiment analysis model for NLP tasks.

This repository consists with Python file and Jupyter Notebook. You can run the Python file to preprocess the data, train the model and perform inference from the command line. The Jupyter Notebook is mainly for interactive exploratory data analysis.



## Dataset Information:

The dataset is mainly consists with a "tweet text" field and a "sentiment" field.

The sentiment field is dived into three categories: "positive", "negative" and "neutral".

The dataset for this task can be extensive and adaptive.



## Requirements:

For the model construction part, I use PyTorch as the main machine learning library. 

Run the below command to install the required dependencies for this project.

```shell
pip install -r requirements.txt
```



## Training 

There is a sample [model file](https://github.com/zymlnlp/Tweets-Sentiment-Analysis/blob/master/model/best_model_state.bin) in the [model](https://github.com/zymlnlp/Tweets-Sentiment-Analysis/tree/master/model) folder mainly for the usage of demo. 

You can also train the model by yourself using different hyper-parameter (e.g. learning rate, number of epochs, etc.) in order to generate better performance. The hyper-parameter can be defined through the command line, and it is also accessible through the [train.py](https://github.com/zymlnlp/Tweets-Sentiment-Analysis/blob/master/code/train.py) file in the [code](https://github.com/zymlnlp/Tweets-Sentiment-Analysis/tree/master/code) folder.

In addition, you can use your preferred sentiment analysis dataset to train the model for your task. Make sure you modify the code or adapt your preferred dataset in consist with the dataset in this project.

```shell
python ./code/train.py 
```




## Inference & Demo

You can interact with the sentiment analysis model through the below command line in your terminal.

Before you run the inference file, please check if you have download the [model file](https://github.com/zymlnlp/Tweets-Sentiment-Analysis/blob/master/model/best_model_state.bin) and make sure it is located in the [model](https://github.com/zymlnlp/Tweets-Sentiment-Analysis/tree/master/model) folder.

Note: It could take a while to load the model depends on the hardware you are using.

```shell
python ./code/inference.py
```

 

## Exploratory Data Analysis

