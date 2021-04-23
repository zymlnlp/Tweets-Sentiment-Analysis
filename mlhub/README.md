# Zeyu Gao's Natural Language Processing Models

This [MLHub](https://mlhub.ai) package provides a demonstration and
command line tools built using Zeyu Gao's pre-built machine learning
models. The model can determine the sentiment of a sentence.

The source code is available from
<https://github.com/gjwgit/zynlp>.

## Quick Start

```console

$ ml sentiment zynlp The flight was bumpy and uncomfortable.
$ ml sentiment zynlp The meals were actually quite good.
```

## Usage

- To install mlhub (Ubuntu):

		$ pip3 install mlhub
		$ ml configure

- To install, configure, and run the demo:

		$ ml install   gjwgit/zynlp
		$ ml configure zynlp
		$ ml readme    zynlp
		$ ml commands  zynlp
		$ ml demo      zynlp
		
- Command line tools:

		$ ml sentiment zynlp [<sentence>] [(-f|--file) <file.txt>]
    
## Command Line Tools

In addition to the *demo* command below, the package provides a number
of useful command line tools.

### *sentiment*

The *sentiment* command will determine the general sentiment of a
sentence. The results are returned one sentence on a line, begining
with the sentiment, followed by the sentence.


```console
$ ml sentiment zynlp Hope you have a great weekend
```

The result returned is pipeline friendly:

```console
positive,Hope you have a great weekend
```

A text file can be specified and each sentence found will be analysed
separately.

## Demonstration

```console
$ ml demo zynlp
=============================================
Zeyu Gao's Sentiment Analysis Pre-Built Model
=============================================

Welcome to a demo of Zeyu Gao's pre-built model for Sentiment
Analysis.

Sentiment analysis is an example application of Natural Language
Processing (NLP). Sentences are assessed by the model to capture their
sentiment. See https://onepager.togaware.com.

The model load may take a few seconds.

Press Enter to continue: 

================================
Here's to having a glorious day.
================================

Passing the above text on to the pre-built model to determine
sentiment identifies the sentiment as being:

	positive

Press Enter to continue: 

=========================
That was a horrible meal.
=========================

Passing the above text on to the pre-built model to determine
sentiment identifies the sentiment as being:

	negative

Press Enter to continue: 

==========================
The chef should be sacked.
==========================

Passing the above text on to the pre-built model to determine
sentiment identifies the sentiment as being:

	negative

Press Enter to continue: 

============================
Hi there, hope you are well.
============================

Passing the above text on to the pre-built model to determine
sentiment identifies the sentiment as being:

	positive

Press Enter to continue: 

==========================
The sun has already risen.
==========================

Passing the above text on to the pre-built model to determine
sentiment identifies the sentiment as being:

	neutral

```
