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
========
Bing Map
========

Welcome to Bing Maps REST service. This service can find the the
latitude and longitude coordinates that correspond to location
information provided  as a query string.

Press Enter to continue: 

=======
GEOCODE
=======

 This part is to generate the latitude and longitude coordinates based
on the query. The result might be several. Here we set the query to
Priceline Pharmacy Albany Creek. In this case, it will generate a pair
of coordinates.

Press Enter to continue: 


Latitude: -27.35361099243164 Longitude: 152.96832275390625
