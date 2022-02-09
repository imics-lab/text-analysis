# Understanding Social Anxiety: Detecting Stress Factors through Text Analysis

# Table of Contents
1. [About this repository](#about)
2. [Abstract](#abstract)
3. [How to run this code](#run)
4. [An explanation of each file](#explain)

## About this Repository <a name = "about"></a>
This repository holds the code used in our experiments detailed in our paper, "Understanding Social Anxiety: Detecting Stress Factors through Text Analysis." Below, you will find instructions for [how to run our code](#run), as well as an [explanation of what each file does](#explain). 

## Abstract <a name = "abstract"></a>
Approximately one-third of the veteran population suffers from post-traumatic stress disorder, a mental illness that is often co-morbid with social anxiety disorder. Student veterans are especially vulnerable as they struggle to adapt to a new, less structured lifestyle with few peers who understand their difficulties. To support mental health experts in the treatment of social anxiety disorder, this study utilized machine learning to detect stress in text transcribed from interviews with patients and applied topic modeling to highlight common stress factors for student veterans. We approach our stress detection task by exploring both deep learning and traditional machine learning strategies such as transformers, transfer learning, and support vector classifiers. Our models provide a tool to support psychologists and social workers in treating social anxiety. The results detailed in our paper could also have broader impacts in fields such as pedagogy and public health.

## How to Run This Code <a name="run"></a>
You can execute any of the python files in this repository via the terminal: 

```python3 script_name.py```

In order for the code to execute, you will need to include a config file containing the following information:
* The filepath to the dataset 
* The filepath to the labels
* The writepath for results

You should format your config file like the example below:

```
[appSettings]
datapath = path\to\data.csv
targetpath = path\to\labels.csv

[writeSettings]
resultpath = path\to\results.csv
```

## Directory of Files <a name="explain"></a>
