# Graph convolutional network learning model based on new integrated data of the protein-protein interaction network and network pharmacology for the prediction of herbal medicines effective for the treatment of metabolic diseases

## Abstract
-------------
Chronic metabolic diseases constitute a group of conditions requiring long-term management and hold significant importance for national public health and medical care.
Currently, in oriental medicine, there are no insurance-covered herbal prescriptions designated primarily for the treatment of metabolic diseases. 
Therefore, the objective of this study was to identify herbal prescriptions from the existing pool of insurance-covered options that could be effective in treating metabolic diseases. 
This research study employed a graph convolutional network (GCN) learning model to identify suitable herbal prescriptions for various metabolic diseases, diverging from literature-based approaches based on classical indications, through network pharmacology.
Additionally, the derived herbal medicine candidates were subjected to transfer learning on a model that binarily classified the approved drugs into those currently used for metabolic diseases and those that are not for data-based verification.
GCN, adept at capturing patterns within protein-protein interaction (PPI) networks, was utilized for classifying and learning the data. 
Moreover, protein scores related to the diseases were extracted from GeneCards and used as weights. Due to the absence of any prior research using similar data and learning structures, an alternative evaluation method of our pre-trained model was deemed necessary. 
The performance of the pre-trained model was validated through 5-fold cross-validation and bootstrapping with 100 iterations. 
Furthermore, to ascertain the superior performance of our proposed model, the number of layers was varied, and the performance of each was evaluated.
Our proposed model structure achieved outstanding performance in classifying drugs, with an average precision of 96.68%, recall of 97.18%, and an F1 score of 96.74%. 
The trained model predicted that the most effective decoction would be Jowiseunggi-tang for hyperlipidemia, Saengmaegsan for hypertension, and Kalkunhaeki-tang for type 2 diabetes. This study is the first of its kind to integrate GCN with network pharmacology, PPI networks, and protein weights.


## Each disease dataset
>HLP (Hyperlipidemia)
>> Positive(11) : Hyperlipidemia dataset
>> 
>> Negative(11)  : other drug dataset (acitretin, APAs, aspirin, azathioprine, Calcipotriol, ciclosporin, Hydroxychloroquine, Leflunomide, mesalazine, Methotrexate, ticlopidine)


>HTN (Hypertension)
>> Positive(16) : Hypertension dataset
>> 
>> Negative(16) : other drug dataset (ALL dataset)


>T2D (Type 2 Diabetes)
>> Positive(16) : Type 2 diabetes dataset
>> 
>> Negative(16) : other drug dataset (ALL dataset)
###### *HLPs are less numerous than other data to ensure a 1:1 ratio of positive to negative data


## Databases
+ [STRING](https://string-db.org/cgi/input?sessionId=bsB4FyslFsBf&input_page_show_search=on) 
+ [STITCH](http://stitch.embl.de/cgi/input.pl?UserId=7KQ28lWyBCV2&sessionId=zoXwJ5hyBL9R)
+ [Genecards](https://www.genecards.org/)
+ [HIRA](https://opendata.hira.or.kr/op/opc/olapMjDiseaseInfoTab1.do#none)
+ [KFDA](https://www.mfds.go.kr/index.do)

## Quick Start

1. Pre-trained with "GCN pre-training.py" (file paths are all annotated).
2. Based on the learned data, apply it to herbal prescription data through the "learning to herbal prescriptions.py".
> 5 fold cross validation: used to check the performance of GCN pre-training model
> Bootstrapping: used to check the performance of GCN pre-training models

## Python Requirements 
+ Python 3.11.2
+ Pytorch 2.1.0+cu121 
+ Pandas 2.1.2
+ Numpy 1.26.1
+ Matplotlib 3.8.0
+ Seaborn 0.13.1
+ CUDA 12.1

## License
This codes released under [MIT License](https://github.com/Rapoudok/GCN-herbal-medicine/blob/main/LICENSE).
