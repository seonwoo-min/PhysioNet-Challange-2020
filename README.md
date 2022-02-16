# Bag of Tricks for Electrocardiogram Classification with Deep Neural Networks (Computing in Cardiology 2020)
Official Pytorch implementation of DSAIL_SNU team (6th place) | [Paper](https://www.cinc.org/archives/2020/pdf/CinC2020-328.pdf) <br/>
Classification of 12-lead ECGs - PhysioNet Challenge 2020 | [Challege](https://physionetchallenges.github.io/2020/) <br/>

## Abstract
Recent algorithmic advances in electrocardiogram (ECG) classification are largely contributed to deep learning. However, these methods are still based on a relatively straightforward application of deep neural networks (DNNs), which leaves incredible room for improvement. In this paper, as part of the PhysioNet / Computing in Cardiology Challenge 2020, we developed an 18-layer residual convolutional neural network to classify clinical cardiac abnormalities from 12-lead ECG recordings. We focused on examining a collection of data pre-processing, model architecture, training, and post-training procedure refinements for DNN-based ECG classification. We showed that by combining these refinements, we can improve the classification performance significantly. Our team, DSAIL_SNU, obtained a 0.695 challenge score using 10-fold cross-validation, and a 0.420 challenge score on the full test data, placing us 6th in the official ranking. <br>

## Data
The [Challenge](https://physionetchallenges.github.io/2020/) provides a training database with data files and a description of the contents and structure of these files. <br/>
Download the training data into the `data` folder from these links:
1. CPSC2018 training set, 6,877 recordings: [link](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_CPSC.tar.gz)
2. China 12-Lead ECG Challenge Database (unused CPSC2018 data), 3,453 recordings: [link](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_2.tar.gz)
3. St Petersburg INCART 12-lead Arrhythmia Database, 74 recordings: [link](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_StPetersburg.tar.gz)
4. PTB Diagnostic ECG Database, 516 recordings: [link](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_PTB.tar.gz)
5. PTB-XL electrocardiography Database, 21,837 recordings:[link](https://storage.googleapis.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_PTB-XL.tar.gz)
6. Georgia 12-Lead ECG Challenge Database, 10,344 recordings: [link](https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_E.tar.gz)

## How to run
We follow the [Challenge Submission](https://moody-challenge.physionet.org/2020/submissions) instructions. <br/>
Build a Docker image and run the code using the following steps:
```
    docker build -t image .
    docker run --gpus device=0 --ipc=host -it -v ~/data:/physionet/data -v ~/models:/physionet/models  -v ~/outputs:/physionet/outputs image bash
    python train_model.py data models
    python driver.py models data outputs
```
