# Reproduction Procedure for Training the ReactiTwin Relation Network Model

This document specifies the procedure used to reproduce the training of the Relation Network model integrated into ReactiTwin. It describes the software environment, dataset construction, pre-processing configuration, and model training procedure.

## 1. Software Environment

- Install python 3.10
- The training and pre-processing procedures were conducted in a Python virtual environment using the following package versions:

    ```bash
    pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.4.2 tensorflow==2.15.0 matplotlib==3.7.2 seaborn==0.13.2
    ```


## 2. Dataset Construction

The training dataset was derived from the TON_IoT dataset, which is available from the following repository:

**https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EvBTaetotpdGnW7rJQ8fCvYBh8063CNeY9W33MpRsarJaQ?e=yZlnxW**

- The network security-event datasets were retrieved from:

    **TON_IoT datasets/SecurityEvents_GroundTruth_datasets/SecurityEvents_Network_datasets/**

- For each dataset file, the observations were filtered according to the `type` attribute. Only observations labelled `backdoor` and `normal` were retained, while observations associated with all other labels were excluded.

- The retained observations from the different files were then merged into a single CSV file: *Ton_IoT_Backdoor_dataset.csv*

- A copy of the dataset containing this retained observations is available and can be downloaded at:  **https://mega.nz/file/vfwHBLYB#9c6EjNVGzRwyGLsTMVRnOwgaCAduyzY3lTEPAWgRF40** 

This dataset constitutes the input to the subsequent pre-processing stage.

## 3. Dataset Pre-processing

- Dataset pre-processing is performed using the following Python script: *RelNet_Model_Preprocessing.py*

- The input and output paths are specified at lines 99 and 100 of the script. The configuration must reference:

    *Ton_IoT_Backdoor_dataset.csv* as the input dataset and: *Ton_IoT_Backdoor_dataset_cleaned_feature.csv* as the output file.

- The pre-processing procedure is executed from the configured Python virtual environment using: *python RelNet_Model_Preprocessing.py*

- The resulting file is *Ton_IoT_Backdoor_dataset_cleaned_feature.csv*


## 4. Relation Network Training

- Model training is performed using the following Python script: *RelNet_Model_Building.py*

- The path to the pre-processed dataset is specified at line 22 of the script: *Ton_IoT_Backdoor_dataset_cleaned_feature.csv*

- The training procedure is then executed within the same Python virtual environment: *python RelNet_Model_Building.py*

This procedure trains the Relation Network model used in ReactiTwin from the pre-processed TON_IoT observations. 

The execution of *RelNet_Model_Building.py* generates the following output files:

    `support.npy` , NumPy array containing the observations that constitute the support set used for the Relation Networks model deployement.

    `support_labels.npy` , NumPy array containing the labels associated with the observations in the support set, with a one-to-one correspondence between             each observation and its corresponding label.

    `embedder.h5` , HDF5 file containing the trained observation encoder model used in the few-shot learning configuration.

    `relation.h5` , HDF5 file containing the trained relation model used in the few-shot learning configuration.

    `onehot_column_name.pkl` , Pickle file containing the fitted One-Hot encoder for the `colum_name` categorical feature.Five such files are generated, one for       each of the five specified columns.

The dataset construction, pre-processing, and training stages should be performed in the specified order to reproduce the experimental configuration.
