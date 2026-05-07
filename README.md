# Graph-And-Text Exchange
The code for paper "GATE: Graph and Text Exchange for Zero-Shot ECG Classification with LLM Prompts"
# Requirements
In order to run this project, you need to install the following packages:
* pytorch 1.12
* python 3.8.10
* numpy 1.24.4
* transformers 4.46.3

you can run:
```
conda activate your_env
pip install requirement.txt
```

# :partying_face:Abstract
Electrocardiography (ECG) is a fundamental tool for diagnosing cardiovascular diseases, yet the scarcity of large-scale annotated data limits the applicability of supervised learning approaches. While self-supervised learning (SSL) has shown promise for ECG representation learning, existing methods often suffer from semantic distortion, insufficient spatial modeling, and a lack of integration with medical knowledge. To address these challenges, we propose GATE (Graph-And-Text Exchange), a novel multimodal SSL framework that enhances the quality of the representation of ECG through cross-modal exchange between graph-structured data and clinical ECG reports. GATE employs a spatiotemporal graph encoder to capture fine-grained intra- and inter-lead dependencies, and introduces a lexical knowledge-embedded codebook to enhance the semantic representation of clinical reports, facilitating effective graph-text alignment. During inference, GATE integrates a large language model with a domain-specific knowledge base to generate semantically enriched disease descriptions, enabling robust zero-shot classification. Extensive experiments on three real-world ECG datasets demonstrate that GATE outperforms state-of-the-art self-supervised and multimodal baselines under both low-resource and zero-shot settings. Notably, GATE achieves competitive performance even when trained on only 1\% of labeled data, highlighting its strong generalization and clinical potential. The code is available at [Alt](https://github.com/csu-anying/GATE/tree/main).
![](https://github.com/csu-anying/GATE/tree/main/pic/GATE.png)

# :triangular_flag_on_post:Dataset
We use four datasets for our experiments, the MIMIC-IV dataset is used for model pre-training, while the other three datasets are dedicated to downstream task fine-tuning and performance evaluation.

## :ballot_box_with_check:MIMIC-IV-ECG
You can download the dataset at this [link](https://physionet.org/content/mimic-iv-ecg/1.0/)<br>
It is a specialized sub-dataset within the broader MIMIC-IV repository, which contains approximately 800,000 diagnostic 12-lead electrocardiograms, each sampled at 500 Hz over 10 seconds, from nearly 160,000 unique patients in the MIMIC-IV Clinical Database. To ensure data quality, we exclude samples with empty or extremely short (fewer than three words) reports, and replace invalid ECG values such as NaN or INF with the average of six neighboring points. After preprocessing, a total of 771,693 ECG-report pairs are retained for pretraining the model. 

## :ballot_box_with_check:PTB-XL
You can download the dataset at this [link](https://physionet.org/content/ptb-xl/1.0.3/)<br>
The PTB-XL ECG dataset contains 21,837 10-second clinical 12-lead electrocardiograms from 18,885 patients. The dataset contained 52% of men and 48% of women, covering the age range 0-95 years. In the annotation file, there are 71 SCP-ECG statements for ECG annotation, which can be divided into three categories: diagnostic statements, formal statements, and rhythm statements. Based on diagnostic statements, the PTB-XL dataset provides five coarse superclasses (NORM, CD, MI, HYP and STTC). In this study, five coarse superclasses were selected for multi-label ECG classification experiments. 
## :ballot_box_with_check:CPCS2018
You can download the dataset at this [link](http://2018.icbeb.org/Challenge.html)<br>
The multi-label dataset, derived from the China Physiological Signaling Challenge 2018 (CPSC 2018), contained 6,877 12-lead electrocardiogram recordings ranging in time from 6 to 60 seconds. The dataset has 9 classes with a sampling frequency of 500hz, and we downsample the data to 100hz when using it. Due to inconsistent data length, we sampled all data to 10s; KFold was used to divide the data into 10 groups, and 10-fold cross-validation was used for the experiment.
## :ballot_box_with_check:Chapman
You can download the dataset at this [link](https://physionet.org/content/ecg-arrhythmia/1.0.0/)<br>
The Chapman-Shaoxing-Ningbo consists of 45,152 recordings sampled at 500 Hz from 45,152 patients.  The original dataset includes annotations of 11 common rhythms and 67 cardiovascular conditions, labeled by clinical experts. To ensure label reliability, ECG samples annotated as "unknown" are removed, yielding a cleaned dataset of 23,026 recordings with 38 clinically validated diagnostic labels. 

# :exclamation:Experimental Results
![](https://github.com/csu-anying/GATE/tree/main/pic/result.png)

# :question:How to use
## :pencil2:Data Preprocessing
First, in order to run the model, you need to run the data processing file.
```
cd data_preprocessing
python preprocess_mimic_iv.py
python preprocess_ptbxl.py
python preprocess_icbeb.py
python preprocess_chapman.py
```
Please remember to change the path in the file to the path where you store your data.
## :page_with_curl:Modify parameter file
You need to modify the `config.yaml` file, replacing the paths with your own. In addition, you can also set your running epoch and batch size.

## :arrow_forward:Run code 
### 1. Pretrain
```
cd pretrain
python pretrain_main.py
```
Completed checkpoints are saved in `checkpoints/pretrain`

### 2. Finetune

```
cd finetune
python finetune_main.py
```
Completed checkpoints are saved in `checkpoints/finetune`

### 3. LinearProbing
LinearProbing also runs in `finetune_main.py`, but you need to modify the parameters, specifically changing `name` to `LinearProbing`.
```
parser.add_argument('--name', default='LinearProbing', type=str, metavar='B',help='LinearProbing or Finetuning')
```

### 4. ZeroShot
```
cd zeroshot
python test_zeroshot.py
```


# Acknowledgement
We thank the codes for [link](https://github.com/cheliu-computation/MERL-ICML2024)

