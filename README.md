# Dialogue Text Summarization Dokument

# DialogSum Dataset:
Please download the DialogSum dataset from the authors' github page: [here](https://github.com/cylnlp/DialogSum)

# Evaluating Multiview and Coref on DialogSum:
For training and evaluating each model please refere to the respective README of each project. Please note that our version of CODS is not equal to the original one. It contains our adaptation for perspective summarization.

[Multiview Repository](https://github.com/GT-SALT/Multi-View-Seq2Seq)

[Coref repository](https://github.com/seq-to-mind/coref_dial_summ)

[CODS repository](https://github.com/salesforce/ConvSumm)

# Getting Started
Install the requirements:

```
pip install -r requirements.txt
```

# Data Preprocessing
```
cd src/data
python split_labels.py --path=path/to/dialogsum/dataset
```

# Training Multi-Headed-CODS
Copy the processed data to `CODS/data/raw/new_dialogsum/`

Then 
```
cd CODS/src/preprocess/
python preprocess_data.py
python -m spacy download en
python extract_key_phrases.py
python segment_dialogue.py
```

Execute Training:
```
cd ../../
./run_train.sh
```
You can adjust the parameters in `run_train.sh`
