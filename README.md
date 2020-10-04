# SANER

This is the implementation of [Named Entity Recognition for Social Media Texts with Semantic Augmentation]() at EMNLP-2020.

## Citations

If you use or extend our work, please cite our paper at EMNLP-2020.
```
@inproceedings{nie-emnlp-2020-saner,
    title = "Named Entity Recognition for Social Media Texts with Semantic Augmentation",
    author = "Nie, Yuyang and
      Tian, Yuanhe  and
      Wan, Xiang and
      Song, Yan  and
      Dai, Bo",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2020",
}
```

## Requirements

- `torch==1.1.0`
- `spacy==2.2.4`
- `tqdm==4.38.0`
- `fastNLP==0.5`

## Downloading Pre-trained Embeddings

For English NER, we use two types of word embeddings, namely ELMo and BERT. Among them, ELMo can be automatically 
downloaded by running the script `run.py`; bert can be downloaded pre-trained BERT-large-cased 
from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz). 
If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

For Chinese NER, we also use three types of word embeddings, namely Tencent Embedding and ZEN. Among them, 
Tencent Embedding can be downloaded from [here](https://ai.tencent.com/ailab/nlp/embedding.html), ZEN can be downloaded from [here](https://github.com/sinovation/ZEN)

All pretrained embeddings should be placed in `./data/`

## Run on sample data

Run `run.py` to train a model on the small sample data under the `sample_data` directory.
