# NeuralBERTClassifier for Medical Slot Filling

## Introduction

NeuralBERTClassifier is designed for quick implementation of neural models for multi-label classification problem: Medical Slot Filling (MSF). A salient feature is that NeuralBERTClassifier currently provides a variety of text encoders, such as FastText, TextCNN, TextRNN, RCNN, VDCNN, DPCNN, DRNN, AttentiveConvNet, Transformer encoder, and BERT etc. It also supports other text classification scenarios, including binary-class and multi-class classification. It is built on [PyTorch](https://pytorch.org/). Corresponding paper **Understanding Medical Conversations with Scattered Keyword Attention and Weak Supervision from Responses** was accepted by [AAAI 2020](https://aaai.org/ojs/index.php/AAAI/article/view/6412).

## Support tasks

* Binary-class text classifcation
* Multi-class text classification
* Multi-label text classification
* Hiearchical (multi-label) text classification (HMC)

## Support text encoders

* TextCNN ([Kim, 2014](https://arxiv.org/pdf/1408.5882.pdf))
* RCNN ([Lai et al., 2015](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552))
* TextRNN ([Liu et al., 2016](https://arxiv.org/pdf/1605.05101.pdf))
* FastText ([Joulin et al., 2016](https://arxiv.org/pdf/1607.01759.pdf))
* VDCNN ([Conneau et al., 2016](https://arxiv.org/pdf/1606.01781.pdf))
* DPCNN ([Johnson and Zhang, 2017](https://www.aclweb.org/anthology/P17-1052))
* AttentiveConvNet ([Yin and Schutze, 2017](https://arxiv.org/pdf/1710.00519.pdf))
* DRNN ([Wang, 2018](https://www.aclweb.org/anthology/P18-1215))
* Region embedding ([Qiao et al., 2018](http://research.baidu.com/Public/uploads/5acc1e230d179.pdf))
* Transformer encoder ([Vaswani et al., 2017](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf))
* Star-Transformer encoder ([Guo et al., 2019](https://arxiv.org/pdf/1902.09113.pdf))

## Requirement

* Python 3
* PyTorch 0.4+
* Numpy 1.14.3+

## Usage

### Training

    python train.py conf/train.json

***Detail configurations and explanations see [Configuration](readme/Configuration.md).***

The training info will be outputted in standard output and log.logger\_file.

### Evaluation
    python eval.py conf/train.json

* if eval.is\_flat = false, hierarchical evaluation will be outputted.
* eval.model\_dir is the model to evaluate.
* data.test\_json\_files is the input text file to evaluate.

The evaluation info will be outputed in eval.dir.

## Input Data Format

    JSON example:

    {
        "doc_label": ["Computer--MachineLearning--DeepLearning", "Neuro--ComputationalNeuro"],
        "doc_token": ["I", "love", "deep", "learning"],
        "doc_keyword": ["deep learning"],
        "doc_topic": ["AI", "Machine learning"]
    }

    "doc_keyword" and "doc_topic" are optional.


## Update

* 2020-10-27
