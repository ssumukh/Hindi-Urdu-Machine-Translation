# Hindi-Urdu-Machine-Translation

### Approach:
We tried both statistical machine translation models, and neural machine translation models for this task. The dataset can not be made public as it was part of a contained study, but these models can be used for any 2 languages.

##### Statistical Machine Translation (SMT) model:
We used a phrase based SMT model, with Giza++ to get the word alignment.

##### Neural Machine Translation (NMT) model:
We had 4 models in place:
* A baseline seq-2-seq model, using LSTM
* Neural Machine Translation By Jointly Learning To Align And Translate ([paper link](https://arxiv.org/pdf/1409.0473.pdf))
* Effective Approaches to Attention-based Neural Machine Translation ([paper link](https://www.aclweb.org/anthology/D15-1166/))
* Modeling Coverage for Neural Machine Translation ([paper link](https://www.aclweb.org/anthology/P16-1008/))

Presentation for this can be found [here](https://docs.google.com/presentation/d/1gFLNJ5rKn0JlKjI47HuYFxF-8RcFhMoiFeevBbLshKI/edit?usp=sharing) where the details of the implementation has been explained in detail, along with the results. [Report](https://docs.google.com/document/d/1QFDVudKToEGptqgMwA42qtisTqjGVqahmVqYoVhf8HA/edit?usp=sharing)

Built and tested using Python3 on Linux.

### Authors:

Saurabh Chand Ramola, Sumukh S
