# CLIP

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020) [[Model Card]](model-card.md) [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.



## Approach

![CLIP](CLIP.png)



## To start
First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.


## Experiment
For this class, we will do zero-shot prediction task done by CLIP's official repo.
Rather than using only a single sentence as input, we will test on various circumstances up to 5 sentences and ensembling results of the 5 different input data.

### Zero-Shot Prediction

The code below performs zero-shot prediction using CLIP, as shown in Appendix B in the paper. This example takes an image from the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), and predicts the most likely labels among the 100 textual labels from the dataset.

```
$ python tests/zero_shot_cifar.py
```

The output will look like the following (the exact numbers may be slightly different depending on the compute device):

```
Processing..... image 9996
0.4963489046714014
Processing..... image 9997
0.4962992598519704
Processing..... image 9998
0.4963496349634963
Processing..... image 9999
0.4963
Overall accuracy:
49.63
```

Note that you need to edit 
```
#pred_index_list.append(pred_index1)
#pred_index_list.append(pred_index2)
pred_index_list.append(pred_index3)
#pred_index_list.append(pred_index4)
#pred_index_list.append(pred_index5)
```
part to control which sentences to be ensembled
