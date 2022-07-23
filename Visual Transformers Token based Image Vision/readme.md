# Visual Transformers

## Introduction

The authors want a module that can be more efficient than CNN. They achieve this by **VT** using spatial attention to get high-level concepts in an image. This high-level concepts
can be used in image-level tasks(classigfication) and pixel-level tasks(segmentation).

## Structure
![image](https://user-images.githubusercontent.com/89610539/180595011-cbcde9b6-4ed0-40e6-ac6d-5a6c69618c2c.png)

Although there are many arrows and lines in the picture above, the main structure is very simple. The picture first go through a convolutional net to a feature map. Then, a tokenizer will transfer feature map into tokens. A transformer will be applied on tokens to get output-tokens. We can use this tokens with MLP to do image classification or project tokens to feature map to do segmentation work.



