# AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

## Introduction

This paper aims to build a transformer for CV usage. We all know that transformer and attention mechanism work well in NLP field, this paper would like to 'borrow' the exactually
same model from NLP. As it said in the paper, there are many high-efficiency implements in NLP field, so it would be great process if we can copy their idea. Previously, there are
multiple paper illustrate their idea about adding attention in CV tasks. However, all of these models on some level apply inductive bias whereas in this paper, the author want to minimize
this bias.

## Net Structure
<img src="net.jpg" width = "800" height = "400" alt="" align=center />

The whole forward can be divided into three parts:  

1 patching   2 embedding  3 transformer encoding  4  classifying

### 1 Patching

In this paper, author would like to make a hyper-pixel consists of $16\times 16$ pixels.Say that we have a image in $224\times 224$,then we will have$14\times 14$patches. Each patch will go through a FC layer and becomes a vector. Now we have 196($14\times 14$) vectors, each vector has 768($16\times 16\times 3$)dimensions.
So we can take a picture as a text consists of 196 words with 768 dimensions words-embedding.

We can also apply other patching size. Smaller patch size will result in longer vector.So, basically, we can train a model with arbitrary patch size. But one thing is, the pretrained model is patch-size-limited, we will come to that part in section 2.The author tried $14\times 14$ and $16\times 16$. In the following text, we are talking about $16\times 16$ by default.

### 2 embedding

Like what we have in **BERT**, we apply a learnable positional embedding. This learnable embedding is also 1D vector with 768 dimensions. And we will have a fixed vector 0, this is the out put of transformer encoder. For more details, see [BERT](https://arxiv.org/abs/1810.04805). 

We mentioned in section 1 that we may hardly change patch size. That's because this positional embedding is learnable. So once the patch size changed, the dimension of embedded vectors also change which means learned positional embedding in the pretrained model can not work. In the paper, author metioned training a competative VIT needs 2500 days in TPUv3, so there is huge cost if we want to try different patch size.

### Transformer encoding

This part is exactually the same as transformer. As mentioned before, the author wants the model as close as original transformer, so there is **no inductive bias**.

### Classifying

Very typical classifying layer constructed by a MLP layer.

## Discussion
There are some interesting points.
###1 Size of training data
When the data is very large(hundreds of millions). Pure transformer > hybrid model > CNN. When the data is small, hybrid model > CNN >transformer.
In the paper, it says CNN has two inductive bias：
  Locality : neighborhood pixels would share some similarity.
  Translation equivariance : f(g(x))=g(f(x)), assume that `f` is `moving the target in the image`, `g` is `convolution`.
But VIT use very little prior knowledge (for 1D positional embedding, only when we do the patch will we use some spatial prior knowledge). That's why when data is insufficient, Vit is less competative than CNN. 

###2 2D positional embedding
It's nature to think that 2D positional embedding will work better than 1D embedding because image is distributed in 2D dimension. However, the result shows there is very little difference between 1D and 2D positional embedding. And we can see the picture:  
![image](https://user-images.githubusercontent.com/89610539/179605444-3e725929-89af-4f9a-af25-c7260f6672b8.png)
1D dimension positional embedding can help model know where those patches come from!
