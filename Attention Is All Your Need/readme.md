# Attention is all you need

## Word Embedding
At the begging of transformer module, we welcome the input embedding layer and output embedding layer. We will start from here to introduce all the module. We all know that
the input and output are in string format. However, string format is not friendly to calculation and storage. The first simple idea is build a dictionary for all the distinct
words in the training dataset. But, this simple way encoding cannot map the *correlation* between each word and *context order* relationship.  

### 1 Correlation
To map the correlation, we need to build a more complex vector for each word. Consider the calculation below.
$$V_E = V_H*W_E$$
Where $V_E$ is the embedded vector,$V_H$ is one-hot row vector, $W_E$ is a matrix. Here we do a linear convertion. Since there is only one element in $V_H$ is 1,$V_E$ is
exactually `i-th row ` of $W_E$.



### 2 positional encoding
![image](https://user-images.githubusercontent.com/89610539/178946797-b285cbfb-a085-4533-973d-5c3b7e7f74a9.png)

From the graph,we can see that positional encoding is added to the input.In this paper ,the author use a general positional encoding method.
$$PE(pos,2i) = sin(\frac{pos}{10000^{2i/dim}})$$
$$PE(pos,2i+1) = cos(\frac{pos}{10000^{2i+1/dim}})$$ 
where pos is the position and i is the dimension. This method can fix the difference between $PE_{pos}$ and $PE_{pos+k}$. In the transformer, we calculate distance in t he context of dot product. We can prove that `PE(pos) · PE(pos+k)` remain stable for any fixed k. So, in other word **Ahthor cares more about relative position rather than absolute position**.

## Encoder

Now, we can step into encoder part. Encoder is consisted of `attention` and `forward net`.

### One-head attention

Multi-Head attention is based on one-head attention. In the one-head attention, we have 3 matrix Q(querry),K(key),V(value) from  
$$Q = XW^Q $$
$$K = XW^K $$ 
$$V = XW^V $$  
$R^{d_m\ *d_m}$
W is  and X is $R^{n*d_m}$,so Q,K,V are $R^{n*d_m}$  
We can see that Q,K,V are from X itself, that why we call it `self-attention`. These three matrix is trainable and will be updated during training. In the self-attention, we will calculate $\tilde{A}$ from  
$$\tilde{A} = \frac{QK^T}{\sqrt{d_m}}$$
where $d_m$ is dimension of embedded word.


