# Attention is all you need
First, we will go through the paper and structure. Then there will be some understanding from my perspectve. Some notes will be marked **[exp_n]** 
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
$$Q = W_QX $$
$$K = W_KX $$ 
$$V = W_VX $$  
As for the shape,note that X is $n\times d_m$ ,**Q,K** are $n\times d_K$ and **V** is $n\times d_v$. n is the input length and $d_m$ is dimension of embedding.  
We can see that Q,K,V are just linear calculation of X, that why we call it `self-attention`. These three matrix is trainable and will be updated during training. In the self-attention, we will calculate $\tilde{A}$ from  
$$\tilde{A} = \frac{QK^T}{\sqrt{d_m}}$$
Apprently, the shape of $\tilde{A}$ is $n\times n$. Then we do softmax for each row in A,that is(in python sytle)  
$$\tilde{A_{i,:}}\  =\ softmax(\tilde{A_{i,:}})$$  
Then we use V to extract information from A. **[exp1]**
$$Attention(Q,K,V) = \tilde{A}V$$

One may find some latent problem here. When we do padding, normally we will add one in the X matrix. And all the calculation we did above are liner calculation without bias which will keep padding value 0 to $\tilde{A}$. And we all know that we will do exponantial calculation in softmax layer and padding value will become 1 to each word.
That's a huge disaster. To solve this problem, we can **inplace padding value 0  to $10^{-9}$**

Another interesting thing is denominator $\sqrt{d_m}$. If there is no such denominator, the product will grow up rapidly with multiple attention blocks. Finally, the value after softmax will extremely close to 1 and we will suffer gradient vanishing there.In the paper, author call this **scaled-dot-product**.

### Multi-head attention
In the paper, author set $d_k=d_v=\frac{d_m}{h}$. One thing different from one-head attention is that the dimention of Q,K,V are $n_\times d_m$.  **[exp2]**
$$Q_i = QW_i^Q,\ \ \ \ \ Q_i\ is\ R^{n\times d_k}$$
$$K_i = KW_i^K,\ \ \ \ \ K_i\ is\ R^{n\times d_k}$$
$$V_i = VW_i^V,\ \ \ \ \ V_i\ is\ R^{n\times d_v}$$
And now,   
$$\tilde{A_i} = \frac{Q_iK_i^T}{\sqrt{d_k}}$$
$$head_i = softmax(\tilde{A_i})V_i$$
We can calculate that $head_i$ is in the shape of $n\times d_v$
And then, like explination in the paper, we concatenate them together,So **[exp3]**.
$$H = [H_1,H_2,H_3....H_h]\in R^{n'times d_v \dot h}$$
