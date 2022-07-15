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
#### version 1 in the paper
In the paper, author set $d_k=d_v=\frac{d_m}{h}$. One thing different from one-head attention is that the dimention of Q,K,V are $n_\times d_m$.  **[exp2]**
$$Q_i = QW_i^Q,\ \ \ \ \ Q_i\ \in \ R^{n\times d_k}$$
$$K_i = KW_i^K,\ \ \ \ \ K_i\ \in \ R^{n\times d_k}$$
$$V_i = VW_i^V,\ \ \ \ \ V_i\ \in \ R^{n\times d_v}$$
And now,   
$$\tilde{A_i} = \frac{Q_iK_i^T}{\sqrt{d_k}}$$
$$head_i = softmax(\tilde{A_i})V_i$$
We can calculate that $head_i$ is in the shape of $n\times d_v$
And then, like explination in the paper, we concatenate them together,So **[exp3]**.
$$H = [H_1,H_2,H_3....H_h]\in R^{n\times (d_v\  \dot\  h)}$$
Since in some case $d_m$ may not be divisible by h, so $d_v\  \dot \ h$ may not equal to $d_m$. So, we apply another fully connected layer here:
$$H^{[1]} = HW^0,\ W^0 \in \ (d_v\ \dot\ h)\times \ \ d_m\ \ and\ \ H^{[1]}\ \in \ n\times d_m$$  
Author put this graph for readers to better understand how multi-head works. The mask here is optional. It mainly depends on whether there is padding. If padding is applied in multi-head, we can use mask to cancel those part. The modern frames like pytorch offer this API to build masked tensor.
![image](https://user-images.githubusercontent.com/89610539/179035794-daad1cfc-639d-49c4-809d-f11aa4fc5346.png)

#### version 2 in practice

The same as previous one, we will have multiple head and $d_k=d_v=\frac{d_m}{h}$. The different thing happens when we start to build $Q_i,K_i\ \ and V_i$.Instead of applying a liner calculation(or projection), we simply devided **Q** into **h** part by its column,in python style,that is:
$$Q_i = Q[1:n,i\dot\ d_k\ :(i+1)d_k]$$
It is totally same for **K** and **V**. This implementation will reduce a lot of calculation.After that, we use the same method to get $H_i$ and concatenate them together, then feed **H** into a fully connected net to get output.

### Res connection and Norm
Res connection is quite simple. Since we have $H^{[1]}$ in the same shape of X, we can just add them together:
$$X^{[2]} =H^{[1]}\ +\  X \ \in \ R^{n\times\ d_m}$$ 

When it comes to normolization, batch normolization is not a good choice. Because our model is dynamic model (**n** will change during the training). In this case, if there is padding in the word, the number of 0 will have strong impact on batch normolization. Layer normolization can help solve this problem. When applying layer normolization, we are normolizing each word.

### Feed Forward
Here is just anothor DNN net which project $X^{[2]}$ to its own space($R^{n\times \ d_m}$).After that, we apply the same res connection and Norm and get output $X^{[3]}$.

Now, we have $X^{[3]}\ \in R^{n'times \ d_m}$ in the same shape as input $X$. We have **N** attention blocks, so we can set X^{[3]} as input to repeat this process....  
That's how encoder part works.  

## Decoder
The same as models in seq2seq, the input of decoder is different between training and testing stage. In the training stage, we apply `teacher force` feeding the decoder the correct target output. The embedding part and positional part are the same as encoder.
`tips: If there is any question about how to train a decoder, please see seq2seq model first.`

