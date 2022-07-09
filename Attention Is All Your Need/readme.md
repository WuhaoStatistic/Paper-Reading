## Word Embedding

At the begging of transformer module, we welcome the input embedding layer and output embedding layer. We will start from here to introduce all the module. We all know that
the input and output are in string format. However, string format is not friendly to calculation and storage. The first simple idea is build a dictionary for all the distinct
words in the training dataset. But, this simple way encoding cannot map the *correlation* between each word and *context order* relationship.

### 1 Correlation
To map the correlation, we need to build a more complex vector for each word. Consider the calculation below.
$$V_E = V_H*W_E$$
Where $V_E$ is the embedded vector,$V_H$ is one-hot row vector, $W_E$ is a matrix. Here we do a linear convertion. Since there is only one element in $V_H$ is 1,$V_E$ is
exactually `i-th row ` of $W_E$.
