## Introduction  
It is always hard for CNN/DNN to do sequence to sequence modelling. Because in CNN network, it requires input and output has fixed length. In the machine translation,
the both the length of input and output may differ a lot. So, RNN is used to solve this problems. When it comes to RNN, we find that it's extremely difficult for RNN
to capture long dependencies. So, LSTM is finally used here.

## The model
In this model, there are two LSTMs. According to author's idea:  one LSTM works as encoder and another one works as decoder. During the training, we use the hidden state
of last LSTM layer in encoder plus target source to train decoder. The work flow is like below.  

$\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ $![image](https://user-images.githubusercontent.com/89610539/177779292-74de2ca3-868c-498f-bc8c-0af0c5cbbacb.png)

The white blocks are LSTM layers in encoder whereas the blue blocks are belong to decoder. During the training, **teacher focing** is applied. That is, we use the correct
translation to train the decoder other than the hidden state of last layer. Of course, iduring the test stage, we have to use hidden stage from last layer to generate new 
words.The Author also mention one thing interesting, a great progress can be seen if we **reverse** the order of the words. They believe it will **help SGD to "establish communication" between input and output**.

seq2seq model is the fundation of transformer, so it's worth to see how it being implemented in pytorch style.
## Encoder
```
class Seq2SeqEncoder(nn.Module):
    """Encoder"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        
        # Embedding layer, we can also get embedding from other word2vec modules.
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,dropout=dropout)

    def forward(self, X, *args):
        # the shape of 'X'：(batch_size,num_steps,embed_size)
        # num_steps is length of sentence within machine translation context.
        X = self.embedding(X)
        
        # We need to permute the shape to make the first axis be time step.That's the requirement
        # of GRU layer.
        X = X.permute(1, 0, 2)
        
        # by default ,the state of RNN is 0.
        output, state = self.rnn(X)
        
        # shape of output:(num_steps,batch_size,num_hiddens)
        # shape of state[0]:(num_layers,batch_size,num_hiddens)
        return output, state
```
We can use code above in this way
```
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
```
## Decoder
```
class Seq2SeqDecoder(nn.Module):
    """Decoder"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # we will concatenate hidden state and embeded vector together
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        # we use dense to do softmax 
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # shape of X：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        
        # broadcast context，make it has the same num_steps as X,noticed that the axis[0] is num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        
        # concatenate as we mentioned in the __init__
        X_and_context = torch.cat((X, context), 2)
        
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output:(batch_size,num_steps,vocab_size)
        # state[0]:(num_layers,batch_size,num_hiddens)
        return output, state
```
We can use code in this way
```
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()

# The initial state of decoder is the last hidden state of encoder.
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
```

