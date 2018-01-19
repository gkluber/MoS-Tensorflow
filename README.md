# MoS-Tensorflow
Tensorflow implementation of the mixture of softmaxes algorithm described in the paper <a href="https://arxiv.org/abs/1711.03953">Breaking the Softmax Bottleneck: A High-Rank RNN Language Model (Yang et al., 2017)</a>.<br>
See https://github.com/zihangdai/mos for an implementation using PyTorch. <br>
#### Why does mixture of softmaxes matter?
In natural language processing, the extent to which the true probability distribution of appropriate responses can be approximated overall by the network depends on the ability to express probabilties.<br>
The problem with using the softmax function is that, when applied to the logits or raw outputs of a neural network, a substantial amount of information is lost. <br>
This loss of information, signified by the low-rank of a resultant matrix one constructs from the logits, encourages the network to fit generic responses to each input. <br>
Ideally, the rank of the matrix should be high, which entails more expressiveness and allows the network to use more information in its generation of responses and its analysis. Thus, this is what the mixture of softmaxes network accomplishes. <br>

### Code incomplete and heavily under construction. <br>
