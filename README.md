# AS_Cours
### Statistical Learning Course of Master DAC (Données, Apprentissage & Connaissances) in UMPC.
<img src="icon.png" align="right" />
### Summary
* [Handling Torch & Linear Regression](https://github.com/Oulaolay/AS_Cours/blob/master/TP1/TP1_AS.ipynb) - First pratical exercice handling the Torch software and the lua language. Introduction to gradient descent methods (Stochastic gradient descent, Batch gradient descent).                                                           
A summary of article  which gives an intuitive description of descent gradient methods (ie, batch, sgd, adadelta, rmsprop, nag, adam) : [An overview of gradient descent optimisation algorithms](http://sebastianruder.com/optimizing-gradient-descent/).

* [Mini-batch, train/test, Criterion implementation](https://github.com/Oulaolay/AS_Cours/blob/master/TP2/TP2_MINI_BATCH.ipynb) - Using of mini-batch on MNIST dataset. Creation of learning and data test. Huber Loss [Criterions](https://github.com/torch/nn/blob/master/doc/criterion.md) implementation.

* [Non-linearity](https://github.com/Oulaolay/AS_Cours/blob/master/TP3/TP_3.ipynb) - Using XOR problem (Binary artificial dataset of 4 gaussians, two positive labels centered and two negative labels centered). Using Kernel Trick. Creating a neural network by hand. Then, using Sequential module.  

* [Function & Module Implementation](...) - ReQU fonction & linear module without bias implementation.    

* [Graph node, Complicated architectures](https://github.com/Oulaolay/AS_Cours/blob/master/TP5/TP5_Timothee_Poulain.ipynb) - Complex structures implementation with library [nn.graph](http://github.com/torch/nngraph).     

* [Highway Network](https://github.com/Oulaolay/AS_Cours/blob/master/TP6%20%26%207/TP6.ipynb) - coming soon.      
* [Gated Recurrent Unit, RNN](https://github.com/Oulaolay/AS_Cours/blob/master/TP6%20%26%207/TP7.ipynb) - GRU & RNN implementation. 
* [Generative Adversarial Nets](https://github.com/Oulaolay/AS_Cours/blob/master/GAN_Project/Rapport_GAN_Basile_Guerrapin_Timothee_Poulain.pdf) - Project with [Basile Guerrapin](https://github.com/guerrapin), understanding and implementation GAN. (coming soon)               

### Installing Dependencies 
* Install Torch
* Install the nngraph and several packages : 
```
luarocks install nngraph
luarocks install nn
```


### Articles
_ [1] [Generative Adversarial Nets](https://github.com/Oulaolay/AS_Cours/blob/master/TP5/TP5_Timothee_Poulain.ipynb) - Ian Goodfellow, Jean Pouget-Abadie         
_ [2] [Empirical Evaluation of Gated Recurrent Neural Network on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf) - Junyoung Chung, Caglar Gulcehre       
_ [3] [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - Andrej Karpathy      
_ [4] [An introduction to Generative Adversarial Networks (with code in TensorFlow)](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/)                     
_ [5] [The Eyescream Project](http://soumith.ch/eyescream/)                
_ [6] [Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](https://arxiv.org/pdf/1506.05751v1.pdf)        
_ [7] [Generating Faces with Torch](http://torch.ch/blog/2015/11/13/gan.html)


