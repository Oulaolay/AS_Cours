 require 'nn'
 require 'gnuplot'
 require 'svm'
 require 'optim'
 require 'nngraph'


-- Test avec mnist (en cours)
local mnist = require 'mnist'

train_file = 'mnist.t7/train_32x32.t7'
test_file = 'mnist.t7/test_32x32.t7'

trainData = torch.load(train_file,'ascii')
testData = torch.load(test_file,'ascii')



trainset = {
    size = 50000,
    data = fullset.data[{{1,50000}}]:double(),
    label = fullset.label[{{1,50000}}]
}
