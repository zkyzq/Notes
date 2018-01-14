# learning notes from -- http://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html
# stuck by the following code
'''
class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)
    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec))

NUM_LABELS, VOCAB_SIZE = 2, 10
model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)
log_probs = model(autograd.Variable(bow_vector))  ########### how to pass the parameter?
'''
#answers are as follows:

#----------------   1    ----------------#
# __init__ and __call__ in python

#1.1 __init__ in python
class foo:
    def __init__(self, a, b, c):
        # ...
x = foo(1, 2, 3) # __init__

#1.2 __call__ in python
class foo:
    def __call__(self, a, b, c):
        # ...
x = foo()
x(1, 2, 3) # __call__

#----------------   2    ----------------#
# Python中函数的参数定义和可变参数: http://blog.csdn.net/feisan/article/details/1729905
# func(*args, **kwargs)

#2.1 带*的参数就是用来接受可变数量个参数
def funcD(a, b, *c):
  print a
  print b
  print c
'''调用funcD(1, 2, 3, 4, 5, 6)结果是
1
2
(3, 4, 5, 6)'''

#2.2 带**的参数就是用来接受一个字典
def funcF(a, **b):
  print 
  for x in b:
    print x + ": " + str(b[x])
'''调用funcF(100, c='你好', b=200)，执行结果
100
c: 你好
b: 200'''

#----------------   3    ----------------#
# the following codes are the basic nn.Module in pytorch
# a new nn.Module will inherit the basic nn.Module in pytorch
# the new nn.Module should override the func. forward() 
# the new nn.Module will inherit the func. __call__(), so when we call
# the new nn.Module, we can directly pass parameters of input to func. forward()
def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)

def forward(self, *input):
        """Defines the computation performed at every call.

        Should be overriden by all subclasses.
        """
        raise NotImplementedError
        
#----------------   4    ----------------#
# http://pytorch.org/docs/master/_modules/torch/nn/modules/module.html#Module
# def __call__(self, *input, **kwargs): 
#   result = self.forward(*input, **kwargs)
# so in a func. forward() of nn.Module, we must return the final results 
# as the last line of code
# or we cannot compute gradients and report errors
class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)
    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec))
