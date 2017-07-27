# http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
# http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

# input_dim, hidden_dim, num_layers = (3, 3, 2)
lstm = nn.LSTM(3, 3, 2)  

# inputs is 3D tensors: sequence_len, minibatch_size, input_dim
inputs = [autograd.Variable(torch.randn((1, 3))) 
          for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state. 
# hidden = (hidden_n, cell_n) # state at time n
# state is 3D tensors: num_layers, minibatch_size, hidden_dim
hidden = (autograd.Variable(torch.randn(1, 1, 3)),
          autograd.Variable(torch.randn((1, 1, 3))))

for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    # out (seq_len, batch, hidden_size)
    out, hidden = lstm(i.view(1, 1, -1), hidden)
