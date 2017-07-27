# http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
# http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# Pytorch’s LSTM expects all of its inputs to be 3D tensors: sequence_len, mini-batch, input_dim

lstm = nn.LSTM(3, 3, 2)  # Input dim is 3, hidden dim is 3, # recurrent layers is 2
inputs = [autograd.Variable(torch.randn((1, 3)))
          for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state. 
# hidden = (hidden_n, cell_n) # state at time n
hidden = (autograd.Variable(torch.randn(1, 1, 3)),
          autograd.Variable(torch.randn((1, 1, 3))))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    # out (seq_len, batch, hidden_size)
    out, hidden = lstm(i.view(1, 1, -1), hidden)
