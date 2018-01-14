# http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
# http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

# ------------- 1 ------------- #
# meaning of para. in ()

# para.: input_dim, hidden_dim, num_layers 
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
          
# ------------- 2 ------------- #
# remember to clear out the hidden& cell state of the LSTM
# before each instance

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
