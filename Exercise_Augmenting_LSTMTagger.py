import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
word2ix, char2ix = {}, {}
for sent, tags in training_data:
    for word in sent:
        if word not in word2ix:
            word2ix[word] = len(word2ix)

char_set = set(list(''.join(word2ix.keys())))
char2ix = {char: i for i, char in enumerate(char_set)}

EMBEDDING_DIM_CHAR, EMBEDDING_DIM_WORD = 3, 5
HIDDEN_DIM_CHAR, HIDDEN_DIM_WORD = 3, 8

# Augmenting the LSTM part-of-speech tagger with
# word and character-level features
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim_char, embedding_dim_word, hidden_dim_char, 
    	hidden_dim_word, char_num, word_num, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim_char = hidden_dim_char
        self.hidden_dim_word = hidden_dim_word
        
        self.char_embeddings = nn.Embedding(char_num, embedding_dim_char)
        self.word_embeddings = nn.Embedding(word_num, embedding_dim_word)

        # The LSTM takes char embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim. 
        # The final hidden state is the character level representation of a word
        self.lstm_char = nn.LSTM(embedding_dim_char, hidden_dim_char)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm_word = nn.LSTM(embedding_dim_char+embedding_dim_word, hidden_dim_word)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim_word, tagset_size)
        self.hidden_char = self.init_hidden_char()
        self.hidden_word = self.init_hidden_word()

    def init_hidden_char(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim_char)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim_char)))

    def init_hidden_word(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim_word)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim_word)))

    def char_level_feat(self, char):
    	embeds_char = self.char_embeddings(char)
    	lstm_out, self.hidden_char = self.lstm_char(
            embeds_char.view(len(char), 1, -1), self.hidden_char)
    	char_level_embeds = self.hidden_char[0]
    	return char_level_embeds.view(1,-1)  

    def forward(self, input):
    	sentence, char_in = input
    	# word_embeddings at char level 
    	char_feat = [self.char_level_feat(char) for char in char_in]
    	embeds_char_level = torch.cat(char_feat, 0)
    	embeds_word_level = self.word_embeddings(sentence)
    	embeds = torch.cat([embeds_word_level, embeds_char_level], 1)
    	lstm_out, self.hidden_word = self.lstm_word(
    		embeds.view(len(sentence), 1, -1), self.hidden_word)          
    	tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
    	tag_scores = F.log_softmax(tag_space)
    	return tag_scores

model = LSTMTagger(EMBEDDING_DIM_CHAR, EMBEDDING_DIM_WORD, HIDDEN_DIM_CHAR, 
		HIDDEN_DIM_WORD, len(char2ix), len(word2ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden_word = model.init_hidden_word()
        model.hidden_char = model.init_hidden_char()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = prepare_sequence(sentence, word2ix)
        char_in = [prepare_sequence(list(word1), char2ix) for word1 in sentence]
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model((sentence_in, char_in))

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
