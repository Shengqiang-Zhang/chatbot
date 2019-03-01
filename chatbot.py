# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import csv
import itertools
import os
import random
import re
import unicodedata
from io import open

import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

MAX_LENGTH = 10  # Maximum sentence length to consider
MIN_COUNT = 3  # Minimum word count threshold for trimming


def print_lines(file, n=10):
    with open(file, "rb") as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


# Splits each line of the file into a dictionary of fields
def load_lines(file_name, fields):
    lines = {}
    with open(file_name, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++")
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# Groups fields of lines from 'load_lines' into conversations based on *movie_conversations.txt*
def load_conversations(file_name, lines, fields):
    conversations = []
    with open(file_name, 'r', encoding="iso-8859-1") as f:
        for line in f:
            values = line.split(" +++$+++")
            conv_obj = {}
            for i, field in enumerate(fields):
                conv_obj[field] = values[i]
            line_ids = eval(conv_obj["utteranceIDs"])
            conv_obj["lines"] = []
            for lineid in line_ids:
                conv_obj["lines"].append(lines[lineid])
            conversations.append(conv_obj)
    return conversations


# Extracts pairs of sentences from conversations
def extract_sentence_pairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i + 1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


# Load and trim data
PAD_token = 0
SOS_token = 1
EOS_token = 2


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print(f"Keep_words {len(keep_words)} / {len(self.word2index)} ="
              f" {len(keep_words) / len(self.word2index):.4f}")

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.add_word(word)


def unicode2ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode2ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Read query/response pairs and return a voc object
def read_vocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding="utf-8").read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def load_prepare_data(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data")
    voc, pairs = read_vocs(datafile, corpus_name)
    print(f"Read {len(pairs)} sentence pairs")
    pairs = filter_pairs(pairs)
    print(f"Trimmed to {len(pairs)} sentence pairs")
    print("Counting words...")
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])

    print("Counted words:", voc.num_words)
    return voc, pairs


def trim_rare_words(voc, pairs, min_count=MIN_COUNT):
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = False
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break
        if keep_input and keep_output:
            keep_pairs.append(pair)
    print(f"Trimmed from {len(pairs)} pairs to {len(keep_pairs)} pairs, "
          f"{len(keep_pairs) / len(pairs) : .4f} of total")
    return keep_pairs


# Prepare data for models
def indexes_from_sentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


# Transpose shape(batch_size, max_length) to shape(max_length, batch_size)
# and zero padding.
def zero_padding(l, fill_value=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fill_value))


def binary_matrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Return padded input sequence tensor and lengths
def input_var(l, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths


# Return padded target sequence tensor, padding mask, and max target length
def output_var(l, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    mask = binary_matrix(pad_list)
    mask = torch.ByteTensor(mask)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, mask, max_target_len


# Return all items for a given batch of pairs
def batch2train_data(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_var(input_batch, voc)
    output, mask, max_target_len = output_var(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method not in ["dot", "general", "concat"]:
            raise ValueError(self.method, "is not an appropriate attention method.")
        if self.method == "general":
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == "concat":
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(hidden)
        return torch.sum(energy, encoder_output, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate attention weights (energies) based on the given method.
        if self.method == "dot":
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == "general":
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attn_energies = self.concat_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttenDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttenDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through bidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the GRU outputs
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply calculated attention weights to encoder outputs to get new weighted sum context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq.5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq.6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


# Define Training Procedure
# Masked Loss
def mask_NLLLoss(inp, target, mask):
    n_total = mask.sum()
    cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, n_total.item()


# Single training iteration
def train(input_variable, lengths, target_variable, mask, max_target_len,
          encoder, decoder, embedding, encoder_optimizer, decoder_optimizer,
          teacher_forcing_ratio, batch_size, clip, max_length=MAX_LENGTH):
    # Zero gradients
    encoder_optimizer.zero_gradients()
    decoder_optimizer.zero_gradients()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass entire input batch through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Initialize decoder input as SOS_token
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set decoder initial hidden state to encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we use teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequence through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, n_total = mask_NLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # No teacher forcing: next input is decoder's own current output word
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, n_total = mask_NLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * n_total)
            n_totals += n_total

    # Perform backpropagation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


# Training iterations
def train_iters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                print_every, save_every, clip, corpus_name, load_file_name):
    # Load batches for each iteration
    training_batches = [batch2train_data(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]


if __name__ == '__main__':
    corpus_name = "cornell movie-dialogs corpus"
    corpus = os.path.join("data", corpus_name)
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")
    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict, conversations list, and field ids
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    print("\nProcessing corpus...")
    lines = load_lines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = load_conversations(os.path.join(corpus, "movie_conversations.txt"),
                                       lines, MOVIE_CONVERSATIONS_FIELDS)

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extract_sentence_pairs(conversations):
            writer.writerow(pair)

    # Print a sample of lines
    print("\nSample lines from file:")
    print_lines(datafile)

    # Load and assemble voc and pairs
    save_dir = os.path.join("data", "save")
    voc, pairs = load_prepare_data(corpus, corpus_name, datafile, save_dir)
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)
    pairs = trim_rare_words(voc, pairs, MIN_COUNT)
