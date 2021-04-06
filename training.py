from data_process import read_file, tag_to_ix
from config import *
from BiLSTM_CRF_PARALLEL import *
# from BiLSTM_CRF import *
import torch
from torch import optim
import os.path
import time


def print_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


print('loading file {} ... @ {}'.format(filename, print_time()))
_, content, label = read_file(filename)
print('file loaded. @ {}'.format(print_time()))


def train_data(content, label):
    train_data = []
    for i in range(len(label)):
        train_data.append((content[i], label[i]))
    return train_data


data = train_data(content, label)

print('start word embedding... @ {}'.format(print_time()))
word_to_ix = {}
word_to_ix['<PAD>'] = 0
for sentence, tags in data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)  # 单词映射，字到序号

if __name__ == "__main__":
    if os.path.isfile('model/tknyz.model'):
        print('model exists, continue @ {}'.format(print_time()))
        model = torch.load('model/tknyz.model')
    else:
        print('model doesnot exist, creating new @ {}'.format(print_time()))
        model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print('start training @ {}'.format(print_time()))

    # 训练
    for epoch in range(epochs):
        model.zero_grad()

        sentence_in_pad, targets_pad = prepare_sequence_batch(data, word_to_ix, tag_to_ix)
        loss = model.neg_log_likelihood_parallel(sentence_in_pad, targets_pad)

        loss.backward()
        optimizer.step()

        # 保存模型
        torch.save(model, 'model/tknyz.model')
        torch.save(model.state_dict(), 'model/tknyz_all.model')
        print('epoch: {}/{}, loss:{:.6f} @ {}'.format(epoch + 1, epochs, loss.item(), print_time()))

    print('training finished @ {}'.format(print_time()))
