from data_process import read_file, tag_to_ix
from config import *
from BiLSTM_CRF_PARALLEL import *
# from BiLSTM_CRF import *
import torch
from torch import optim
import os.path
import time


def print_log(log):
    print('[{}]{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), log))


print_log('loading file {} ...'.format(MATERIAL_FILE))
_, content, label = read_file(MATERIAL_FILE)
print_log('file loaded.')


def train_data(content, label):
    train_data = []
    for i in range(len(label)):
        train_data.append((content[i], label[i]))
    return train_data


data = train_data(content, label)

print_log('start word embedding...')
word_to_ix = {'<PAD>': 0}
for sentence, tags in data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)  # 单词映射，字到序号

if __name__ == "__main__":
    if os.path.isfile(MODEL_FILE):
        print_log('model exists, continue')
        model = torch.load(MODEL_FILE)
    else:
        print_log('model doesnot exist, creating new')
        model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print_log('start training')

    # 训练
    for epoch in range(TRAIN_EPOCHS):
        print_log('start epoch {}'.format(epoch + 1))

        model.zero_grad()

        print_log('start loading nn')
        sentence_in_pad, targets_pad = prepare_sequence_batch(data, word_to_ix, tag_to_ix)
        loss = model.neg_log_likelihood_parallel(sentence_in_pad, targets_pad)

        print_log('start gradient descent')
        loss.backward()
        print_log('start optimize')
        optimizer.step()

        # 保存模型
        print_log('saving model...')
        torch.save(model, MODEL_FILE)
        torch.save(model.state_dict(), MODEL_DICT)
        print_log('epoch: {}/{}, loss:{:.6f}'.format(epoch + 1, TRAIN_EPOCHS, loss.item()))

    print_log('training finished')
