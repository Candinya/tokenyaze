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


print('loading file {} ... @ {}'.format(MATERIAL_FILE, print_time()))
_, content, label = read_file(MATERIAL_FILE)
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
    if os.path.isfile(MODEL_FILE):
        print('model exists, continue @ {}'.format(print_time()))
        model = torch.load(MODEL_FILE)
    else:
        print('model doesnot exist, creating new @ {}'.format(print_time()))
        model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print('start training @ {}'.format(print_time()))

    # 训练
    for epoch in range(TRAIN_EPOCHS):
        print('start epoch {} @ {}'.format(epoch + 1, print_time()))

        model.zero_grad()

        sentence_in_pad, targets_pad = prepare_sequence_batch(data, word_to_ix, tag_to_ix)
        loss = model.neg_log_likelihood_parallel(sentence_in_pad, targets_pad)

        print('start optimize @ {}'.format(print_time()))
        loss.backward()
        optimizer.step()

        # 保存模型
        print('saving model... @ {}'.format(print_time()))
        torch.save(model, MODEL_FILE)
        torch.save(model.state_dict(), MODEL_DICT)
        print('epoch: {}/{}, loss:{:.6f} @ {}'.format(epoch + 1, TRAIN_EPOCHS, loss.item(), print_time()))

    print('training finished @ {}'.format(print_time()))
