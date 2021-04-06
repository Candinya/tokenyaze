from data_process import read_file, tag_to_ix
from config import *
from BiLSTM_CRF_PARALLEL import *
# from BiLSTM_CRF import *
import torch
from torch import optim
import os.path

print('loading file ' + filename + ' ...')
_, content, label = read_file(filename)
print('file loaded.')


def train_data(content, label):
    train_data = []
    for i in range(len(label)):
        train_data.append((content[i], label[i]))
    return train_data


data = train_data(content, label)

print('start word embedding...')
word_to_ix = {}
word_to_ix['<PAD>'] = 0
for sentence, tags in data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)  # 单词映射，字到序号

if __name__ == "__main__":
    if os.path.isfile('tknyz.model'):
        print('model exists, continue')
        model = torch.load('tknyz.model').cuda()
    else:
        print('model doesnot exist, creating new')
        model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print('start training')

    # 训练
    for epoch in range(epochs):
        model.zero_grad()

        # sentence_in = prepare_sequence(sentence, word_to_ix)
        # targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long).cuda()
        sentence_in_pad, targets_pad = prepare_sequence_batch(data, word_to_ix, tag_to_ix)
        # loss = model.neg_log_likelihood(sentence_in, targets)
        loss = model.neg_log_likelihood_parallel(sentence_in_pad, targets_pad)

        loss.backward()
        optimizer.step()

        torch.save(model, 'tknyz.model')
        torch.save(model.state_dict(), 'tknyz_all.model')
        print('epoch/epochs: {}/{}, loss:{:.6f}'.format(epoch + 1, epochs, loss.item()))

    # 保存模型
    torch.save(model, 'tknyz.model')
    torch.save(model.state_dict(), 'tknyz_all.model')

    print('training finished')
