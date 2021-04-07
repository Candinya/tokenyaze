from training import word_to_ix
from data_process import prepare_sequence
from config import MODEL_FILE
import torch

net = torch.load(MODEL_FILE)
net.eval()


def tknyz(stri):
    precheck_sent = prepare_sequence(stri, word_to_ix)

    label = net(precheck_sent)[1]

    cws = []
    for i in range(len(label)):
        cws.extend(stri[i])
        if label[i] == 2 or label == 3:
            cws.append('/')

    return ''.join(cws)


if __name__ == "__main__":
    while True:
        print('分词结果：', tknyz(input('输入未分词语句：')))
