from training import word_to_ix
from data_process import prepare_sequence
import torch

net = torch.load('tknyz.model')
net.eval()
stri = "改善人民生活水平，建设社会主义政治经济。"
precheck_sent = prepare_sequence(stri, word_to_ix)
# precheck_sent= tensor([ 45, 102,  23,  24,  80,  98, 140, 141,  17,  32,  33,  37,  38,  39,  40, 103, 104,  60,  61,  12])

label = net(precheck_sent)[1]
# net(precheck_sent)= (tensor(32.3123, grad_fn=<SelectBackward>), [0, 2, 0, 2, 0, 2, 0, 2, 3, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2])

cws = []
for i in range(len(label)):
    cws.extend(stri[i])
    if label[i] == 2 or label == 3:
        cws.append('/')
# cws= ['改', '善', '/', '人', '民', '/', '生', '活', '/', '水', '平', '/', '，', '建', '设', '/', '社', '会', '/', '主', '义', '/', '政', '治', '/', '经', '济', '/', '。']

print('输入未分词语句：', stri)
print('分词结果：', ''.join(cws))
