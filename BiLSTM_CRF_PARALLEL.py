import torch
from data_process import START_TAG, STOP_TAG
from torch import nn


def argmax(vec):  # 返回每一行最大值的索引
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):  # seq是字序列，to_ix是字和序号的字典
    idxs = [to_ix[w] for w in seq]  # idxs是字序列对应的向量
    return torch.tensor(idxs, dtype=torch.long).cuda()


def prepare_sequence_batch(data, word_to_ix, tag_to_ix):
    seqs = [i[0] for i in data]
    tags = [i[1] for i in data]
    max_len = max([len(seq) for seq in seqs])
    seqs_pad = []
    tags_pad = []
    for seq, tag in zip(seqs, tags):
        seq_pad = seq + ['<PAD>'] * (max_len - len(seq))
        tag_pad = tag + ['<PAD>'] * (max_len - len(tag))
        seqs_pad.append(seq_pad)
        tags_pad.append(tag_pad)
    idxs_pad = torch.tensor([[word_to_ix[w] for w in seq] for seq in seqs_pad], dtype=torch.long).cuda()
    tags_pad = torch.tensor([[tag_to_ix[t] for t in tag] for tag in tags_pad], dtype=torch.long).cuda()
    return idxs_pad, tags_pad


# LSE函数，模型中经常用到的一种路径运算的实现
def log_sum_exp(vec):  # vec.shape=[1, target_size]
    max_score = vec[0, argmax(vec)]  # 每一行的最大值
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_add(args):
    return torch.log(torch.sum(torch.exp(args)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim).cuda()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True).cuda()

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size).cuda()

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)).cuda()  # 随机初始化转移矩阵

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000  # tag_to_ix[START_TAG]: 3（第三行，即其他状态到START_TAG的概率）
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000  # tag_to_ix[STOP_TAG]: 4（第四列，即STOP_TAG到其他状态的概率）
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).cuda(),
                torch.randn(2, 1, self.hidden_dim // 2).cuda())  # 初始隐状态概率，第1个字是O1的实体标记是qi的概率

    # 所有路径的得分，CRF的分母
    def _forward_alg_new_parallel(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([feats.shape[0], self.tagset_size], -10000.).cuda()  # 初始隐状态概率，第1个字是O1的实体标记是qi的概率
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list = [init_alphas]  # 初始状态的forward_var，随着step t变化
        for feat_index in range(feats.shape[1]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            # gamar_r_l = torch.transpose(gamar_r_l,0,1)
            t_r1_k = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)  # +1
            # t_r1_k = feats[:,feat_index,:].repeat(feats.shape[0],1,1).transpose(1, 2)
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)
            # forward_var_list.append(log_add(aa))
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[
            self.tag_to_ix[STOP_TAG]].repeat([feats.shape[0], 1])
        # terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).unsqueeze(dim=0)
        #embeds = self.word_embeds(sentence).view(len(sentence), 1, -1).transpose(0,1)
        lstm_out, self.hidden = self.lstm(embeds)
        #lstm_out = lstm_out.view(embeds.shape[1], self.hidden_dim)
        lstm_out = lstm_out.squeeze()
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # 得到feats，维度=len(sentence)*tagset_size，表示句子中每个词是分别为target_size个tag的概率
    def _get_lstm_features_parallel(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # 正确路径的分数，CRF的分子
    def _score_sentence_parallel(self, feats, tags):
        # Gives the score of provided tag sequences
        # feats = feats.transpose(0,1)

        score = torch.zeros(tags.shape[0]).cuda()
        tags = torch.cat([torch.full([tags.shape[0], 1], self.tag_to_ix[START_TAG], dtype=torch.long).cuda(), tags], dim=1)
        for i in range(feats.shape[1]):
            feat = feats[:, i, :]
            score = score + \
                    self.transitions[tags[:, i + 1], tags[:, i]] + feat[range(feat.shape[0]), tags[:, i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:, -1]]
        return score

    # 解码，得到预测序列的得分，以及预测的序列
    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).cuda()
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = []
        forward_var_list.append(init_vvars)

        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            gamar_r_l = torch.squeeze(gamar_r_l)
            next_tag_var = gamar_r_l + self.transitions
            # bptrs_t=torch.argmax(next_tag_var,dim=0)
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t, 0) + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        # 翻译到结束标签
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]  # 其他标签（B,I,E,Start,End）到标签next_tag的概率
        best_tag_id = torch.argmax(terminal_var).tolist()  # 选择概率最大的一条的序号
        path_score = terminal_var[0][best_tag_id]

        # 从后向前走，找到一个best路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 弹出起始标签
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # 安全性检查
        best_path.reverse()  # 把从后向前的路径倒置
        return path_score, best_path

    # 求负对数似然，作为loss
    def neg_log_likelihood_parallel(self, sentences, tags):
        feats = self._get_lstm_features_parallel(sentences)  # emission score
        forward_score = self._forward_alg_new_parallel(feats)  # 所有路径的分数和，即b
        gold_score = self._score_sentence_parallel(feats, tags)  # 正确路径的分数，即a
        return torch.sum(forward_score - gold_score)  # 注意取负号 -log(a/b) = -[log(a) - log(b)] = log(b) - log(a)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
