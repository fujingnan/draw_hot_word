from collections import Counter, OrderedDict
import math
from LAC import LAC
import os
from TextRank4ZH.textrank4zh import *
import warnings
warnings.filterwarnings("ignore")
lac = LAC(mode='lac')




class TF_IDF:
    def __init__(self, tag_filter, user_dict='', is_lower=True, print_num=20, word_min_len=2, window=2):
        self.tag_filter = tag_filter
        self.user_dict = user_dict
        if os.path.join(self.user_dict):
            lac.load_customization(self.user_dict)
        self.tr4w = TextRank4Keyword(allow_speech_tags=self.tag_filter, user_dict=user_dict)
        self.is_lower = is_lower
        self.print_num = print_num
        self.word_min_len = word_min_len
        self.window = window

    def text_cut(self, texts):
        text_stack = []
        for text in texts:
            cut_res = lac.run(text)
            cut_res = [cut_res[0][i] for i in range(len(cut_res[1])) if cut_res[1][i] in self.tag_filter and len(cut_res[0][i]) >= 2]
            text_stack.append(cut_res)
        return text_stack

    def stem_count(self, word_set):
        count = Counter(word_set)  # 实现计数功能
        return count

    # 定义TF-IDF的计算过程
    def D_con(self, word, count_list):
        D_con = 0
        for count in count_list:
            if word in count:
                D_con += 1
        return D_con

    def tf(self, word, count):
        return count[word] / sum(count.values())

    def idf(self, word, count_list):
        return math.log(len(count_list)) / (1 + self.D_con(word, count_list))

    def tfidf(self, word, count, count_list):
        return self.tf(word, count) * self.idf(word, count_list)

    def word_clean(self, texts):
        text_set = self.text_cut(texts)
        count_list = []
        tf_idf = {}
        for c in text_set:
            count_list.append(self.stem_count(c))
        for i in range(len(count_list)):
            tempwords = []
            tempweights = []
            for word in count_list[i]:
                if not word:
                    continue
                tempwords.append(word)
                weight = self.tfidf(word, count_list[i], count_list)
                if weight == 0.:
                    self.tr4w.analyze(text=texts[i], lower=self.is_lower, window=self.window)
                    for item in self.tr4w.get_keywords(self.print_num, word_min_len=self.word_min_len):
                        if not item.word == word:
                            continue
                        tempweights.append(item.weight)
                        break
                else:
                    tempweights.append(weight)
            tf_idf['word'] = tempwords
            tf_idf['weights'] = tempweights
        return tf_idf


if __name__ == '__main__':
    pass
