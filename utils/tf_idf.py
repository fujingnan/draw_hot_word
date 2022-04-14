from collections import Counter, OrderedDict
import math
import jieba


# class Init_Jieba:
#     def __init__(self, cut_ignore):
#         self.cut_ignore = cut_ignore
#         for word in self.cut_ignore:
#             jieba.add_word(word)

class TF_IDF:
    def __init__(self, cut_ignore, tag_filter):
        self.tag_filter = tag_filter
        # Init_Jieba(cut_ignore)
        self.cut_ignore = cut_ignore

    def text_cut(self, texts):
        # for word in self.cut_ignore:
        #     jieba.add_word(word)
        # jieba.add_word('远海')
        text_stack = []
        for text in texts:
            cut_res = jieba.cut(text)
            # cut_res = [x.word for x in cut_res if x.flag in self.tag_filter]
            cut_res = [x for x in cut_res]
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
        tfidf_res = []
        word_tfidf_map = {}
        for c in text_set:
            count_list.append(self.stem_count(c))
        for i in range(len(count_list)):
            tf_idf = {}
            for word in count_list[i]:
                if not word in set(self.cut_ignore):
                    continue
                tf_idf[word] = self.tfidf(word, count_list[i], count_list)
            tfidf_res.append(tf_idf)
        for res in tfidf_res:
            for key, value in res.items():
                word_tfidf_map.setdefault(key, []).append(value)
        sorts = {}
        for w in self.cut_ignore:
            sorts[w] = max(word_tfidf_map[w])
        return OrderedDict(sorted(sorts, key=lambda x: x[1], reverse=True))


if __name__ == '__main__':
    pass
