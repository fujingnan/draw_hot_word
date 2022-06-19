"""
author: fujingnan
tel: 18813102099
date: 2022-04-18
功能：热词挖掘
"""
from ahocorapy.keywordtree import KeywordTree
from math import *
from collections import OrderedDict


class HotWordMining:
    def __init__(self, duration=7):
        self.kwtree = KeywordTree(case_insensitive=True)
        self.duration = duration

    def compute_word_info(self, words, history_text, current_text):
        """
        为每个词计算历史词频、当前词频、总词频(历史词频+当前词频)、总词频率(当前词频/总词频)
        :param words: 词列表
        :param history_text: 历史文本
        :param current_text: 当前文本
        :return: word_rate_map: Dict: 每个词的历史词频和当前词频,
                 compute_res_for_each_word: Dict: 每个词的总词频和总词频率
                 num_count_avg: Float： 所有词总词频均值: (W1(总词频) + W2(总词频) +...+WN(总词频)) / N
                 rate_count_avg: Flaot: 所有词总词频率均值: (W1(总词频率) + W2(总词频率) +...+WN(总词频率)) / N
        """
        for word in words:
            self.kwtree.add(str(word))
        self.kwtree.finalize()
        his_is_found = self.kwtree.search_all(history_text)
        cur_is_found = self.kwtree.search_all(current_text)
        word_rate_map = dict()
        for w, _ in his_is_found:
            if not w in word_rate_map:
                word_rate_map[w] = {'history_num': 1, 'current_num': 0}
            else:
                word_rate_map[w]['history_num'] += 1
        for w, _ in cur_is_found:
            if not w in word_rate_map:
                word_rate_map[w] = {'history_num': 0, 'current_num': 1}
            else:
                word_rate_map[w]['current_num'] += 1
        compute_res_for_each_word = dict()
        num_count = 0
        rate_count = 0
        for w, tp in word_rate_map.items():
            if not w in compute_res_for_each_word:
                compute_res_for_each_word[w] = dict()
                compute_res_for_each_word[w]['sum'] = tp['history_num'] + tp['current_num']
                compute_res_for_each_word[w]['rate'] = tp['current_num'] / compute_res_for_each_word[w]['sum']
            num_count += compute_res_for_each_word[w]['sum']
            rate_count += compute_res_for_each_word[w]['rate']
        num_count_avg = num_count / (len(words) + 1)
        rate_count_avg = rate_count / (len(words) + 1)
        return (word_rate_map, compute_res_for_each_word, num_count_avg, rate_count_avg)

    def bayes_func(self, cur_sum, cur_rate, num_avg, rate_avg):
        """
        贝叶斯均值计算
        :param cur_sum: 当前总词频
        :param cur_rate: 当前总词频率
        :param num_avg: 所有词总词频均值
        :param rate_avg: 所有词总词频率均值
        :return:
        """
        return (cur_sum * cur_rate + num_avg * rate_avg) / (cur_sum + num_avg + 1)

    def newton_func(self, cur_num, his_num, time_diff):
        """
        牛顿冷却系数计算
        :param cur_num: int: 当前词频
        :param his_num: int: 历史词频
        :param time_diff: int: 时间差
        :return:
        """
        return log((cur_num + 1) / (his_num + 1), e) / time_diff

    def bn_hot(self, words, history_text, current_text):
        """
        词热度值计算: Hot = 0.7 * bayes_func + 0.3 * newton_func
        :param words: 词列表
        :param history_text: 历史文本
        :param current_text: 当前文本
        :return: Dict: {
            word: hot_value
        }
        已按热度值降序排序
        """
        bn_word_hot_map = dict()
        word_rate_map, compute_res_for_each_word, num_count_avg, rate_count_avg = self.compute_word_info(words,
                                                                                                         history_text,
                                                                                                         current_text)
        for word in words:
            bn_word_hot_map[word] = 0.7 * self.bayes_func(compute_res_for_each_word[word]['sum'],
                                                          compute_res_for_each_word[word]['rate'],
                                                          num_count_avg,
                                                          rate_count_avg) + \
                                    0.3 * self.newton_func(word_rate_map[word]['current_num'],
                                                           word_rate_map[word]['history_num'], self.duration - 1)
        sortDict = sorted(bn_word_hot_map.items(), key=lambda x:x[1], reverse=True)
        return OrderedDict(sortDict)


if __name__ == '__main__':
    pass
