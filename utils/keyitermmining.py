from tqdm import tqdm
from TextRank4ZH.textrank4zh import *
from utils.tf_idf import TF_IDF
import warnings
warnings.filterwarnings("ignore")


class KeyItermMining(TF_IDF):
    def __init__(self, is_lower=True, window=2, print_num=20, word_min_len=2, min_phrase_len=2, topic_num=3,
                 tag_filter=util.allow_speech_tags, user_dict=''):
        super().__init__(tag_filter=tag_filter, user_dict=user_dict, is_lower=True, print_num=20, word_min_len=2, window=2)
        self.tag_filter = tag_filter
        self.tr4s = TextRank4Sentence(allow_speech_tags=self.tag_filter, user_dict=user_dict)
        self.tr4w = TextRank4Keyword(allow_speech_tags=self.tag_filter, user_dict=user_dict)
        self.is_lower = is_lower
        self.window = window
        self.print_num = print_num
        self.word_min_len = word_min_len
        self.min_phrase_len = min_phrase_len
        self.topic_num = topic_num

    def keywords(self, texts):
        """
        提取关键词、短语
        :param texts: 文本列表
        :return: Dict: {
                        word: [List[words], List[words], ...],
                        key_phrase: [phrases, phrases, ...],
                        weight: [List[weights], List[weights], ...],
                       }

        """
        try:
            assert len(texts) >= 1
        except:
            print(print('Error! No text found! At least one text input!'))
        results = dict()
        # print('关键词和关键短语提取……')
        for text in texts:
            self.tr4w.analyze(text=text,
                              lower=self.is_lower,
                              window=self.window)
            tmpphrase, tmpweight, tmpwords = [], [], []
            for phrase in self.tr4w.get_keyphrases(keywords_num=self.print_num, min_occur_num=self.min_phrase_len):
                tmpphrase.append(phrase)
            results.setdefault('key_phrase', []).append(tmpphrase)
            for item in self.tr4w.get_keywords(self.print_num, word_min_len=self.word_min_len):
                tmpwords.append(item.word)
                tmpweight.append(item.weight)
            results.setdefault('words', []).append(tmpwords)
            results.setdefault('weight', []).append(tmpweight)
        return results

    def keysentence(self, texts):
        """
        提取摘要
        :param texts: 文本列表
        :return: Dict：{
            "topics": {
                index: [List[indexs], List[indexs], ...],
                weight: [List[weights], List[weights], ...],
                sentence: [List[sentences], List[sentences], ...]
            }
        }
        """
        try:
            assert len(texts) >= 1
        except:
            print(print('Error! No text found! At least one text input!'))
        results = {'topics': {}}
        # print('摘要提取……')
        for text in texts:
            self.tr4s.analyze(text=text, lower=self.is_lower, source='all_filters')
            tmpindex, tmpweight, tmpsentence = [], [], []
            for item in self.tr4s.get_key_sentences(num=self.topic_num):
                tmpindex.append(item.index)
                tmpweight.append(item.weight)
                tmpsentence.append(item.sentence)
            results['topics'].setdefault('index', []).append(tmpindex)
            results['topics'].setdefault('weight', []).append(tmpweight)
            results['topics'].setdefault('sentence', []).append(tmpsentence)
        return results

    def alls(self, texts):
        """
        提取关键词、短语、摘要
        :param texts: 文本列表
        :return: Dict: {
            topics: {
                index: [List[indexs], List[indexs], ...],
                weight: [List[weights], List[weights], ...],
                sentence: [List[sentences], List[sentences], ...]
            }
            word: [List[words], List[words], ...],
            key_phrase: [phrases, phrases, ...],
            weight: [List[weights], List[weights], ...],

        }
        """
        alls = dict()
        keywords_process = self.keywords(texts)
        keySentence = self.keysentence(texts)
        keywords_process.update(keySentence)
        alls = keywords_process
        return alls

    def tf_idf(self, texts):
        return self.word_clean(texts)


if __name__ == '__main__':
    process = KeyItermMining()
    result = process.alls(['当创建一个模块有可能抛出多种不同的异常时，一种通常的做法是为这个包建立一个基础异常类，然后基于这个基础类为不同的错误情况创建不同的子类',
                           '前面的例子里充斥了很多Python内置的异常类型,读者也许会问,我可以创建自己的异常类型吗? 答案是肯定的,Python 允许用户自定义异常类型。实际开发中,有时候系统提供的异常类型不能满足开发的需求'])
    from pprint import pprint

    pprint(result)
