import pandas as pd
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from utils.keyitermmining import KeyItermMining
from TextRank4ZH.textrank4zh import util


class Parser(KeyItermMining):
    def __init__(self, is_lower=True, window=2, print_num=20, word_min_len=2, min_phrase_len=2, topic_num=3,
                 tag_filter=util.allow_speech_tags, filepath='', current_day=20220402, user_dict='/Users/fujingnan/PycharmProjects/draw_hot_word/datas/custom.dict'):
        """

        :param is_lower: 是否统一小写
        :param window: textrank滑动窗口大小
        :param print_num: 每个文本关键短语最大输出数量
        :param word_min_len: 关键词最小长度
        :param min_phrase_len: 关键短语最小长度
        :param topic_num: 每个文本摘要提取最大数量
        :param tag_filter: 分词结果中需要保留的词性
        :param filepath: 热词计算源文件路径: .csv
        :param current_day: 当前日期: yyyyMMdd
        :param re_symbol_format: 去除股票标志正则
        :param re_at_format: 去除@xxx标志正则
        :param re_http_format: 去除网址格式正则
        :param history_day: 历史DataFrame
        :param current_day: 当前DataFrame
        """
        super().__init__(
            is_lower=is_lower,
            window=window,
            print_num=print_num,
            word_min_len=word_min_len,
            min_phrase_len=min_phrase_len,
            topic_num=topic_num,
            tag_filter=tag_filter,
            user_dict=user_dict
        )
        self.re_symbol_format = "[$][^$\\n\\f\\r\\t\\v]{0,80}[(（]([a-zA-Z0-9\\.\\_\\-\\+]{0," + "20})[)）][$]"
        self.re_at_format = "[@＠]([\u4E00-\u9FFFa-zA-Z0-9_-]{2,})(?![\u4E00-\u9FFFa-zA-Z0-9_-]*\[¥([.0-9]+)\])"
        self.re_http_format = "((?:ftp://|https://|http://|www\\.)[a-zA-Z0-9?%&=#" \
                              + "./_!+:\\-\\[\\]~,@;\\*]*\\.[a-zA-Z0-9?%&=#./_!+:\\-\\[\\]~,@;\\*]*?" \
                              + "(?=(\\/\\/[@＠][\\u4E00-\\u9FFFa-zA-Z0-9_-]+( )?)|[^a-zA-Z0-9?%&=#./_!+:\\-\\[\\]~,@;]|(&nbsp;)|$))" \
                              + "(\\{([^{\\s]+)\\})?"
        self.current_day = current_day
        self.filepath = filepath
        # self.tag_filter = tag_filter
        # self.is_lower = is_lower
        # self.window = window
        # self.print_num = print_num
        # self.word_min_len = word_min_len
        # self.min_phrase_len = min_phrase_len
        # self.topic_num = topic_num
        # self.keyitemmining = KeyItermMining,
        self.datas = pd.read_csv(self.filepath, keep_default_na=False)
        self.history_day = self.datas[self.datas['day'] < self.current_day]
        self.current_day = self.datas[self.datas['day'] == self.current_day]

    def transform_map(self, is_current):
        """
        生成 symbol->{'id': , 'text': } 对应关系
        :param is_current: bool: 是否生成当前时间里symbol->{'id': , 'text': }对应关系;True：当前，False: 历史
        :return: Dict: {
            symbol: {
                'id': [status_ids],
                'text': [texts]
            },
            ...
        }
        """
        if is_current:
            type_day = self.current_day
        else:
            type_day = self.history_day
        re_symbol_format = re.compile(self.re_symbol_format)
        re_at_format = re.compile(self.re_at_format)
        re_http_format = re.compile(self.re_http_format)
        symbol_text_map = dict()
        for row in type_day.itertuples():
            symbol = getattr(row, 'stock_name')
            for content_type in ['title', 'text']:
                content = str(getattr(row, content_type))
                if content:
                    soup = BeautifulSoup(content, "html.parser", from_encoding='utf-8')
                    content = re_symbol_format.sub('', soup.get_text()).replace('\xa0', '')
                    content = re_at_format.sub('', content)
                    content = re_http_format.sub('', content)
                if not symbol in symbol_text_map:
                    symbol_text_map[symbol] = dict()
                if content_type == 'title':
                    symbol_text_map[symbol].setdefault('id', []).append(getattr(row, 'id'))
                    symbol_text_map[symbol].setdefault('user_id', []).append(getattr(row, 'user_id'))
                    symbol_text_map[symbol].setdefault('评论数', []).append(getattr(row, 'reply_count'))
                    symbol_text_map[symbol].setdefault('day', []).append(getattr(row, 'day'))
                if not content:
                    symbol_text_map[symbol].setdefault(content_type, []).append('')
                else:
                    symbol_text_map[symbol].setdefault(content_type, []).append(content)


        return symbol_text_map

    def get_text_info(self, extract_level, texts):
        """
        为每个文本提取关键词/关键短语/摘要
        :param extract_level: 文本解析粒度: keywords: 仅提取关键词、关键短语
                                          keysentence: 仅提取摘要
                                          alls: 提取所有关键信息
        :param texts: 文本列表
        :return: 参考 KeyItermMining 类
        """
        # if not extract_level in {'keywords', 'keysentence', 'alls', 'tf-idf'}:
        #     "please choose one of parameters in {'keywords', 'keysentence', 'alls', 'tf-idf'}"
        if extract_level == 'keywords':
            return self.keywords(texts)
        if extract_level == 'keysentence':
            return self.keysentence(texts)
        if extract_level == 'alls':
            return self.alls(texts)
        if extract_level == 'tf-idf':
            return self.tf_idf(texts)

    def main(self, extract_level):
        """
        主函数，提取当前和历史文本解析结果
        :param extract_level: 参考 get_text_info 方法
        :return: (curr_symbol_text_map, his_symbol_text_map)Dict: {
            symbol: {
                'extract_results_title': 标题文本解析结果(同 get_text_info 方法返回格式)
                'extract_results_text': 正文文本解析结果(同 get_text_info 方法返回格式)
            },
            ...
        }
        """
        curr_symbol_text_map = self.transform_map(True)
        his_symbol_text_map = self.transform_map(False)
        print("解析当天文本信息……")
        for symbol, item in tqdm(curr_symbol_text_map.items()):
            for content_type in ['title', 'text']:
                texts = item[content_type]
                extract_results = self.get_text_info(extract_level, texts)
                curr_symbol_text_map[symbol]['extract_results_{}'.format(content_type)] = extract_results
        print("解析历史文本信息……")
        for symbol, item in tqdm(his_symbol_text_map.items()):
            for content_type in ['title', 'text']:
                texts = item[content_type]
                extract_results = self.get_text_info(extract_level, texts)
                his_symbol_text_map[symbol]['extract_results_{}'.format(content_type)] = extract_results
        return curr_symbol_text_map, his_symbol_text_map


if __name__ == '__main__':
    from pprint import pprint
    parser = Parser(filepath='../csv/test.csv')
    cur_res, his_res = parser.main('alls')
    print(his_res)
