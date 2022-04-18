from utils.data_parser import Parser
from utils.wordhotcompute import HotWordMining
from tqdm import tqdm
import json
from collections import OrderedDict
import os
from ahocorapy.keywordtree import KeywordTree
import pandas as pd
from pprint import pprint
from bs4 import BeautifulSoup
import re

tag_filter = ['n', 'vn', 'nz', 'a', 'PER', 'LOC', 'ORG', 'TIME', 'nw']
parser = Parser(filepath='/Users/fujingnan/Downloads/hot_stock_cards.csv',
                tag_filter=tag_filter,
                word_min_len=2)

re_symbol_format = "[$][^$\\n\\f\\r\\t\\v]{0,80}[(（]([a-zA-Z0-9\\.\\_\\-\\+]{0," + "20})[)）][$]"
re_at_format = "[@＠]([\u4E00-\u9FFFa-zA-Z0-9_-]{2,})(?![\u4E00-\u9FFFa-zA-Z0-9_-]*\[¥([.0-9]+)\])"
re_http_format = "((?:ftp://|https://|http://|www\\.)[a-zA-Z0-9?%&=#" \
                      + "./_!+:\\-\\[\\]~,@;\\*]*\\.[a-zA-Z0-9?%&=#./_!+:\\-\\[\\]~,@;\\*]*?" \
                      + "(?=(\\/\\/[@＠][\\u4E00-\\u9FFFa-zA-Z0-9_-]+( )?)|[^a-zA-Z0-9?%&=#./_!+:\\-\\[\\]~,@;]|(&nbsp;)|$))" \
                      + "(\\{([^{\\s]+)\\})?"
re_symbol_format = re.compile(re_symbol_format)
re_at_format = re.compile(re_at_format)
re_http_format = re.compile(re_http_format)

if os.path.exists('./cache/current_res.json') and os.path.exists('./cache/history_res.json'):
    fp1 = open('./cache/current_res.json', 'r', encoding='utf-8')
    current_res = json.load(fp1)
    fp2 = open('./cache/history_res.json', 'r', encoding='utf-8')
    history_res = json.load(fp2)
    fp1.close()
    fp2.close()
else:
    if not os.path.exists('cache'):
        os.mkdir('cache')
    current_res, history_res = parser.main('alls')
    json_str1 = json.dumps(current_res, ensure_ascii=False)
    json_str2 = json.dumps(history_res, ensure_ascii=False)
    with open('./cache/current_res.json', 'w', encoding='utf-8') as fout:
        fout.write(json_str1)
    with open('./cache/history_res.json', 'w', encoding='utf-8') as fout:
        fout.write(json_str2)

datas = pd.read_csv('/Users/fujingnan/Downloads/hot_stock_cards.csv', keep_default_na=False)

finall_res = {}
sid_phrase_map = {}
sid_topic_map = {}
for symbol, item in tqdm(current_res.items()):
    for content_type in ['title', 'text']:
        cur_extract_res = item['extract_results_{}'.format(content_type)]
        cur_ids = item['id']
        cur_text = '。'.join(item[content_type])
        word_sid_map = {}
        assert len(cur_ids) == len(cur_extract_res['words'])
        for index, ws in enumerate(cur_extract_res['words']):
            for w in ws:
                word_sid_map.setdefault(w, []).append(cur_ids[index])
        for index, sid in enumerate(cur_ids):
            sid_phrase_map.setdefault(sid, []).extend(
                cur_extract_res['key_phrase'][index] if cur_extract_res['key_phrase'][index] else '')
            sid_topic_map.setdefault(sid, []).extend(
                cur_extract_res['topics']['sentence'][index] if cur_extract_res['topics']['sentence'][index] else '')
        words = list(word_sid_map.keys())
        kwtree = KeywordTree(case_insensitive=True)
        for word in words:
            kwtree.add(str(word))
        kwtree.finalize()
        word_count_map = {}


        # for index, text in enumerate(item[content_type]):
        #     match_res = kwtree.search_all(text)
        #     date = str(item['day'][index])
        #     for ws, _ in match_res:
        #         if not ws in word_count_map:
        #             word_count_map[ws] = dict()
        #         if not date in word_count_map[ws]:
        #             word_count_map[ws][date] = 0
        #         word_count_map[ws][date] += 1


        for date in datas.day.unique():
            datetimetext = '。'.join(datas[(datas.day==date)&(datas.stock_name==symbol)][content_type].tolist())
            soup = BeautifulSoup(datetimetext, "html.parser", from_encoding='utf-8')
            content = re_symbol_format.sub('', soup.get_text()).replace('\xa0', '')
            content = re_at_format.sub('', content)
            content = re_http_format.sub('', content)
            match_res = kwtree.search_all(content)
            for w, _ in match_res:
                if not w in word_count_map:
                    word_count_map[w] = dict()
                # if not str(date) in word_count_map[w]:
                #     word_count_map[w][str(date)] = 0
                word_count_map[w].setdefault(str(date), 0)
                word_count_map[w][str(date)] += 1
        for w in word_count_map:
            for date in datas.day.unique():
                if not str(date) in word_count_map[w]:
                    word_count_map[w][str(date)] = 0
            word_count_map[w] = OrderedDict(sorted(word_count_map[w].items(), key=lambda x:x[0], reverse=True))


        if symbol in history_res:
            try:
                his_text = '。'.join(history_res[symbol][content_type])
            except:
                pprint(history_res[symbol]['extract_results_{}'.format(content_type)])
        else:
            his_text = ''
        hwm = HotWordMining(duration=7)

        hot_res = hwm.bn_hot(words, his_text, cur_text)
        word_info = {}
        for w, h in hot_res.items():
            if not w in word_info:
                word_info[w] = dict()
            word_info[w]['热度值'] = h
            word_info[w]['status_id'] = word_sid_map[w]
            word_info[w]['daly_count'] = word_count_map[w]
        if not symbol in finall_res:
            finall_res[symbol] = dict()
        if not word_info:
            finall_res[symbol][content_type + '_info'] = '{}'
        else:
            finall_res[symbol][content_type+'_info'] = word_info
    finall_res[symbol]['other_info'] = current_res[symbol]
# pprint(finall_res)
js_str1 = json.dumps(finall_res, ensure_ascii=False)
# js_str2 = json.dumps(sid_phrase_map, ensure_ascii=False)
with open('results/热词挖掘结果.json', 'w', encoding='utf-8') as f:
    f.write(js_str1)
#
# with open('results/帖子短语挖掘结果.json', 'w', encoding='utf-8') as f:
#     f.write(js_str2)