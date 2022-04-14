from utils.data_parser import Parser
from utils.wordhotcompute import HotWordMining
from tqdm import tqdm
import json
from collections import OrderedDict
import os
from pprint import pprint

tag_filter = ['n', 'nr', 'nr2', 'nrj', 'nrf', 'ns', 'nt', 'nz', 's', 'vn']
parser = Parser(filepath='csv/hot_stock_cards.csv',
                tag_filter=tag_filter,
                word_min_len=2)

if os.path.exists('./cache/current_res.json') and os.path.exists('./cache/history_res.json'):
    fp1 = open('./cache/current_res.json', 'r', encoding='utf-8')
    current_res = json.load(fp1)
    fp2 = open('./cache/history_res.json', 'r', encoding='utf-8')
    history_res = json.load(fp2)
    fp1.close()
    fp2.close()
else:
    current_res, history_res = parser.main('keywords')
    json_str1 = json.dumps(current_res, ensure_ascii=False)
    json_str2 = json.dumps(history_res, ensure_ascii=False)
    with open('./cache/current_res.json', 'w', encoding='utf-8') as fout:
        fout.write(json_str1)
    with open('./cache/history_res.json', 'w', encoding='utf-8') as fout:
        fout.write(json_str2)

finall_res = OrderedDict()
sid_phrase_map = {}
for symbol, item in tqdm(current_res.items()):
    cur_extract_res = item['extract_results']
    cur_ids = item['id']
    cur_text = '。'.join(item['text'])
    word_sid_map = {}
    assert len(cur_ids) == len(cur_extract_res['words'])
    for index, ws in enumerate(cur_extract_res['words']):
        for w in ws:
            word_sid_map.setdefault(w, []).append(cur_ids[index])
    for index, sid in enumerate(cur_ids):
        sid_phrase_map.setdefault(sid, []).extend(cur_extract_res['key_phrase'][index] if cur_extract_res['key_phrase'][index] else '')
    words = list(word_sid_map.keys())

    # clean_monitor = tf_idf.TF_IDF(tag_filter=tag_filter, cut_ignore=words)
    # word_cleaned_map = clean_monitor.word_clean(item['text'])
    # words = word_cleaned_map.keys()[:10]

    if symbol in history_res:
        try:
            his_text = '。'.join(history_res[symbol]['text'])
        except:
            pprint(history_res[symbol]['extract_results'])
    else:
        his_text = ''
    hwm = HotWordMining(duration=7)

    hot_res = hwm.bn_hot(words, his_text, cur_text)
    word_info = {}
    for w, h in hot_res.items():
        if not w in word_info:
            word_info[w] = OrderedDict()
        word_info[w]['热度值'] = h
        word_info[w]['status_id'] = word_sid_map[w]
    finall_res[symbol] = word_info
js_str1 = json.dumps(finall_res, ensure_ascii=False)
js_str2 = json.dumps(sid_phrase_map, ensure_ascii=False)
with open('results/热词挖掘结果.json', 'w', encoding='utf-8') as f:
    f.write(js_str1)

with open('results/帖子短语挖掘结果.json', 'w', encoding='utf-8') as f:
    f.write(js_str2)