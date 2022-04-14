import requests
import logging
import json


def get_hotdata():
    hotURL = (
        "http://api.inter.xueqiu.com/internal/stock/screener/quote/list.json?"
        "order_by=value&type=hot_1h&page=0&size=100"
    )
    header = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/79.0.3945.79 Safari/537.36"
    }
    empty_res = {
        "data": {},
        "error": 0,
        "error_description": "",
    }
    try:
        res = requests.get(hotURL, headers=header)
        if res.status_code == 200:
            logging.info("get hot data success")
            empty_res = res.json()
        else:
            logging.error("get hot data error")
            return empty_res
    except RuntimeError:
        logging.error("can't get hot data")
    finally:
        if len(empty_res["data"]) == 0:
            logging.error("can't get hot data")
        return empty_res


if __name__ == "__main__":
    hotData = get_hotdata()
    print(hotData)
    # for item in hotData["data"]["list"]:
    #     item["reason"] = "今天大涨"
    # print(hotData)
    # with open("hotdata.json", "w") as fp:
    #     json.dump(hotData, fp, indent=4, ensure_ascii=False)
    # print(len(hotData["data"]["list"]))