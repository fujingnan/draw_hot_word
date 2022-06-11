# draw_hot_word
对某时间段内的文章提取热词

#### 核心代码

`utils/wordhotcompute.py`: 主要有两部分计算：1、贝叶斯均值法；2、牛顿冷却法。最终热度值由两部分结果加权求和得出；

需要注意的是，热词挖掘单靠热度值是不够的，还需要综合考虑文章的动态特征，例如评论数、term重要性等；
