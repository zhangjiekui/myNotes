import re
def get_word_list(s1):
    # 把句子按字分开，中文按字分，英文按单词，数字按空格
    res = re.compile(r"([\u4e00-\u9fa5，、。！？])")  # [\u4e00-\u9fa5]中文范围,以及中文符号“，、。！？”拆分
    p1 = res.split(s1.lower())
    str1_list = []
    for seg in p1:
        str1_list=str1_list+(seg.split(' '))
    list_word1 = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符
    return list_word1


if __name__ == '__main__':
    s = "12、China's Legend Holdings will split its several business arms to go public on stock markets,  the group's president Zhu Linan said on Tuesday.该集团总裁朱利安周二表示，haha中国联想控股将分拆其多个业务部门在股市上市。"
    list_word1 = get_word_list(s)
    print(list_word1)
    print("\n")
    w = 'hi新手oh'
    list_word2 = get_word_list(w)
    print(list_word2)
    print("\n")
    w = 'John比Tom Marry年轻得多'
    list_word2 = get_word_list(w)
    print(list_word2)
    print("\n")
