import re

def no_space(char, prev_char):
    return char in set("',.!?") and prev_char != ' '  #set(',.!?，。！？')

def preprocess_nmt(raw_text):
    text=raw_text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 调用no_space
    out=[' '+char if (i>0 and no_space(char,text[i-1])) else char for i,char in enumerate(text)]
    return ''.join(out)

def get_word_list(text):
    # 调用preprocess_nmt
    text=preprocess_nmt(text)
    res = re.compile(r"([\u4e00-\u9fa5，、。！？])")  # [\u4e00-\u9fa5]中文范围,以及中文符号“，、。！？”拆分
    p1 = res.split(text.lower())
    str_list = []
    for seg in p1:
        str_list=str_list+(seg.split(' '))
    list_word = [w for w in str_list if len(w.strip()) > 0]  # 去掉为空的字符
    return list_word

if __name__ == '__main__':
    s = "12、China's Legend Holdings will split its several business arms to go public on stock markets,  the group's president Zhu Linan said on Tuesday.该集团总裁朱利安周二表示，haha中国联想控股将分拆其多个业务部门在股市上市。"
    list_word1 = get_word_list(s)
    print(list_word1)
    print("\n")
    w = "it's 新手oh"
    list_word2 = get_word_list(w)
    print(list_word2)
    print("\n")
    w = 'John比Tom Marry年轻得多'
    list_word2 = get_word_list(w)
    print(list_word2)
    print("\n")
