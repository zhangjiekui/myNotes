import os
import platform
from os.path import join
from transformers import AutoModel,AutoTokenizer
from ChatGLM3.composite_demo.tool_registry import register_tool,dispatch_tool,_TOOL_HOOKS,_TOOL_DESCRIPTIONS
from typing import Annotated
import json

@register_tool
def get_weather(
    city_name: Annotated[str, 'The name of the city to be queried', True],
) -> str:
    """
    Get the current weather for `city_name`
    """

    if not isinstance(city_name, str):
        raise TypeError("City name must be a string")

    key_selection = {
        "current_condition": ["temp_C", "FeelsLikeC", "humidity", "weatherDesc",  "observation_time"],
    }
    import requests
    try:
        resp = requests.get(f"https://wttr.in/{city_name}?format=j1")
        resp.raise_for_status()
        resp = resp.json()
        ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}
    except:
        import traceback
        ret = "Error encountered while fetching weather data!\n" + traceback.format_exc() 

    return str(ret)

@register_tool
def random_number_generator(
    seed: Annotated[int, 'The random seed used by the generator', True], 
    range: Annotated[tuple[int, int], 'The range of the generated numbers', True],
) -> int:
    """
    Generates a random number x, s.t. range[0] <= x < range[1]
    """
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer")
    if not isinstance(range, tuple):
        raise TypeError("Range must be a tuple")
    if not isinstance(range[0], int) or not isinstance(range[1], int):
        raise TypeError("Range must be a tuple of integers")

    import random
    return random.Random(seed).randint(*range)


if __name__ == "__main__":

    print("---------------------------------")
    print(_TOOL_HOOKS)
    print("=================================")
    print(_TOOL_DESCRIPTIONS)
    print("---------------------------------")





    win_model_base = r"D:\biaoshu\localmodels" #本地模型文件夹
    wls_model_base = r"/mnt/d/biaoshu/localmodels/"

    system_name = platform.system()
    is_win =  system_name == "Windows"

    model_base = win_model_base if is_win else wls_model_base
    pt = "chatglm3-6b"
    pt = join(model_base,pt)
    print(pt)



    tokenizer = AutoTokenizer.from_pretrained(pt,trust_remote_code = True)
    model     = AutoModel.from_pretrained(pt,trust_remote_code=True).quantize(4).cuda()
    model    = model.eval()

    sys_message={
        'role':'system',
        'content':'Answer the following questions as best as you can. You hava access to the following tools:',
        'tools':_TOOL_DESCRIPTIONS
    }
    query = '请使用工具get_weather查询合肥今天的天气怎么样？然后告诉我今天适合穿什么衣服？'
    print(sys_message)
    

    response,history = model.chat(tokenizer,query,history = [sys_message])
    print("first response:",response)

    while isinstance(response,dict):
        query_result=dispatch_tool(response['name'],response['parameters'])
        result = json.dumps(query_result, ensure_ascii=False)
        response, history = model.chat(tokenizer, result, history=history, role="observation")
    print(response)