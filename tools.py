import requests
from tavily import TavilyClient



get_weather_tools = {
    "type" : "function" ,
    "strict" : True , 
    "function" : {
        "name" : "get_weather" ,
        "description" : "查询指定城市的实时天气",
        "parameters" : {
            "type" : "object" ,
            "properties" : {
                "city" : {
                    "type" : "string" ,
                    "description" : "城市名称。例如 '北京', 'Shanghai', 'New York'。如果用户提到‘首都’或‘魔都’，请转换为具体城市名。"
                }
            },
            "required" : ["city"]
        }
    }
}

get_attraction_tools = {
    "type" : "function" ,
    "strict" : True ,
    "function" : {
        "name" : "get_attraction" ,
        "description" : "根据城市和天气搜索推荐的旅游景点,并且提供推荐的理由",
        "parameters" : {
            "type" : "object" ,
            "properties" : {
                "city" : {
                    "type" : "string" ,
                    "description" : "城市名称。例如 '北京', 'Shanghai', 'New York'。如果用户提到‘首都’或‘魔都’，请转换为具体城市名。"
                },
                "weather" : {
                    "type" : "string" ,
                    "description" : "天气描述。例如 '晴朗', '雨天', '多云'。请根据用户提供的天气信息进行推荐。"
                }
            },
            "required" : ["city", "weather"]
        }
    }
}

tools = [get_weather_tools, get_attraction_tools]

def get_weather(city: str) -> str:

    url = f"https://wttr.in/{city}?format=j1"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]

        return f"{city}的当前天气是{weather_desc}，温度为{temp_c}°C。"
    except requests.RequestException as e:
        return f"无法获取天气信息: {e}"
    except (KeyError, IndexError) as e:
        return f"解析天气信息时发生错误: {e}"



def get_attraction(city: str, weather: str ) -> str:

    tavily_client = TavilyClient(api_key="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")

    query = f"'{city}' 在 '{weather}' 天气下的旅游景点推荐以及理由"


    try:
        response = tavily_client.search(query=query, search_depth="basic" , include_answer=True)

        if response.get("answer"):
            return response["answer"]

        formatted_results = []

        for result in response.get("results", []):
            formatted_results.append(f"景点: {result['title']}\n理由: {result['connect']}\n")

        
        if not formatted_results:
            return "没有找到相关的旅游景点推荐。"
        
        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"无法获取景点信息: {e}"
    



available_tools = {"get_weather": get_weather, "get_attraction": get_attraction}