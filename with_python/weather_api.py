from ollama import chat
from ollama import ChatResponse
from ollama import Client
import json

LLM_MODEL = 'llama3.2:latest'##'deepseek-r1:1.5b'

## Tools
def get_weather_details(city):
    city_lower = city.lower()
    temp_dict = {
        'bangalore': '23',
        'kolkata': '35',
        'delhi': '42',
        'chennai': '38'
    }
    return temp_dict.get(city,'30')

tools = {
    "get_weather_details": get_weather_details
}

### SYSTEM PROMPT
SYSTEM_PROMPT = '''
You are an AI assistant with START, PLAN, ACTION, OBSERVATION and OUTPUT state.
Wait for the user prompt and first PLAN using available tools.
After PLAN, take the ACTION with appropiate tools and wait for OBSERVATION based on ACTION.
Once you get the OBSERVATION, Return the AI response as OUTPUT based on START prompt and OBSERVATION.

Strictly follow the JSON output format as in examples

Available Tools:
 - def get_weather_details(city)
 get_weather_details is a function that accepts city as string and returns the weather


Example:
START
{"type": "user", "user": "What is the sum of weather of delhi and bangalore"}
{"type": "plan", "plan": "I will call the function get_weather_details for delhi"}
{"type": "action", "function": "get_weather_details", "input": "delhi"}
{"type": "observation", "observation": "42"}
{"type": "plan", "plan": "I will call the function get_weather_details for bangalore"}
{"type": "action", "function": "get_weather_details", "input": "bangalore"}
{"type": "observation", "observation": "23"}
{"type": "output", "output": "The sum of temperature for delhi and bangalore is 65"}

'''

## LLM chat response
def get_llm_reponse(user_query):
    client = Client(
        host='http://localhost:11434'
    )
    response: ChatResponse = client.chat(
        model=LLM_MODEL,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_query}
        ]
    )
    return response.message.content

def display_llm_response_streaming(user_query):
    client = Client(
        host='http://localhost:11434'
    )
    stream_response: ChatResponse = client.chat(
        model=LLM_MODEL,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_query}
        ],
        stream=True,
    )
    for chunk in stream_response:
        print(chunk['message']['content'], end='', flush=True)





if __name__ == '__main__':
    # display_llm_response_streaming('what is the current temperature of delhi')

    USER_PROMPT = 'Hey, What is the current weather of chennai'
    # response: ChatResponse = chat(
    #     model=LLM_MODEL,
    #     messages=[
    #         {'role': 'system', 'content': SYSTEM_PROMPT},
    #         {
    #             'role': 'assistant',
    #             'content': '{"type": "plan", "plan": "I will call the function get_weather_details for chennai"}'
    #         },
    #         {
    #             'role': 'assistant',
    #             'content': '{"type": "action", "function": "get_weather_details", "input": "chennai"}'
    #         },
    #         {
    #             'role': 'assistant',
    #             'content': '{"type": "observation", "observation": "35"}'
    #         },
    #         {'role': 'user', 'content': USER_PROMPT}
    #     ]
    # )

    # print(response.message.content)

    chat_messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT}
    ]

    # print('Initial chat_messages :: ', chat_messages)

    while(True):
        user_query = input('Ask your Question >> ')
        query_format = {
            'type': 'user',
            'user': user_query
        }
        query_message = {
            'role': 'user',
            'content': json.dumps(query_format)
        }
        chat_messages.append(query_message)

        while(True):
            chat_response: ChatResponse = chat(
                model=LLM_MODEL,
                messages = chat_messages
            )

            print('chat_response.message --> ', chat_response.message)
            messages = chat_response.message
            print('type of messages => ', type(messages))
            # message_list = messages.split('\n')
            message_content_list = []
            for message in messages:
                print('message -', message)
                # message_content = message.content
                # message_content_list.append(message_content)

            print('message_content_list => ', message_content_list)




            result = chat_response.message.content
            print('first result = ', result)
            chat_messages.append({
                'role': 'assistant',
                'content': result
            })
            print('\n\n------------ AI ----------------\n')
            print(result)
            print('-------------END AI -------------------\n\n')
            # print('chat_messages :: post appending user query response ==> ', chat_messages)


            call = json.loads(result)

            if (call.get('type') == 'output'):
                print('Output :: ', call.output)
                break
            elif (call.get('type') == 'action'):
                fn = tools[call.get('function')]
                observation = fn(call.get('input'))
                obs = {"type": "observation", "observation": observation}
                chat_messages.append({
                    'role': 'assistant',
                    'content': json.dumps(obs)
                })








