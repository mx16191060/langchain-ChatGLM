# -*- coding:utf-8 -*-

import openai

openai.api_key = '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'

def gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': "user", 'content': prompt}
        ]
    )

    text = response["choices"][0]["message"]["content"]
    print(text)
    return text

if __name__ == '__main__':
    gpt("")