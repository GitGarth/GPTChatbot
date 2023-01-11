import openai
import uuid


def read_file(filename):
    with open(filename, "r") as f:
        lines = "".join(f.readlines())
        return lines


class GPT3(object):
    def __init__(self, key_file=None):
        self.open_ai_key = read_file(key_file)
        openai.api_key = self.open_ai_key

    @staticmethod
    def gpt_completion(prompt, model="text-davinci-003", temp=0.6, max_tokens=1000,
                       frequency_penalty=0.6):
        prompt_uuid = str(uuid.uuid4())
        response = openai.Completion.create(
            prompt=f"{prompt_uuid}\n\n{prompt}",
            model=model,
            temperature=temp,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
        )['choices'][0]['text']
        return response

    @staticmethod
    def gpt_embedding(text, engine="text-similarity-davinci-001"):
        response = openai.Embedding.create(
            input=text,
            engine=engine)['data'][0]['embedding']
        return response

    @staticmethod
    def gpt_edit(text, instruction, model="text-davinci-edit-001"):
        response = openai.Edit.create(
            input=text,
            instruction=instruction,
            model=model)['choices'][0]['text']
        return response


if __name__ == "__main__":
    import os
    filepath = os.path.expanduser(".")
    gpt3 = GPT3(key_file=os.path.join(filepath, "openai_key.txt"))
    gpt3_response = gpt3.gpt_completion(prompt="Write a sentence.")
    vector = gpt3.gpt_embedding(text=gpt3_response)
    print(vector)
    edit = gpt3.gpt_edit(text=gpt3_response, instruction="Replace all nouns in input with the phrase, 'I like bacon.'")
    print(vector, gpt3_response, edit)

