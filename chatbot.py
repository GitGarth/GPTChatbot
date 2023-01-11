import json
import numpy as np
import os
import time

from gpt_core import GPT3, read_file

class ExitChat(Exception):
    def __init__(self):
        super().__init__()

class ChatBot(object):
    def __init__(self, bot_name="AVA", prompt=None, prompt_filepath=None):
        self.gpt3 = GPT3(key_file="OpenAIKey")
        self.bot_name = bot_name
        self.conversation = list()
        self.similar_conversation = list()
        self.prompt = prompt

        if not prompt_filepath and not prompt:
            self.prompt_filepath = os.path.join(
                os.path.expanduser("."),
                "prompts",
                "chat_prompt.txt"
            )

    @staticmethod
    def prompt_user(prompt):
        user_input = input(prompt)
        return user_input

    @staticmethod
    def similarity(v1, v2):
        return np.dot(v1, v2)

    def search_index(self, recent_line, all_lines, count):
        scores = list()
        for line in list(all_lines):
            if line['vector'] != recent_line['vector']:
                score = self.similarity(recent_line["vector"], line["vector"])
                scores.append({'line_in': line["line_in"], 'score': score})
        ordered_scores = sorted(scores, key=lambda d: d['score'])[-count:]
        lines_out = [line['line_in'] for line in ordered_scores]
        return lines_out

    def handle_input(self, line_in):
        if line_in not in self.conversation:
            f_line = f"USER: {line_in}"
            vector = self.get_embedding(line_in=f_line)
            return {'line_in': f_line, 'vector': vector}

    def get_embedding(self, line_in):
        return self.gpt3.gpt_embedding(text=line_in)

    def get_completion(self, prompt):
        return self.gpt3.gpt_completion(prompt=prompt)

    def get_edit(self, line_in, instruction, model="text-davinci-edit-001"):
        return self.gpt3.gpt_edit(text=line_in, instruction=instruction, model=model)

    def save_session(self, log_name):
        current_directory = os.path.abspath(".")
        file_location = os.path.join(current_directory, "logs", f"{log_name}_{time.time()}.json")
        with open(file_location, "w") as f:
            json.dump(self.conversation, f)

    def load_session(self, log_name):
        current_directory = os.path.abspath(".")
        file_location = os.path.join(current_directory, "logs", f"{log_name}.json")
        with open(file_location, "r") as f:
            self.conversation = json.load(f)

    def find_similar_lines(self, line_in, count=10):
        old_lines = self.search_index(line_in, self.conversation, count)
        return old_lines

    def get_recent_conversation(self, count=30):
        return [line.get('line_in') for line in self.conversation[-count:]
                if line.get('line_in') not in self.similar_conversation]

    def form_prompt(self):
        self.similar_conversation = self.find_similar_lines(self.conversation[-1], count=10)
        similar = "\n".join(self.similar_conversation)
        recent = "\n".join(self.get_recent_conversation())
        raw_prompt = self.prompt
        if self.prompt_filepath:
            with open(self.prompt_filepath, "r") as f:
                raw_prompt = "\n".join(f.readlines())
        block = f"{similar}\n{recent}"
        prompt = raw_prompt.replace(
            "<<BLOCK>>",
            block)
        return prompt

    def get_response(self, prompt):
        response = f"{self.bot_name}: {self.get_completion(prompt).strip()}"
        vector = self.gpt3.gpt_embedding(response)
        return {'line_in': response, 'vector': vector}

    def prepare_subject(self, subject):
        info = self.handle_input(subject)
        self.conversation.append(info)

        return info['line_in']

    def chat_step(self, user_input):
        if user_input in ["quit", "exit"]:
            raise ExitChat
        info = self.handle_input(user_input)
        if info is not None:
            self.conversation.append(info)
        prompt = self.form_prompt()
        response = self.get_response(prompt)
        self.conversation.append(response)
        print(self.conversation[-1]['line_in'])

    def chat_loop(self, greeting=None):
        if greeting:
            print(greeting)

        while True:
            try:
                user_input = self.prompt_user("USER: ")
                self.chat_step(user_input)
            except ExitChat:
                break

    def bot_loop(self):
        pass


if __name__ == "__main__":
    bot = ChatBot(bot_name="AVA")
    bot.chat_loop(greeting="AVA: Hello, I'm Ava. How can I help you?\n" +
                           "AVA: Please type 'exit' or 'quit' to end our session.")
    bot.save_session("conversation")
