"""
This project is created by authors: Yulin Hu, Yijun Liu
The code and dataset available in the repository are under Apache License 2.0
Feel free to use or modify
"""

import logging
import re
import requests
import json
import random

# Configuration
# DO NOT CHANGE THESE PARAMETERS
MODEL_NAME = "mistral:7b"
MAX_NEW_TOKENS = 128

# YOU CAN CHANGE THESE PARAMETERS
NUM_CONVERSATIONS = 1000
QUALITY_THRESHOLD = 3

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Cache structure: {"conversation": String, "quality": Integer}
conversation_quality_cache = {}
conversations = []

# Use Ollama to host local model: https://ollama.ai/library/mistral/tags
def run_mistral(prompt, max_new_tokens=MAX_NEW_TOKENS):
    try:
        r = requests.post('http://localhost:11434/api/generate',
                          json={
                              'model': MODEL_NAME,
                              'prompt': prompt,
                              "max_new_tokens": max_new_tokens
                          },
                          stream=True)
        r.raise_for_status()

        response_text = ""
        for line in r.iter_lines():
            body = json.loads(line)
            response_part = body.get('response', '')
            response_text += response_part
            if 'error' in body:
                raise Exception(body['error'])

            if body.get('done', False):
                return response_text
    except Exception as e:
        logging.error(f"API call failed with error: {e}")
        return ""

def check_conversation_quality(conversation):
    quality = 0
    prompt = f"Conversation:\n<SYS> The following conversation is between a human and a helpful AI assistant. They are talking about the weather </SYS>\nUSER: Hi, how are you?\nASSISTANT: Hi\nUSER: Great wethear\nASSISTANT: yeahhh\nComment: \nsimple sentences, grammar errors, and informal language, don't answer the user back\nRating: 1\n\n Conversation:\n <SYS> The following conversation is between a human and a helpful AI assistant. They are talking about USA </SYS>\n USER: Do you know where's the capital city of the USA?\n ASSISTANT: Yes, it's Washington D.C.. \n USER: Oh great! Thanks for that. Do you know the history of Washington D.C.?\n ASSISTANT: Washington, D.C., the capital city of the United States, has a rich and unique history. The location of the capital was selected by President George Washington in 1790. The area chosen was a square measuring 10 miles on each side, sitting on land donated by Maryland and Virginia. Comment: \n Rich content, free of grammar errors, variety of entities included in the assistant's response\n Rating: 5\n\n Conversation: {conversation}\n Rating: "
    result = run_mistral(prompt)
    score = quality_score(result)
    if score is not None:
        quality = score
    else:
        result = run_mistral(prompt)
        score = quality_score(result)
        if score is None:
            score = 1
    return quality

def quality_score(text):
    numbers = re.findall(r'\d+', text)
    if numbers:
        first_number = int(numbers[0])
        if 1 <= first_number <= 5:
            return first_number
    return None

def generate_conversation(topic, starter):
    '''
    The goal of this function is to generate a different conversation each time.
    '''

    conversation = ""
    max_try = 0

    if conversation == "":
        system_message = (f"<SYS> The following conversation is between a human and a helpful AI assistant. They are talking about {topic}. </SYS> ")
        user_message = (starter)
        prompt = f"{system_message}\nUSER: {user_message}\nASSISTANT:"
        conversation = run_mistral(prompt)
        max_try = 0
    else:
        while(check_conversation_quality(conversation) < QUALITY_THRESHOLD and max_try < 3):
            max_try += 1
            system_message = (f"<SYS> The following conversation is between a human and a helpful AI assistant. They are talking about {topic}. </SYS> ")
            user_message = (starter)
            prompt = f"{system_message}\nUSER: {user_message}\nASSISTANT:"
            conversation = run_mistral(prompt)
    return conversation

def get_some_topics():
    # generate n nouns as topics
    word_list = ["water", "usa", "god", "apple", "bicycle", "cloud", "dragon", "elephant", "flower", "guitar", "harbor",    "iceberg", "jungle", "kettle", "lighthouse", "mountain", "notebook", "ocean",    "piano", "quilt", "rainbow", "star", "telescope", "umbrella", "volcano", "waterfall",    "xylophone", "yacht", "zebra", "battery", "calendar", "desk", "elevator", "fountain",    "grapes", "helmet", "island", "jewel", "key", "lemon", "mirror", "nest", "owl",    "pumpkin", "queen", "robot", "snake", "tree", "universe", "vase", "window", "x-ray", "yo-yo"]
    sampled_word_1 = random.choice(word_list)
    sampled_word_2 = random.choice(word_list)
    sampled_word_3 = random.choice(word_list)
    sampled_word_4 = random.choice(word_list)
    sampled_word_5 = random.choice(word_list)

    user_message = (f"Continue this comprehensive list of topics in conversations:\n 1. {sampled_word_1}\n 2. {sampled_word_2}\n 3. {sampled_word_3}\n 4. {sampled_word_4}\n 5. {sampled_word_5}")
    prompt = f"{user_message}\n"
    result = run_mistral(prompt)
    return result

def get_n_topics(n):
    unique_terms = set()
    while len(unique_terms) < n:
        print(f"{len(unique_terms)} topics generated...")
        output = get_some_topics()
        for line in output.split('\n'):
            parts = line.split('. ')
            if len(parts) > 1:
                term = parts[1].strip()
                if ' ' not in term and term and term not in unique_terms and not term.isnumeric():
                    unique_terms.add(term)
    return list(unique_terms)

def get_some_starter_sentence(word):
    # generate n nouns as topics
    user_message = (f"Give me one open-ended question on the topic of {word}. Question: ")
    prompt = f"{user_message}\n"
    result = run_mistral(prompt)
    return find_first_question(result)

def get_starter_sentence_list(topics):
    starters = []
    for i in topics:
        question = get_some_starter_sentence(i)
        max_try = 0
        while question is None and max_try < 5:
            max_try += 1
            question = get_some_starter_sentence(i)
        if question is not None:
            starters.append(question)
    return starters

def get_starter_sentence(topics, n):
    starters = []
    max_try = 3

    while len(starters) < n and max_try > 0:
        max_try -= 1
        question_list = get_starter_sentence_list(topics)
        starters.extend(question_list)

    return starters


def find_first_question(text):
    sentences = re.findall(r'[^.?!]+[.?!]', text)
    for sentence in sentences:
        if '?' in sentence:
            return sentence.strip()
    return None


def find_substring_indices(original, substring):
    """
    Function to find all the starting indices of occurrences of a substring in a given string.
    """
    main_string = original.lower()
    indices = []
    index = main_string.find(substring)
    while index != -1:
        indices.append(index)
        # Move to the next possible start position
        index = main_string.find(substring, index + 1)

    return indices


def main():
    # 1. decide the number of conversations
    NUM_CONVERSATIONS = 1000

    # write to log file
    data_to_log = f"\nStarting to generate {NUM_CONVERSATIONS} conversations..."
    with open('logs.txt', 'a') as file:
        file.write(data_to_log)

    # 2. generate n words (nouns) as conversation topics

    # write to log file
    data_to_log = f"\nGenerating {NUM_CONVERSATIONS} topics..."
    with open('logs.txt', 'a') as file:
        file.write(data_to_log)

    # generate topics
    topics = get_n_topics(NUM_CONVERSATIONS)  # topics = ["war", "usa", "html"]

    # save topics to topics.txt
    with open("topics.txt", "w") as file:
        for string in topics:
            file.write(string + "\n")

    # read from topics.txt
    topics = []
    with open("topics.txt", "r") as file:
        for line in file:
            temp = line.strip().lower()
            letters_only = ''.join(char for char in temp if char.isalpha())
            topics.append(letters_only)

# 3. generate n starter sentences for the conversations

    # write to log file
    data_to_log = f"\nGenerating {NUM_CONVERSATIONS} starters..."
    with open('logs.txt', 'a') as file:
        file.write(data_to_log)

    # generate starters
    starters = get_starter_sentence(topics, NUM_CONVERSATIONS)

    # save starters to starters.txt
    with open("starters.txt", "w") as file:
        for string in starters:
            file.write(string + "\n")

    # read from starters.txt
    starters = []
    with open("starters.txt", "r") as file:
        starters = [line.strip() for line in file]

    # write to log file
    data_to_log = f"\n{NUM_CONVERSATIONS} starters generated in starters.txt"
    with open('logs.txt', 'a') as file:
        file.write(data_to_log)

    # 4. update sys context and generate conversations

    # write to log file
    data_to_log = f"\nGenerating {NUM_CONVERSATIONS} conversations..."
    with open('logs.txt', 'a') as file:
        file.write(data_to_log)

    # generate conversations
    for i in range(NUM_CONVERSATIONS):
        conversations.append(generate_conversation(topics[i], starters[i]))
        print(f"{i} conversations generated...")

    # 5. truncate conversations to limit to 6 turns
    substring = "assistant"
    for i in conversations:
        index_list = find_substring_indices(i, substring)
        if len(index_list) >= 7:
            i = i[:index_list[5]]

    # add the starter
    final_conversations = []

    for i in range(len(conversations)):
        system_prompt = f"<SYS> The following conversation is between a human and a helpful AI assistant. They are talking about {topics[i]}. </SYS>"
        starter_prompt = f"{system_prompt}\nUSER: {starters[i]}\nASSISTANT:"
        final_conversations.append(starter_prompt + conversations[i])

    # end with assistant repsonse
    for conv in final_conversations:
        index_of_last_user = conv.lower().rfind("user")
        index_of_last_assistant = conv.lower().rfind("assistant")
        if index_of_last_user != -1 and index_of_last_assistant != -1:
            if index_of_last_user > index_of_last_assistant:
                conv = conv[:index_of_last_assistant]

    # save the result data
    with open("synthetic_dataset.txt", "w") as f:
        for conv in final_conversations:
            conversation_formatted = conv.replace("\n",
                                                  "\\n")
            f.write(conversation_formatted + "\n")

    # write to log file
    data_to_log = f"\nTask finished: {NUM_CONVERSATIONS} conversation are saved in synthetic_dataset.txt."
    with open('logs.txt', 'a') as file:
        file.write(data_to_log)


if __name__ == "__main__":
    main()
