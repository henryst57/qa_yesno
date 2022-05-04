import os
import random
import numpy as np

def create_bio(output):
    bio_file = os.path.join("..", output, "bioasq_data_full.tsv")
    general_file = os.path.join("..", output, "squad_data.tsv")
    bio_data = []
    general_data = []
    with open(bio_file, encoding="utf-8") as f:
        bio_lines = f.readlines()
        for line_num in range(len(bio_lines)):
            bio_data.append([bio_lines[line_num].strip(), 1])
    with open(general_file, encoding="utf-8") as f:
        bool_lines = f.readlines()
        for line_num in range(len(bool_lines)):
            general_data.append([bool_lines[line_num].strip(), 0])
    data_size = 0
    if (len(bio_data) < len(general_data)):
        data_size = len(bio_data)
    else:
        data_size = len(general_data)

    random.shuffle(general_data)
    random.shuffle(bio_data)

    data = general_data[:data_size] + bio_data[:data_size]
    random.shuffle(data)

    file = os.path.join("..", output, "bio_dataset.tsv")
    with open(file, "wt", encoding="utf-8") as f:
        for d in data:
            f.write("{}\t{}\n".format(d[0], d[1]))

def create_yesno(output):
    bool_file = os.path.join("..", output, "boolq_data.tsv")
    general_file = os.path.join("..", output, "squad_data.tsv")
    bool_data = []
    general_data = []
    with open(bool_file, encoding="utf-8") as f:
        bool_lines = f.readlines()
        for line_num in range(len(bool_lines)):
            bool_lines[line_num] = bool_lines[line_num][:-2]
            store = ''
            for c in bool_lines[line_num]:
                if c == '?':
                    store += c
                    break
                store += c
            bool_data.append([store[1:].capitalize(), 1])
    with open(general_file, encoding="utf-8") as f:
        bool_lines = f.readlines()
        for line_num in range(len(bool_lines)):
            general_data.append([bool_lines[line_num].strip(), 0])
    data_size = 0
    if (len(bool_data) < len(general_data)):
        data_size = len(bool_data)
    else:
        data_size = len(general_data)

    random.shuffle(general_data)
    random.shuffle(bool_data)

    data = general_data[:data_size] + bool_data[:data_size]
    random.shuffle(data)

    file = os.path.join("..", output, "yesno_dataset.tsv")
    with open(file, "wt", encoding="utf-8") as f:
        for d in data:
            f.write("{}\t{}\n".format(d[0], d[1]))

def create_question(output):
    file = os.path.join("..", output, "train-v2.0.json")
    questions = []
    statements = []
    data = []
    with open(file, encoding="utf-8") as f:
        file_lines = f.readlines()
        more_lines = file_lines
        file_lines = file_lines[0].split('"question"')
        more_lines = more_lines[0].split('"context"')
        for i in range(1, len(more_lines)):
            x = more_lines[i].find('.')
            statements.append([more_lines[i][3:x + 1], 0])
        for i in range(1, len(file_lines)):
            question = ''
            for c in file_lines[i]:
                if c == '?':
                    question += c
                    break
                question += c
            questions.append([question[3:].strip(), 1])
    data_size = 0
    if len(questions) < len(statements):
        data_size = len(questions)
    else:
        data_size = len(statements)

    random.shuffle(questions)
    random.shuffle(statements)

    # Choose percentage for each invalid error function
    div = int(np.round((data_size * .1)))
    for i in range(0, div):
        questions[i][0] = remove_spaces(questions[i][0])
    for i in range(div, div * 2):
        questions[i][0] = switch_punctuation(questions[i][0])
    for i in range(div * 2, div * 3):
        questions[i][0] = add_symbols(questions[i][0])
    for i in range(div * 3, div * 4):
        questions[i][0] = scramble_words(questions[i][0])

    random.shuffle(questions)
    data = questions[:data_size] + statements[:data_size]
    random.shuffle(data)

    # write to file
    file = os.path.join("..", output, "question_dataset.tsv")
    with open(file, "wt", encoding="utf-8") as f:
        for d in data:
            f.write("{}\t{}\n".format(d[0], d[1]))

# Scramble letters in statement
def scramble_words(x):
    words = x.split()
    random.shuffle(words)
    return ' '.join(words)


# Sprinkle in random symbols
def add_symbols(x):
    symbols = ['~', '@', '*', '#', '$', '%']
    new_string = ''
    for i in x:
        add = random.randint(0,1)
        if add == 1:
            s = random.randint(0,5)
            new_string += symbols[s]
        else:
            new_string += i
    return  new_string

# Remove spaces
def remove_spaces(x):
    return x.replace(" ", '')

# 25% without question mark
def switch_punctuation(x):
    temp = x[:-1]
    if x[-1] == '.':
        temp += '?'
    elif x[-1] == '?':
        temp += '.'
    return temp


if __name__ == '__main__':
    output = input("Please enter the name of the folder where the data will be stored: ")
    create_bio(output)
    print("Bio dataset created.")
    create_question(output)
    print("Question dataset created.")
    create_yesno(output)
    print("Yes/no dataset created.")