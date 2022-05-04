import os
import random

def format_squad(output):
    file = os.path.join("..", output, "train-v2.0.json")
    questions = []
    with open(file, encoding="utf-8") as f:
        file_lines = f.readlines()
        file_lines = file_lines[0].split('"question"')
        for i in range(1, len(file_lines)):
            question = ''
            for c in file_lines[i]:
                if c == '?':
                    question += c
                    break
                question += c
            questions.append(question[3:].strip())

    file = os.path.join("..", output, "squad_data.tsv")
    with open(file, "wt", encoding="utf-8") as f:
        for d in questions:
            f.write("{}\n".format(d))
    print("SQuAD Dataset formatted.")

def format_bioASQ(output):
    file = os.path.join("..", output, "training8b.json")
    qa_data = []
    full_data = []
    with open(file) as f:
        file_lines = f.readlines()
        flag = False
        for line_num in range(len(file_lines)):
            if '"body":' in file_lines[line_num]:
                question = line_num
                full_data.append(file_lines[question].strip()[9:-2])
            elif '"exact_answer":' in file_lines[line_num]:
                answer = line_num
            elif '"type":' in file_lines[line_num]:
                if '"yesno"' in file_lines[line_num]:
                    if file_lines[answer].strip()[17:-2] == "yes":
                        answer = 1
                    elif file_lines[answer].strip()[17:-2] == "no":
                        answer = 0
                    else:
                        answer = -1
                    qa_data.append(file_lines[question].strip()[9:-2] + "|" + str(answer))
                    flag = True
            elif flag and '"text":' in file_lines[line_num]:
                flag = False
                qa_data[len(qa_data) - 1] += "|" + file_lines[line_num].strip()[9:-2]

    file = os.path.join("..", output, "bioasq_data.tsv")
    with open(file, "wt") as f:
        for data in qa_data:
            data = data.split("|")
            f.write("{} {}\t{}\n".format(data[0], data[2], data[1]))

    file = os.path.join("..", output, "bioasq_data_full.tsv")
    with open(file, "wt") as f:
        for data in full_data:
            f.write("{}\n".format(data))

    file = os.path.join("..", output, "bioasq_split.tsv")
    with open(file, "wt") as f:
        for data in qa_data:
            data = data.split("|")
            f.write("{}\t{}\t{}\n".format(data[0], data[2], data[1]))
    print("BioASQ Dataset formatted.")

def format_boolQ(output):
    file = os.path.join("..", output, "train.jsonl")
    qa_data = []
    yes_count = []
    no_count = []
    yes = False
    no = False
    with open(file, encoding="utf-8") as f:
        file_lines = f.readlines()
        flag = False
        for line_num in range(len(file_lines)):
            try:
                slice1 = file_lines[line_num].split('"question":')
                slice2 = slice1[1].split('"title":')
                slice3 = slice2[1].split('"answer":')
                slice4 = slice3[1].split('"passage":')
                question = slice2[0][2:-3] + '?'
                answer = slice4[0][1:-2]
                if answer == 'true':
                    answer = 1
                    yes = True
                    no = False
                elif answer == 'false':
                    answer = 0
                    no = True
                    yes = False
                passage = slice4[1][2:-3]
                qa_data.append("{}|{}|{}".format(question, str(answer), passage))
                if answer == 1:
                    yes_count.append("{}|{}|{}".format(question, str(answer), passage))
                elif answer == 0:
                    no_count.append("{}|{}|{}".format(question, str(answer), passage))
            except:
                print("error detected")

    equal_size = 0
    if len(yes_count) < len(no_count):
        equal_size = len(yes_count)
    else:
        equal_size = len(no_count)

    random.shuffle(yes_count)
    random.shuffle(no_count)

    balanced_data = yes_count[:equal_size] + no_count[:equal_size]
    random.shuffle(balanced_data)

    file = os.path.join("..", output, "boolq_data.tsv")
    with open(file, "wt", encoding="utf-8") as f:
        for data in qa_data:
            data = data.split("|")
            f.write("{} {}\t{}\n".format(data[0], data[2], data[1]))

    file = os.path.join("..", output, "balanced_boolq.tsv")
    with open(file, "wt", encoding="utf-8") as f:
        for data in balanced_data:
            data = data.split("|")
            f.write("{} {}\t{}\n".format(data[0], data[2], data[1]))

    file = os.path.join("..", output, "balanced_boolq_split.tsv")
    with open(file, "wt", encoding="utf-8") as f:
        for data in balanced_data:
            data = data.split("|")
            f.write("{}\t{}\t{}\n".format(data[0], data[2], data[1]))
    print("BoolQ Dataset formatted.")

if __name__ == '__main__':
    output = input("Please enter the name of the folder where the data will be stored: ")
    format_squad(output)
    format_boolQ(output)
    format_bioASQ(output)
