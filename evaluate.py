from run_system import *
import os

def evaluate_question(file_name):
    evaluate(file_name, determine_question)

def evaluate_yesno(file_name):
    evaluate(file_name, determine_yesno)

def evaluate_bio(file_name):
    evaluate(file_name, determine_bio)

def evaluate(file_name, function):
    file = "../qa_data/" + file_name
    print("Evaluating {}...".format(file_name))
    with open(file, encoding="utf-8") as f:
        file_lines = f.readlines()
        total = len(file_lines)
        correct = 0
        false_positive = 0
        true_positive = 0
        false_negative = 0

        for line in file_lines:
            line_split = line.split("\t")
            correct_answer = int(line_split[1].strip())
            determined_answer = function(line_split[0])
            if (determined_answer == correct_answer):
                correct += 1
            if (correct_answer == 1 and determined_answer == 1 ):
                true_positive += 1
            if (correct_answer == 1 and determined_answer == 0 ):
                false_positive += 1
            if (correct_answer == 0 and determined_answer == 0 ):
                false_negative += 1

        accuracy = correct/total
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = (2 * precision * recall)/ (precision + recall)
        print("Evaluation Completed.\n\nAccuracy: {}\nPrecision: {}\nRecall: {}\nFl: {}\n".format(
            accuracy, precision, recall, f1
        ))

def evaluate_nohup_output():
    # file = input("Please enter the name of the file: ")
    file = "nohup_output"
    filepath = os.path.join("..", "nohup_output", file+".txt")
    data = ''
    with open(filepath) as f:
        f_lines = f.readlines()
        for line in f_lines:
            if "leaning" in line or "Testing" in line:
                data += "\n\n"
                data += line
            if "Epoch" in line:
                data += line
            if "f1_score" in line:
                divide = (line.split("-"))
                remove = divide[11].split(":")
                precision = divide[9].split(":")
                precision = float(precision[1].strip())
                recall = divide[10].split(":")
                recall = float(recall[1].strip())
                if precision > 0 or recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    divide[11] = remove[0] + ": " + str("%.4f" % round(f1, 4))
                else:
                    divide[11] = remove[0] + ": N/A"

                data += ''.join(divide)
    filepath = os.path.join("..", "nohup_output", file+"_data.txt")
    with open(filepath, "wt") as f:
        f.write(data)



if __name__ == '__main__':
    evaluate_nohup_output()
