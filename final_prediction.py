"""
Rutgers capstone--Team 37
final_prediction.py
This is image level prediction program based on image preprocessing, VGG-16 net loaded with our trained weights, and
a word corrector and LSTM neural network as prediction support to enhance our prediction step.
"""
from VGGprediction import VGGprediction
from LSTM_prediction import lstmprediction
from autocorrect import spell
from preprocess import write_text

IMAGE_NAME = 'IMAGE.jpg'
CNN_string = VGGprediction(IMAGE_NAME)
print(CNN_string)
feed_LSTM = ''
counter = 0
LSTM_string = []
word = ''
different = False
words = ''
lines = []
line_num = 0
for j, i in enumerate(CNN_string):
    if i != 'NULL' and i != 'LINE':
        word += i
        if different == True:
            print('The original CNN preidiction character is: "' + i + ' "\n')
            print('The LSTM neural network\'s prediction based on the last 3 characters is: "' + next_char + '"\n')
        feed_LSTM += i
        LSTM_string.append(i)
        different = False
        counter += 1
        if counter >= 3:
            next_char = lstmprediction(feed_LSTM[-3:])
            try:
                if CNN_string[j+1] == next_char:
                    continue
                else:
                    different = True
            except IndexError:
                break

    elif i == 'LINE':
        print(' The input word is ', word)
        print('The most likely word checked by auto_correct module is "', spell(word), '"')
        words = words + spell(word)
        word = ''
        lines.append(words)
        words = ''
        line_num += 1

    else:
        print(' The input word is ', word)
        print('The most likely word checked by auto_correct module is "', spell(word), '"')
        words = words + spell(word) + ' '
        word = ''

if word:
    print(' Input word is ', word)
    print('The most likely word checked by autocorrect module is "', spell(word), '"')
    words = words + spell(word)
    lines.append(words)
print('The LSTM prediction is: ', LSTM_string)
print("The word corrector's result is: ", lines)

write_text(IMAGE_NAME, lines)
