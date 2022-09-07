import Perceptron as perc

TRAIN_FILE = "./data/OCR-data/ocr_train.txt"
TEST_FILE = "./data/OCR-data/ocr_test.txt"

# determines if c is a value
# returns 1 if it is, -1 if not
def is_vowel(c):
    vowels = ['a', 'e', 'i', 'o', 'u']
    if c in vowels:
        return 1
    return -1

# Parses the OCR data from the given file into a list of tuples in the form:
# (feature vector, label)
def get_data(in_file, dataList):
    with open(in_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 128:
                i = line.find('i')
                line = line[i + 2:135]
                line = line.strip()
                line = line.replace("\t", "")
                labelChar = line[-1]
                line = line[0:-1]
                data = []
                for c in line:
                    data.append(int(c))
                dataList.append((data, is_vowel(labelChar)))

# Retrieves the test and train data for the OCR into lists from their repective files
def OCR_get_data(testData, trainData):
    get_data(TEST_FILE, testData)
    get_data(TRAIN_FILE, trainData)

# Entry method for the OCR classifier
def OCR_classifier():
    trainData = []
    testData = []
    OCR_get_data(trainData, testData)
    perc.classifier(trainData, testData, 128, "\nPart 2: OCR")