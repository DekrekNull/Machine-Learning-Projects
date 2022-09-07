# Derek Wright HW 3 - 11766151
import Perceptron as perc

STOP_FILE="./data/fortune-cookie-data/stoplist.txt"
TRAIN_DATA_FILE = "./data/fortune-cookie-data/traindata.txt"
TRAIN_LABEL_FILE = "./data/fortune-cookie-data/trainlabels.txt"
TEST_DATA_FILE = "./data/fortune-cookie-data/testdata.txt"
TEST_LABEL_FILE = "./data/fortune-cookie-data/testlabels.txt"

# Reads the data file to collect all of the available vocab
def get_cookie_vocab(file):
	vocab = []
	with open(file, "r") as f:
		lines = f.readlines()
	f.close()
	for line in lines:
		line = line[:-1]
		line = line.split(' ')
		for word in line:
			if word not in vocab and len(word) > 0:
				vocab.append(word)
	with open(STOP_FILE, 'r') as f:
		lines = f.readlines()
	f.close()
	for line in lines:
		line = line[:-1]
		line = line.split(' ')
		for word in line:
			if word in vocab:
				vocab.remove(word)
	return vocab

# Uses the vocab to make feature vectors for each fortune
def get_cookie_features(m, vocab, file):
	features = []
	with open(file, 'r') as f:
		lines = f.readlines()
	f.close()
	for line in lines:
		feature = [0] * m
		line = line[:-1]
		line = line.split(' ')
		for word in line:
			if word in vocab:
				i = vocab.index(word)
				feature[i] = 1
		features.append(feature)
	return features

# Reads the labels file to get the labels for each feature vector
def get_cookie_labels(file):
	labels = []
	with open(file) as f:
		lines = f.readlines()
		for line in lines:
			line = line[:-1]
			for label in line:
				label = int(label)
				if label == 0:
					label = -1
				labels.append(label)
	return labels

# combines the feature vectors and their respective labels into a tuple
def combine_items(features, labels, items):
	for i in range(0,len(labels) - 1):
		items.append((features[i], labels[i]))
	return items

# Retrieves all of the data from their respective files and stores them in lists
def cookie_get_data(trainData, testData):
	vocab = get_cookie_vocab(TRAIN_DATA_FILE)
	vocab.sort()
	m = len(vocab)
	features = get_cookie_features(m, vocab, TRAIN_DATA_FILE)
	labels = get_cookie_labels(TRAIN_LABEL_FILE)
	combine_items(features, labels, trainData)
	features = get_cookie_features(m, vocab, TEST_DATA_FILE)
	labels = get_cookie_labels(TEST_LABEL_FILE)
	combine_items(features, labels, testData)
	return m

# Main entry point for the cookie classifier
def cookie_classifier():
	trainData = []
	testData = []
	m = cookie_get_data(trainData, testData)
	perc.classifier(trainData, testData, m, "Part 1: Fortune Cookie Classifier")