OUT_FILE ="./data/output.txt"

# Takes a decimal value and converts it to a percentage as a string
def to_percent_str(num):
	num = num * 100
	num = round(num, 2)
	return "%" + str(num)

# Writes results to part a to the outfile
def dump_part_a(l, partHeader):
	with open(OUT_FILE, "a") as text_file:
		i = 1
		text_file.write(partHeader + "\n\n" + "(A)\n")
		for item in l:
			text_file.write("\tit-" + str(i) + "\t| mistakes: " + str(item))
			i += 1
			text_file.write('\n')
	text_file.close()

# Writes results to part b to the outfile
def dump_part_b(test, train):
	with open(OUT_FILE, "a") as text_file:
		text_file.write("(B)\n")
		for i in range (0, len(train)):
			text_file.write("\tit-" 
							+ str(i + 1) 
							+ "\t| training-accuracy: " 
							+ to_percent_str(train[i])
							+ " \n\t\t| testing-accuracy:  " 
							+ to_percent_str(test[i]))
			text_file.write('\n')
	text_file.close()

# Writes results to part c to the outfile
def dump_part_c(testStandard, trainStandard, testAveraged, trainAveraged):
	with open(OUT_FILE, "a") as text_file:
		text_file.write("(C)\n")
		text_file.write("\ttraining-accuracy-standard-perceptron: " 
						+ to_percent_str(trainStandard[-1])
						+ "\n\ttraining-accuracy-averaged-perceptron: "
						+ to_percent_str(trainAveraged[-1]) + '\n')
		text_file.write("\ttesting-accuracy-standard-perceptron:  "
						+ to_percent_str(testStandard[-1])
						+ "\n\ttesting-accuracy-averaged-perceptron:  "
						+ to_percent_str(testAveraged[-1]) + '\n')
	text_file.close()

# Calculates the dot product of the feature and weight value and returns the sign as 1 or -1
def get_y(w, feature):
	dot = 0
	for i in range (0, len(w) - 1):
		dot = dot + (w[i] * feature[i])
	if(dot > 0):
		return 1
	return -1

# Runs a single round of a standard perceptron
def standard_round(items, w, mistakes, train):
	count = 0
	for item in items:
		y = get_y(w, item[0])
		if y != item[1]:
			count += 1
			if (train):
				for j in range(0, len(w) - 1):
					w[j] = w[j] + (item[0][j] * item[1])
	mistakes.append(count)
	return w

# Averages the wieght vectors
def averaged(weights):
	n = len(weights)
	m = len(weights[0])
	total = [0] * m
	for w in weights:
		for i in range(0, m - 1):
			total[i] += w[i]
	for i in range(0, m - 1):
		total[i] = total[i] / n
	return total

# Runs a single round of an averaged perceptron
def averaged_round(items, w, mistakes, train):
	count = 0
	weights = [w]
	for item in items:
		y = get_y(w, item[0])
		if y != item[1]:
			count += 1
			if (train):
				for j in range(0, len(w) - 1):
					w[j] = w[j] + (item[0][j] * item[1])
		weights.append(w)
	mistakes.append(count)
	return averaged(weights)

# Runs the data through an averaged perceptron 20 times, gathering the required results
def classifier_averaged(trainItems, testItems, trainMistakes, testMistakes, m):
	w = [0] * m
	for i in range(0,20):
		w = averaged_round(trainItems, w, trainMistakes, True)
		w = averaged_round(testItems, w, testMistakes, False)
	testAccuracies = []
	trainAccuracies = []
	for mistakes in testMistakes:
		testAccuracies.append((len(testItems) - mistakes) / len(testItems))
	for mistakes in trainMistakes:
		trainAccuracies.append((len(trainItems) - mistakes) / len(trainItems))
	return (trainAccuracies, testAccuracies)

# Runs the data through a standard perceptron 20 times, gathering the required results
def classifier_standard(trainItems, testItems, trainMistakes, testMistakes, m):
	w = [0] * m
	for i in range(0,20):
		w = standard_round(trainItems, w, trainMistakes, True)
		w = standard_round(testItems, w, testMistakes, False)
	testAccuracies = []
	trainAccuracies = []
	for mistakes in testMistakes:
		testAccuracies.append((len(testItems) - mistakes) / len(testItems))
	for mistakes in trainMistakes:
		trainAccuracies.append((len(trainItems) - mistakes) / len(trainItems))
	return (trainAccuracies, testAccuracies)

# Runs the given traindata set through the perceptron to change the weight, 
# then runs the through the testdata after each round.
# Writes the required results to the outfile.
def classifier(trainData, testData, m, partHeader):
	trainMistakesStandard = []
	testMistakesStandard = []
	(trainAccuraciesStandard, testAccuraciesStandard) = classifier_standard(trainData, testData, trainMistakesStandard, testMistakesStandard, m)
	dump_part_a(trainMistakesStandard, partHeader)
	dump_part_b(testAccuraciesStandard, trainAccuraciesStandard)
	trainMistakesAveraged = []
	testMistakesAveraged = []
	(trainAccuraciesAveraged, testAccuraciesAveraged) = classifier_averaged(trainData, testData, trainMistakesAveraged, testMistakesAveraged, m)
	dump_part_c(testAccuraciesStandard, trainAccuraciesStandard, testAccuraciesAveraged, trainAccuraciesAveraged)