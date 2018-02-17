import random

TOTAL_EXAMPLES = 1728
TEST_TO_TRAIN_SIZE_RATIO = 0.75
NUM_FEATURES = 6
LABELS = ['unacc', 'acc', 'good', 'vgood']

def to_csv_format():
    """
    Converts the raw data into a training and test csv with a similar ratio
    """
    training_cars_in_csv = open('cars_train.csv', 'w');
    testing_cars_in_csv = open('cars_test.csv', 'w');
    training_cars_in_csv.write(','.join([str(TOTAL_EXAMPLES), str(NUM_FEATURES), LABELS[0], LABELS[1], LABELS[2], LABELS[3] + '\n']))
    testing_cars_in_csv.write(','.join([str(TOTAL_EXAMPLES), str(NUM_FEATURES), LABELS[0], LABELS[1], LABELS[2], LABELS[3] + '\n']))
    cars_data = open('raw_cars_data', 'r')
    lines_to_write = []
    for line in cars_data:
        cardata = __parse_line_to_cardata(line)
        lines_to_write.append(cardata.to_csv_line() + '\n')
    random.shuffle(lines_to_write)
    count = 0
    for line_to_write in lines_to_write:
        if (count < TOTAL_EXAMPLES * TEST_TO_TRAIN_SIZE_RATIO):
            training_cars_in_csv.write(line_to_write)
        else:
            testing_cars_in_csv.write(line_to_write)
        count = count + 1
    training_cars_in_csv.close()
    testing_cars_in_csv.close()
    cars_data.close()

def __parse_line_to_cardata(line):
    # The label is the last and the first are features
    label = None
    features = []
    for value in line.replace('\n','').split(','):
    	if len(features) is NUM_FEATURES:
    	    label = str(LABELS.index(value))
    	else:
    	    features.append(value)
    return __CarData(features, label)

class __CarData:

    def __init__(self, features, label):
	    self.features = features
	    self.label = label

    def to_csv_line(self):
	    features_portion = reduce(lambda accum_vals, feature: accum_vals + ',' + feature, self.features)
	    return features_portion + ',' + self.label

if __name__ == '__main__':
    to_csv_format()
