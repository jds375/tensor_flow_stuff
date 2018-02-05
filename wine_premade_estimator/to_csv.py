import random

def to_csv_format():
    """
    Converts the raw data into a training and test csv with a 75 25 ratio
    """
    training_wines_in_csv = open('wines_train.csv', 'w');
    testing_wines_in_csv = open('wines_test.csv', 'w');
    training_wines_in_csv.write('178,13,variety_1,variety_2,variety_3\n')
    testing_wines_in_csv.write('178,13,variety_1,variety_2,variety_3\n')
    wines_data = open('raw_wine_data', 'r')
    lines_to_write = []
    for line in wines_data:
        winedata = __parse_line_to_winedata(line)
        lines_to_write.append(winedata.to_csv_line() + '\n')
    random.shuffle(lines_to_write)
    count = 0
    for line_to_write in lines_to_write:
        if (count < 178 * 0.75):
            training_wines_in_csv.write(line_to_write)
        else:
            testing_wines_in_csv.write(line_to_write)
        count = count + 1
    training_wines_in_csv.close()
    testing_wines_in_csv.close()
    wines_data.close()

def __parse_line_to_winedata(line):
    # The label is the first and the rest are features
    label = None
    features = []
    for value in line.replace('\n','').split(','):
	if label is None:
	    label = value
	else:
	    features.append(value)
    return __WineData(features, label)

class __WineData:

    def __init__(self, features, label):
	self.features = features
	self.label = label

    def to_csv_line(self):
	features_portion = reduce(lambda accum_vals, feature: accum_vals + ',' + feature, self.features)
	return features_portion + ',' + self.label

if __name__ == '__main__':
    to_csv_format()
