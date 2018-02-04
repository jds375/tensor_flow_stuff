def to_csv_format():
    wines_in_csv = open('wines.csv', 'w');
    wines_in_csv.write('178,13,variety_1,variety_2,variety_3\n')
    wines_data = open('raw_wine_data', 'r')
    for line in wines_data:
	winedata = __parse_line_to_winedata(line)
	wines_in_csv.write(winedata.to_csv_line() + '\n')
    wines_in_csv.close()
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
