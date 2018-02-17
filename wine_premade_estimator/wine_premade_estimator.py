from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pandas as pd
import tensorflow as tf

CSV_COLUMN_NAMES = ['Alcohol', 'MalicAcid', 'Ash', 'AshAlcalinity', 'Magnesium', 'TotalPhenols', 'Flavanoids', 'NonflavanoidPhenols', 'Proanthocyanins', 'ColorIntensity', 'Hue', 'OD280/OD315', 'Proline', 'Variety']
LABEL_NAME = 'Variety'
VARIETIES = ['variety_1', 'variety_2', 'variety_3']
TRAIN_PATH = "wines_train.csv"
TEST_PATH = "wines_test.csv"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    # Assumes dataset has been generated using to_csv.py
    # ...
    # Get args
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_features, train_labels), (test_features, test_labels) = __load_data()

    # Feature columns describe how to use the input.
    feature_columns = [tf.feature_column.numeric_column(key=k) for k in train_features.keys()]

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        n_classes=3)

    # Train the Model.
    classifier.train(
        input_fn=lambda:__train_input_fn(train_features, train_labels, args.batch_size),steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(input_fn=lambda:__eval_input_fn(test_features, test_labels, args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['variety_1', 'variety_2']
    predict_x = {
        'Alcohol': [14.23, 12.37],
        'MalicAcid': [1.71, 1.07],
        'Ash': [2.43, 2.1],
        'AshAlcalinity': [15.6, 18.5],
        'Magnesium': [127, 88],
        'TotalPhenols': [2.8, 3.52],
        'Flavanoids': [3.06, 3.75],
        'NonflavanoidPhenols': [0.28, 0.24],
        'Proanthocyanins': [2.29, 1.95],
        'ColorIntensity': [5.64, 4.5],
        'Hue': [1.04, 1.04],
        'OD280/OD315': [3.92, 2.77],
        'Proline': [1065, 660],
    }

    predictions = classifier.predict(input_fn=lambda:__eval_input_fn(predict_x, labels=None, batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(VARIETIES[class_id],
                              100 * probability, expec))

def __load_data():
    """Returns the wine dataset as (train_features, train_labels), (test_features, test_labels)."""
    train = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
    train_features, train_labels = train, train.pop(LABEL_NAME)

    test = pd.read_csv(TEST_PATH, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_labels = test, test.pop(LABEL_NAME)

    return (train_features, train_labels), (test_features, test_labels)

def __train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

def __eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
