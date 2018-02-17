from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pandas as pd
import tensorflow as tf

LABEL_NAME = 'acceptability'
CSV_COLUMN_NAMES = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', LABEL_NAME]
FEATURE_NAMES = CSV_COLUMN_NAMES[:-1]
FEATURE_POSSIBLE_VALUES = [['vhigh', 'high', 'med', 'low'],['vhigh', 'high', 'med', 'low'],['2','3','4','5more'],['2','4','more'],['small', 'med', 'big'],['low', 'med','high']]
FEATURE_NAMES_TO_VALUES = {FEATURE_NAMES[i]:FEATURE_POSSIBLE_VALUES[i] for i in range(0, len(FEATURE_NAMES))}
LABELS = ['unacc', 'acc', 'good', 'v-good']
TRAIN_PATH = "cars_train.csv"
TEST_PATH = "cars_test.csv"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
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
    feature_columns = [tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(key=k, vocabulary_list=FEATURE_NAMES_TO_VALUES[k])) for k in train_features.keys()]

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 10], n_classes=len(LABELS))

    # Train the Model.
    classifier.train(input_fn=lambda:__train_input_fn(train_features, train_labels, args.batch_size), steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(input_fn=lambda:__eval_input_fn(test_features, test_labels, args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = [LABELS[0], LABELS[3]]
    predict_x = {
        FEATURE_NAMES[0]: ['low', 'low'],
        FEATURE_NAMES[1]: ['vhigh', 'med'],
        FEATURE_NAMES[2]: ['3', '2'],
        FEATURE_NAMES[3]: ['2', '4'],
        FEATURE_NAMES[4]: ['med', 'big'],
        FEATURE_NAMES[5]: ['med', 'high'],
    }

    predictions = classifier.predict(input_fn=lambda:__eval_input_fn(predict_x, labels=None, batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(LABELS[class_id],
                              100 * probability, expec))

def __load_data():
    """Returns the cars dataset as (train_features, train_labels), (test_features, test_labels)."""
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
