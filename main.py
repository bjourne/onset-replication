from argparse import ArgumentParser
from config import get_config
from glob import glob
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import load_model
from os.path import join
from sys import exit
from evaluation import evaluate
from preprocessing import load_data

########################################################################
# Training
########################################################################
def cv_train_fold(nn, parts, test_idx, epochs):
    model_dir = get_config()['model-dir']
    fold_dir = join(model_dir, nn.__name__, '%02d' % test_idx)
    n = len(parts)
    val_idx = (test_idx + 1) % n

    # Create three sequences derived from keras.utils.Sequence
    test = nn.ArchSequence(parts[test_idx])
    val = nn.ArchSequence(parts[val_idx])
    test_parts = [p for i, p in enumerate(parts)
                  if i not in (test_idx, val_idx)]
    assert len(test_parts) == 6
    train = nn.ArchSequence([d for p in test_parts for d in p])

    verbose = 1
    mca = ModelCheckpoint(join(fold_dir, 'model_{epoch:03d}.h5'),
                          monitor = 'loss',
                          save_best_only = False)
    mcb = ModelCheckpoint(join(fold_dir, 'model_best.h5'),
                          monitor = 'loss',
                          save_best_only = True)
    mcv = ModelCheckpoint(join(fold_dir, 'model_best_val.h5'),
                          monitor = 'val_loss',
                          save_best_only = True)
    es = EarlyStopping(monitor = 'val_loss',
                       min_delta = 1e-4, patience = 20,
                       verbose = verbose)
    tb = TensorBoard(log_dir = join(fold_dir, 'logs'),
                     write_graph = True, write_images = True)
    callbacks = [mca, mcb, mcv, es, tb]

    files = sorted(glob(join(fold_dir, 'model_???.h5')))
    if files:
        model_file = files[-1]
        initial_epoch = int(model_file[-6:-3])
        print('* Resuming training using %s' % model_file)
        model = load_model(model_file)
    else:
        model = nn.model()
        initial_epoch = 0
    model.fit_generator(train,
                        steps_per_epoch = len(train),
                        initial_epoch = initial_epoch,
                        epochs = epochs,
                        shuffle = True,
                        validation_data = val,
                        validation_steps = len(val),
                        callbacks = callbacks)

def train(nn, folds, int_range, epochs):
    for i in int_range:
        cv_train_fold(nn, folds, i, epochs)

########################################################################
# Argument parsing
########################################################################
def int_range(string):
    s1, s2 = string.split(':')
    return range(int(s1), int(s2))

def main():
    parser = ArgumentParser(description = 'Onset Detection Trainer')
    parser.add_argument(
        '--network-type', '-n',
        required = True,
        choices = ['cnn', 'rnn'],
        help = 'network type')
    parser.add_argument(
        '--epochs',
        type = int,
        default = 100,
        help = 'number of epochs to train each model for')
    req_action = parser.add_mutually_exclusive_group(required = True)
    req_action.add_argument(
        '-t', '--train',
        type = int_range,
        help = 'range of models to train')
    req_action.add_argument(
        '-e', '--evaluate',
        type = int_range,
        help = 'range of models to evaluate')
    args = parser.parse_args()

    # Load the eight folds
    nn, folds = load_data(args.network_type)
    print('* Created folds with sizes %s.' % list(map(len, folds)))

    if args.evaluate:
        evaluate(nn, folds, args.evaluate)
    else:
        train(nn, folds, args.train, args.epochs)

if __name__ == '__main__':
    main()
