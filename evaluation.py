########################################################################
# Evaluation
# ==========
# This file contains functions for evaluating the models built.
########################################################################
from config import ONSET_EVAL_COMBINE, ONSET_EVAL_WINDOW, get_config
from keras.models import load_model
from madmom.evaluation.onsets import OnsetEvaluation
from os.path import join
import numpy as np

def sum_evaluation(evals):
    tp = sum(e.num_tp for e in evals)
    fp = sum(e.num_fp for e in evals)
    tn = sum(e.num_tn for e in evals)
    fn = sum(e.num_fn for e in evals)

    # Do some calucations
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f_measure = 2 * prec * rec / (prec + rec)

    ret = 'sum for %d files\n' % len(evals)
    ret += '  #: %6d TP: %6d FP: %5d FN: %5d\n' \
        % (tp + fn, tp, fp, fn)
    ret += '  Prec: %.3f Rec: %.3f F-score: %.3f' \
        % (prec, rec, f_measure)
    return ret

def mean_evaluation(evals):
    n_anns = np.mean([e.num_annotations for e in evals])
    tp = np.mean([e.num_tp for e in evals])
    fn = np.mean([e.num_fn for e in evals])
    fp = np.mean([e.num_fp for e in evals])
    prec = np.mean([e.precision for e in evals])
    recall = np.mean([e.recall for e in evals])
    f_measure = np.mean([e.fmeasure for e in evals])
    ret = 'sum for %d files\n' % len(evals)
    ret += '  #: %5.2f TP: %6.2f FP: %5.2f FN: %5.2f\n' \
        % (n_anns, tp, fp, fn)
    ret += '  Prec: %.3f Rec: %.3f F-score: %.3f' \
        % (prec, recall, f_measure)
    return ret

def evaluate_audio_sample(nn, model, d):
    x = nn.samples_in_audio_sample(d)
    y_guess = model.predict(x)
    y_guess = y_guess.squeeze()
    a_guess = nn.postprocess_y(y_guess)
    return OnsetEvaluation(
        a_guess, d.a,
        combine = ONSET_EVAL_COMBINE,
        window = ONSET_EVAL_WINDOW)

def evaluate_fold(nn, model, fold):
    for audio_sample in fold:
        print(audio_sample.name, end = ' ', flush = True)
        yield evaluate_audio_sample(nn, model, audio_sample)

def evaluate_folds(nn, folds, int_range):
    output_dir = join(get_config()['model-dir'], nn.__name__)
    evals = []
    fmt = '[%d/%d] Evaluating fold, '
    n_folds = len(folds)
    model_file = 'model_best_val.h5'
    for i in int_range:
        print(fmt % (i + 1, n_folds), end = '', flush = True)
        model_path = join(output_dir, '%02d' % i, model_file)
        print('loading %s, ' % model_file, end = '', flush = True)
        model = load_model(model_path)
        fold = folds[i]
        evals.extend(evaluate_fold(nn, model, fold))
        print('DONE')
    return evals

def evaluate(nn, folds, int_range):
    evals = evaluate_folds(nn, folds, int_range)
    print(sum_evaluation(evals))
