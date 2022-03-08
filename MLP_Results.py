"""
Plots the results from the neural network fittings
"""


import numpy as np
import pickle as pkl
from glob import glob
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


CLASS  = False


def classify_results(data):
    """classifies the results into pass/fail"""
    x = []
    for val in data:
        x.append(int(round(val[0], 0)))
    return x


def balanced_accuracy(actu, pred):
    """calculates the balanced accuracy"""
    TP, FP, TN, FN, AP, AN = 0, 0, 0, 0, 0, 0
    for i, y in enumerate(actu):
        z = pred[i]
        if y == z and y == 0:
            TN += 1
            AN += 1
        elif y == z:
            TP += 1
            AP += 1
        elif y == 0:
            FP += 1
            AN += 1
        else:
            FN += 1
            AP += 1
    p, n = 100*(TP/AP), 100*(TN/AN)
    #print(100*(TP/AP), 100*(TN/AN), 0.5*(p+n))
    return 0.5*(p+n), TP, FP, TN, FN, AP, AN


def count_existing():
    """counts the number of existing runs"""
    files = glob('NN Files\\NN_Completed_Run*LOOV.txt')
    counts = None
    for file in files:
        run = int(file.split('.txt')[0].split('_Run')[1].split('_LOOV')[0])
        if counts is None:
            counts = run
        elif run > counts:
            counts = run
    if counts is None:
        return 0
    else:
        return counts + 1


def main():
    """main"""
    existing = count_existing()
    print(existing)
    #exit()
    RESULTS = []
    OBS, PRED = [], []
    nans, infs = 0, 0
    #plt.subplot(121)
    for E in range(existing):
        infiles = glob('NN Files\\NewResults_Set%i_LOOV.pkl' % E)
        results, omit = [], False
        for infile in infiles:
            epX, epY, epR, y_test_pred, TestY = pkl.load(open(infile, 'rb'))
            print('*' * 50 + '\n\n', y_test_pred, TestY, '\n\n' + '*' * 50)
            if np.isnan(TestY[0][0]) or np.isnan(y_test_pred[0][0]):
                nans += 1
                continue
            if np.isinf(abs(y_test_pred[0][0])):
                infs += 1
                continue
            OBS.append(TestY[0][0])
            PRED.append(y_test_pred[0][0])
            #if CLASS:
            #    classed = classify_results(y_test_pred)
            #    accs = balanced_accuracy([int(i[0]) for i in TestY], classed)
            #    results.append(accs[0])
            #    #print(accs)
            #else:
            #    r, p = pearsonr(np.array([i[0] for i in TestY]),
            #                    np.array([i[0] for i in y_test_pred]))
            #    print(np.array([i[0] for i in TestY]), np.array([i[0] for i in y_test_pred]))
            #    #print(r**2)
            #    if np.isnan(r):
            #        print('\n' + '='*80)
            #        print('NaN Case:')
            #        print([i[0] for i in y_test_pred])
            #        print('='*80)
            #        omit = True
            #        continue
            #    results.append(r**2)
            #    #plt.scatter(np.array([i[0] for i in TestY]),
            #    #            np.array([i[0] for i in y_test_pred]))
        #if not omit:
        #    print(results)
        #    print(np.mean(results))
        #    RESULTS.append(np.mean(results))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.scatter(OBS, PRED, color='none', edgecolor='b')
    plt.plot([0,100], [0,100], color='k', linestyle='--')
    r, q = pearsonr(OBS, PRED)
    print('N =', len(OBS))
    print('Pearson R2 = %.2f' % r**2)
    plt.title('R$^2$ = %.2f\tN = %i' % (r**2, len(OBS)))
    #plt.subplot(122)
    #print(RESULTS)
    #plt.hist(RESULTS, fc=(0,0,1,0.2), edgecolor='b')
    #plt.xlabel('Pearson R$^2$')
    #plt.ylabel('Number of Models')
    print('NaNs', nans)
    print('Infs', infs)
    plt.show()


if __name__ in '__main__':
    main()
    #input('Code terminated Normally. Press Enter to Close.')
