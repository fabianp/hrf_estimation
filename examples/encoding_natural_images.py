"""
Evaluate the different GLM models in encoding with the "Natural Images dataset".
This script will compute an encoding model and evaluate on left out data.
The correlation between the true BOLD signal and the predicted signal will
be dumped together with the estimated HRF into a file in the directory
results/.

Dependencies:
    scipy (recent, tested with 0.13.2)
    numpy
    scikit-learn
    hrf_estimation (development version)

Examples:

    %run encoding_natural_images.py -m r1glm -b fir -j -1

will run a glm Rank-1 model with a FIR basis. This will create two .npy files
containing the correlation scores and the generated HRFs

The available options are:

    -m : model to use. Available options are:
        glm : GLM with canonical HRF
        glms : GLM with separate designs
        glm_r1 : Rank-1 estimated HRF

    -b: basis to use
        hrf : fixed hrf
        dhrf: hrf + temporal and dispersion derivatives (3 elements)
        fir: FIR basis

    -j : number of cpu to use (-1 means all)

    -s : subect (1 or 2)

    -v : voxels to use (full or best)
       full : perform full brain analysis
       best : use the top 500 voxels

TODO
----
change hrf to dhrf basis

Authors
-------
Fabian Pedregosa <f@bianp.net>
Michael Eickenberg <michael.eickenberg@nsup.org>
"""

import os
import os.path
import numpy as np
from scipy import io, sparse
from sklearn import linear_model, cross_validation
import hrf_estimation as he

#.. constant parameters ..
hrf_length = 30
TR = 1.0
DIR = os.path.dirname(os.path.realpath(__file__))

## parse command line
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-j", "--njobs", dest="n_jobs",
                help="nb of jobs", default=1)
parser.add_option("-s", "--subj", dest="subj",
                help="subject", default=1)
parser.add_option("-m", "--mode", dest="mode",
                help="Mode (glm, r1glm_fir, r1glm_dhrf, glms, r1glms)",
                default='glm')
parser.add_option("-b", "--basis", dest="basis",
                help="Basis (dhrf, fir)",
                default='dhrf')
parser.add_option("-v", "--voxels", dest="voxels",
                help="which voxels to use (best or full)", default="best")
options, args = parser.parse_args()
n_jobs = int(options.n_jobs)

OUTDIR = os.path.join(DIR, 'results')
if options.voxels == 'full':
    OUTDIR = os.path.join(OUTDIR, 'full_brain')
else:
    OUTDIR = os.path.join(OUTDIR, 'top500')
OUTDIR = os.path.join(OUTDIR, 'subj%s' % options.subj)

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

#.. helper functions ..
def compute_correlation(betas, hrfs, scatt_stim, fir_matrix, train, test, test_bold):
    """
    Predict BOLD on a test set from betas, hrf and stimuli and
    return the correlation with test_bold
    """
    # check arguments
    assert betas.shape[1] == hrfs.shape[1]
    # fit ridge on given betas
    ridge = linear_model.RidgeCV()
    beta_preds = ridge.fit(
        scatt_stim[train], betas).predict(scatt_stim[test])
    beta_pred_times_hrf = (beta_preds[:, np.newaxis, :] *
                           hrfs[np.newaxis, :, :]).reshape(-1, test_bold.shape[-1])
    # generate bold on predicted betas
    bold_pred = fir_matrix.dot(beta_pred_times_hrf)
    # normalize prediction
    bold_pred = (bold_pred - bold_pred.mean(0)) / bold_pred.std(0)
    # normalize test bold
    norm_test_bold = (test_bold - test_bold.mean(0)) / test_bold.std(0)
    # compute correlation
    # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#For_a_sample
    corr = (bold_pred * norm_test_bold).sum(0) / \
                (bold_pred.shape[0] - 1.)
    return corr

def get_data(n_sess, full_brain=False):
    """
    Download the data for the current session and subject
    """
    ds = np.DataSource(DIR)
    BASEDIR = 'http://fa.bianp.net/projects/hrf_estimation/data'
    BASEDIR_COMMON = BASEDIR + '/data_common/'
    if full_brain:
        BASEDIR += '/full_brain'
    BASEDIR_SUBJ = BASEDIR + '/data_subj%s/' % options.subj
    event_matrix = io.mmread(ds.open(
        BASEDIR_COMMON + 'event_matrix.mtx')).toarray()
    print('Downloading BOLD signal')
    voxels = np.load(ds.open(
        BASEDIR_SUBJ + 'voxels_%s.npy' % n_sess))
    print('Downloading Scatting Stim')
    scatt_stim = np.load(ds.open(
        BASEDIR_SUBJ + 'scatt_stim_%s.npy' % n_sess))

    ### perform detrending: Savitzky-Golay filter
    n_voxels = voxels.shape[-1]
    voxels = voxels.reshape(-1, 672, n_voxels)
    voxels = voxels - he.savitzky_golay.savgol_filter(voxels, 91, 4, axis=1)
    voxels = voxels.reshape(-1, n_voxels)
    print "Performed Savitzky-Golay"
    voxels = ((voxels - voxels.mean(axis=0)) / voxels.std(axis=0))

    return voxels, scatt_stim, event_matrix

# .. iterate through session ..
for n_sess in range(5):

    ####### Load data for current session ########
    voxels, scatt_stim, event_matrix = get_data(n_sess,
        full_brain=(options.voxels == 'full'))
    em = sparse.coo_matrix(event_matrix)
    fir_matrix = he.utils.convolve_events(event_matrix, np.eye(hrf_length))
    events_train = sparse.block_diag([event_matrix] * 4).toarray()
    conditions_train = sparse.coo_matrix(events_train).col
    onsets_train = sparse.coo_matrix(events_train).row

    # cross-validation folds
    cv_bold = cross_validation.KFold(voxels.shape[0], 5)
    cv = cross_validation.KFold(event_matrix.shape[1] * 5, 5)

    #.. GLM with rank-1 constraint ..
    hrfs = []
    correlation = []
    print('Running %s' % options.mode)
    for n_fold, ((train, test), (train_bold, test_bold)) in enumerate(
            zip(cv, cv_bold)):
        hrfs_train, betas_train = [], []
        train_bold_split = np.split(train_bold, 4)
        conditions_train_split = np.split(conditions_train, 4)
        onsets_train_split = np.split(onsets_train, 4)
        for i in range(4):
            # iterate through runs
            train_bold_i = train_bold_split[i]
            conditions_train_i = conditions_train_split[0]
            onsets_train_i = onsets_train_split[0]

            hrfs_train_i, betas_train_i = he.glm(
                conditions_train_i, onsets_train_i, TR,
                voxels[train_bold_i],
                hrf_length=hrf_length, basis=options.basis,
                verbose=0, n_jobs=n_jobs, maxiter=100,
                mode=options.mode, cache=True)

            hrfs_train.append(hrfs_train_i)
            betas_train.append(betas_train_i)
        if options.mode.startswith('r1'):
            hrfs_train = np.mean(hrfs_train, 0)
            hrfs_train /= hrfs_train.max(0)
        else:
            hrfs_train = np.mean(hrfs_train, 0).mean(1)
            hrfs_train /= hrfs_train.max(0)
        betas_train = np.concatenate(betas_train)

        # encoding step
        corr = compute_correlation(
            betas_train, hrfs_train, scatt_stim, fir_matrix,
            train, test, voxels[test_bold])
        print('Mean Correlation on fold %s: %s' % (n_fold, np.mean(corr)))
        correlation.append(corr)
        hrfs.append(hrfs_train)
    print('Mean correlation on session %s: %s' % (n_sess, np.mean(corr)))

    # .. save the result ..
    np.save(os.path.join(OUTDIR,
        'corr_%s_%s_%s.npy' % (options.mode, options.basis, n_sess)),
            correlation)
    np.save(os.path.join(OUTDIR,
        'hrfs_%s_%s_%s.npy' % (options.mode, options.basis, n_sess)), hrfs)
