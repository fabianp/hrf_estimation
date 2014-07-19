import numpy as np
from scipy import io, sparse
import tempfile

# local imports
from . import utils, savitzky_golay

HRF_LENGTH = 20

def get_sample_data(n_sess, full_brain=False, subj=1):
    """
    Download the data for the current session and subject

    Parameters
    ----------
    n_sess: int
        number of session, one of {0, 1, 2, 3, 4}
    subj: int 
        number of subject, one of {1, 2}
    """
    DIR = tempfile.mkdtemp()
    ds = np.DataSource(DIR)
    BASEDIR = 'http://fa.bianp.net/projects/hrf_estimation/data'
    BASEDIR_COMMON = BASEDIR + '/data_common/'
    if full_brain:
        BASEDIR += '/full_brain'
    BASEDIR_SUBJ = BASEDIR + '/data_subj%s/' % subj
    event_matrix = io.mmread(ds.open(
        BASEDIR_COMMON + 'event_matrix.mtx')).toarray()
    print('Downloading BOLD signal')
    voxels = np.load(ds.open(
        BASEDIR_SUBJ + 'voxels_%s.npy' % n_sess))
    # print('Downloading Scatting Stim')
    # scatt_stim = np.load(ds.open(
    #     BASEDIR_SUBJ + 'scatt_stim_%s.npy' % n_sess))

    em = sparse.coo_matrix(event_matrix)
    fir_matrix = utils.convolve_events(event_matrix, np.eye(HRF_LENGTH))
    events_train = sparse.block_diag([event_matrix] * 5).toarray()
    conditions_train = sparse.coo_matrix(events_train).col
    onsets_train = sparse.coo_matrix(events_train).row

    return voxels, conditions_train, onsets_train