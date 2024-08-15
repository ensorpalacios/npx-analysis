""" Code to process output of DeepLabCut
Attention: now possible to select whisker data based on time
of CNO dropping; can select either before or after drop time"""

# Import modules & packages

from pathlib import Path
import numpy as np
import pandas as pd

# for phase extraction
from scipy.signal import buttord, butter, filtfilt
# from scipy.signal import hilbert
from scipy import stats
import pdb
import os


def load_wdata(h5dir, t_drop=False, pre=True, sr=299):
    """ Load dlc h5 file about whiskers; plus, save original
    recording length (starting from 0 sec), and trim data
    based on CNO dropping (if t_drop is not False)"""
    # Get file with h5 extension if not present
    if h5dir[-2:] != 'h5':
        for file in os.listdir(h5dir):
            if file.endswith('h5'):
                h5dir = os.path.join(h5dir, file)

    full_path = Path(h5dir)  # get path
    df = pd.read_hdf(full_path)  # get dataframe

    # Save recording length if trimming data
    # -1 because first frame starts at 0
    tot_wskdlen = (len(df) - 1) / 299

    if t_drop is not False:
        toframe = 60 * sr
        if pre:
            df = df.iloc[:t_drop * toframe]
        else:
            # df = df.iloc[t_drop * toframe:]
            df = df.iloc[(t_drop + 5) * toframe:(t_drop * 2 + 5) * toframe]


    scorer = df.columns.get_level_values(0)[
        0]  # df is MultiIndex with levels scorer,bodyparts,coords
    df = df[scorer]  # descend one level (ignore scorer)
    bodyparts = df.columns.get_level_values(0).unique()  # save bodypart labels
    # df = df.drop('likelihood',axis=1,level=1) # drop likelihood
    return df, bodyparts, tot_wskdlen

    # take in account likelihood - to add!!!!!!!!!!!!!!!!!!!!


def calc_angle(df, bodyparts):
    """ Calculate raw angles """
    # Basal marker for each whisker - slice, select column x, convert to numpy array
    x_one = df[bodyparts[::4]].xs('x', axis=1, level=1).to_numpy()
    y_one = df[bodyparts[::4]].xs('y', axis=1, level=1).to_numpy()
    # next closer marker for each whisker
    x_two = df[bodyparts[1::4]].xs('x', axis=1, level=1).to_numpy()
    y_two = df[bodyparts[1::4]].xs('y', axis=1, level=1).to_numpy()

    # calculate angle between basal and second marker
    # if a and b > 0 angle θ in 1st quadrant
    # if a > 0 and b < 0 then 4th quadrant
    # if a < 0 and b > 0 then 2nd quadrant
    a = y_one - y_two  # r*sin(angle)
    b = x_one - x_two  # r*cos(angle)

    # Correct for possible 0 denominator
    __ = np.argwhere(b == 0)
    b[__] = 0.0000001

    angle_trace = np.arctan(
        a / b
    )  # (reflected) angle in rad; 0 is horizonal line based on camera view
    # Correct for 2nd quadrant (b < 0) - only anterior whisker!
    for wsk in range(3):
        __ = np.argwhere(x_one[:, wsk] - x_two[:, wsk] < 0)[:, 0]
        angle_trace[__, wsk] = np.pi + np.arctan(a[__, wsk] / b[__, wsk])

    # _, axs = plt.subplots(2,1, sharex=True)
    # axs[0].plot(angle_trace[:, 2])
    # axs[1].plot(x_one[:, 2])
    # plt.show()
    return angle_trace  # np.array (nf*nw)


def calc_phase(angle_trace, sr=299, bp=[6, 30]):
    """ Computationally, the Hilbert Transform is the Fourier
    % Transform with zero amplitude at all negative frequencies.  This
    % is equivalent to phase-shifting the time-domain signal by 90 degrees
    % at all frequencies and then adding this as an imaginary signal to the
    % original signal.
    % So for example, the signal cos(t) becomes cos(t) + i sin(t).
    %
    %The phase of the original signal is taken as the angle of this new
    %complex signal.
    %
    %inputs -
    %    signal - vector containing whisker angle
    %    Sr   - sampling rate (Hz)
    %    bp   - frequency range for band-pass filtering, defaults= [6, 30] as
    %           used in Chen, S., Augustine, G. J., & Chadderton, P. (2016)
    %
    %outputs -
    %    phase  - phase estimate from Hilbert transform
    %    filtered_signal - input signal after filtering

    Script adapted by Marie T. from Hill et al, Neuron, 2011
    """
    n_frame, n_whisk = angle_trace.shape  # number frames and whiskers

    # de-trend the signal with a band-pass filter
    # scipy.signal.buttord(wp, ws, gpass, gstop, analog=False)
    bp = np.array(bp) * 2 / sr  # convert Hz to radians/S
    [N, Wn] = buttord(bp, bp * [.5, 1.5], 3, 20)
    [B, A] = butter(N, Wn, btype='bandpass')

    # zero-phase filtering, emulate matlab's behaviour with odd padding
    filtered_sig = np.empty([n_frame, n_whisk])
    for w in range(n_whisk):
        filtered_sig[:, w] = filtfilt(B,
                                      A,
                                      angle_trace[:, w],
                                      padtype='odd',
                                      padlen=3 * (max(len(B), len(A)) - 1))

    # remove negative frequency component of Fourier transform
    ht_signal = np.empty([n_frame, n_whisk], dtype='complex')
    for w in range(n_whisk):
        X = np.fft.fft(filtered_sig[:, w])
        halfway = int(1 + np.ceil(len(X) / 2))
        X[halfway:] = 0
        ht_signal[:, w] = np.fft.ifft(X)

    phase_trace = np.angle(ht_signal)

    # # Using scipy function (doubling of positive frequences; not done above!)
    # ht_signal = np.empty([t_step,whisker],dtype='complex')
    # for w in range(whisker):
    #     ht_signal[:,w] = hilbert(filtered_sig[:,w])

    # phase = np.empty([t_step,whisker])
    # for w in range(whisker):
    #     phase[:,w] = np.angle(ht_signal[:,w])

    return ht_signal, phase_trace, filtered_sig  # nf*nw


def get_slow_var(angle_trace, phase_trace, operation):  # 1d vectors
    """ Use the phase to find the turning points of the whisks (tops and
    bottoms), and calculate a value on each consecutive whisk using the
    function handle (operation).  The values are calculated twice per
    whisk cycle using both bottom-to-bottom and top-to-top.  The values
    are linearly interpolated between whisks.
    Python code for decomposing a whisking bout into phase, amplitude, and
    offset using the Hilbert Transform.
    This code was based on code developed by the Neurophysics Lab at UCSD.
    Primary motor cortex reports efferent control of vibrissa motion on
    multiple timescales DN Hill, JC Curtis, JD Moore, D Kleinfeld - Neuron,
    2011"""
    n_frame = len(angle_trace)  # number frames (single whisker)

    # Find crossings
    tops = (phase_trace[0:-1] < 0) & (phase_trace[1:] >= 0)
    bottoms = (phase_trace[:-1] >= np.pi / 2) & (phase_trace[1:] <= -np.pi / 2)
    out = np.zeros(n_frame)
    # out = np.zeros((len(sig),))

    # Evaluate at transitions
    temp = []
    pos = []
    # inx = [i for i, x in enumerate(tops) if x]
    inx = np.nonzero(tops)[0]  # idx frame turning point
    for j in range(1, len(inx)):
        vals = angle_trace[inx[j - 1]:inx[j]]
        temp.append(operation(
            vals))  # function over trace between consecutive turning points

    if len(inx) > 1:
        pos = np.round(inx[0:-1] + np.diff(inx) / 2)  # middle frame

    # inxb = [i for i, x in enumerate(bottoms) if x]
    inxb = np.nonzero(bottoms)[0]
    for j in range(1, len(inxb)):
        vals = angle_trace[inxb[j - 1]:inxb[j]]
        temp.append(operation(vals))

    if len(inxb) > 1:
        pos = np.append(pos, np.round(inxb[:-1] + np.diff(inxb) / 2))

    # Sort
    i = np.argsort(pos)  # sorted index
    posa = pos[i]  # reorder
    pos = np.concatenate([np.array([0]), posa, [n_frame]])  # add extremities

    if not temp:
        temp = operation(angle_trace) * [1, 1]  # ?
    else:
        temp = np.array(temp)
        temp = np.concatenate([[temp[i[0]]], temp[i], [
            temp[i[-1]]
        ]])  # sort temp (i) and fill extremities (copied values)

    # make piecewise linear signal (interpolation)
    for j in np.arange(1, len(pos)):
        ins = np.arange(int(pos[j - 1]), int(pos[j]))
        out[ins] = np.linspace(temp[j - 1], temp[j], len(ins))

    return out, inx, inxb  # nf*1
    # return (out,tops,bottoms)


"""
Adapted from Marie Tolkiehn script:

Created on Wed Feb 26 2020
@author: Marie Tolkiehn
This script loads the whisking data and extracts the parts of the recording where whisking was detected.
It was previously shown that cerebellum linearly encodes whisker position during voluntary movement.
Chen, S., Augustine, G. J., & Chadderton, P. (2016). The cerebellum linearly encodes whisker position
during voluntary movement. ELife, 1–16. https://doi.org/10.7554/eLife.10509
Whisking parameter extraction is based on the script accompanies the Primer "Analysis of Neuronal Spike
Trains, Deconstructed", by J. Aljadeff, B.J. Lansdell, A.L. Fairhall and D. Kleinfeld (2016) Neuron,
91 link to manuscript: http://dx.doi.org/10.1016/j.neuron.2016.05.039
This script takes as input the filepath to the whisking data.
The input data consist of traces were generated by Nathan Clack's automated tracking of single rows of
whiskers in high-speed video Janelia https://wiki.janelia.org/wiki/display/MyersLab/Whisker+Tracking
The (huge) output files from the automated tracking were processed in Matlab using Janelia's available
code and converted to .npy using custom matlab function conv_meas_py.m available on github. The .npy
contain three columns [FID WhiskerLabels Angles], where FID corresponds to frame ID, whisker labels to
the whisker ID (0, 1, 2, ...) and the Angles to the measured angle of traced whiskers in degrees. Angles
are wrapped to 0-360 degrees. FIDs and Angles are linearly interpolated for missing frames. This
interpolation can be improved.
Please see https://github.com/ahoimarie/cb_recordings for more explanation.
"""


def loadWhiskerData(df, angle_trace, phase_trace, filtered_sig, sr=299):
    """Load _whiskermeasurements.npy data and assign them
       to fid, labels and angles. Process them further to
       create a class containing the whisker data with different
       parameters such as times of whisking on each whisker,
       and the Hilbert transformed signals. """

    ### Initialise variables and functions ###
    n_frame, n_whisk = angle_trace.shape  # number frames and whiskers
    setpt_func = lambda x: (max(x) + min(
        x)) / 2.0  # function describing the setpoint location
    amp_func = np.ptp  # function describing the magnitude of amplitude
    thr = 10 * np.pi / 180  # (10 degrees to rad) threshold on whisker angle amplitude above which that part of a recording is taken to be a whisking bout
    a = 0.01  # 0.005 parameter for amplitude smoothing filter such that whisking bout 'cutouts' are not too short

    lag = 750  # % number of lags used to compute whisker autocorrelation
    tau = [x / sr for x in range(-lag, lag, 1)
           ]  # % autocorrelation temporal lag vector

    # for each whisking bout recorded we will have a list that contains
    # the following variables
    isw_ifrm = []  # frame numbers
    iisw_ispk = []  # frame numbers during which a spike was recorded
    isw_itop = []  # sample numbers during the whisker angle has a peak
    isw_angl = []  # whisker angle
    isw_phs = []  # whisker phase
    isw_amp = []  # whisker amplitude
    isw_spt = []  # whisker set-point
    # isw_sam = []  # sample id

    # lists that will hold relevant information for each entire recording interval:
    # frame = []
    # time = []
    phase = []
    amplitude = []
    spikes = []
    setpoint = []
    angle = []
    tops = []
    iswhisking = [
    ]  # a list of indices pointing to times where the animal was whisking

    ACisw = np.zeros(
        (2 * lag + 1, 100))  # autocorrelation of whisker angle during whisking
    ACall = np.zeros(
        (2 * lag + 1,
         n_whisk))  # autocorrelation of whisker angle during all times

    # individuate whisking bounts and extract variables ###
    for w_id in range(n_whisk):
        angl = angle_trace[:, w_id]  # angle during recording
        frm = np.arange(n_frame)  # array of frames
        # angl = np.array(np.transpose(angle_traces[labels == j]))
        # sam = np.array(np.transpose(fids[labels == j])) - 1  # sample number
        # spk = np.array(np.transpose(sp.st[sp.clu==sp.clu[i]]))  # samples during which spike was recorded

        phs = phase_trace[:,
                          w_id]  # phase from Hilbert transform of whisker angl

        amp, itop, _ = get_slow_var(angl, phs, amp_func)  # get amplitudes
        spt, _, _ = get_slow_var(angl, phs, setpt_func)  # get setpoint
        # spt = setpoint_lowpass(angl, 2, 6, sr)  #         order = 2, Wn = 6


        # ACall[:,j] = np.correlate(amp*cos(phs)/2,lag,'full') # needs work

        # Construct and apply filter
        [bb, aa] = butter(1, a / 3)
        # ampfilt = filtfilt(a,[1,a-1],amp,padtype = 'odd',
        # padlen=3*(max(np.size(a),len([1,a-1]))-1)) # almost identical to Aljadeffs ver
        ampfilt = filtfilt(bb,
                           aa,
                           amp,
                           padtype='odd',
                           padlen=3 * (max(np.size(bb), len(aa)) - 1))

        iisw = np.heaviside(
            ampfilt - thr,
            .5)  # % indices of times where the animal was whisking
        iifrm = frm[iisw == True]
        isw_con = group_consecutives(
            iifrm, step=1)  # list of lists of consecutive whisking times

        # Lists of variables for each whisker bout
        tisw_ifrm = []  # frame numbers
        tiisw_ispk = []  # frame numbers during which a spike was recorded
        tisw_itop = []  # frame numbers during the whisker angle has a peak
        tisw_angl = []  # whisker angle
        tisw_phs = []  # whisker phase
        tisw_amp = []  # whisker amplitude
        tisw_spt = []  # whisker set-point
        # tisw_sam = []   # sample id

        for iisw_temp in isw_con:
            # looping over connected components of the list iiswcon
            # (each connected component is a whisking bout)
            # the variables in each whisking bout are put into the
            # appropriate list
            tisw_ifrm.append(iisw_temp)  # same as isw_con!
            tisw_itop.append(np.intersect1d(itop, iisw_temp))
            tisw_angl.append(angl[iisw_temp])
            tisw_phs.append(phs[iisw_temp])
            tisw_amp.append(amp[iisw_temp])
            tisw_spt.append(spt[iisw_temp])
            # tisw_sam.append(sam[iisw_temp])

        # For each whisker, varibles saved for each whisking bouts
        isw_ifrm.append(tisw_ifrm)
        isw_itop.append(tisw_itop)
        isw_angl.append(tisw_angl)
        isw_phs.append(tisw_phs)
        isw_amp.append(tisw_amp)
        isw_spt.append(tisw_spt)
        # isw_sam.append(tisw_sam)

        # For each whisker, varibles saved for entire recording
        # frame.append(frm)
        phase.append(phs)
        amplitude.append(amp)
        tops.append(itop)
        setpoint.append(spt)
        angle.append(angl)
        iswhisking.append(iisw)
        # mic.append(len(isw_con))
        # time.append(frm / float(sr)) # converte into sec

    # Array of frames and times of recording
    frame = frm
    time = frm / float(sr)  # converte into sec

    # Loading whisking epochs start as event_times (fancy way)
    # event_times = []

    # [
    #     event_times.append([item[0] / sr for item in isw_ifrm[j]])
    #     for j in range(n_whisk)
    # ]  # in sec

    event_times = [[item[0] / sr for item in isw_ifrm[j]]
                   for j in range(n_whisk)]  # in sec

    # Store variables in class whiskdata
    whiskdata = type('whiskdata', (),
                     {})  # see https://realpython.com/python-metaclasses/
    # class whiskdata:            # equivalent way to generate empty clase
    #     pass
    whiskdata.df = df  # dlc dataframe without top level
    whiskdata.nwhisk = n_whisk  # number of whiskers
    whiskdata.sr = sr  # sampling rate
    whiskdata.frame = frame  # array of recording frames
    whiskdata.nframe = n_frame  # number of frames
    whiskdata.time = time  # array of recording times in sec
    whiskdata.phase = phase  # phase entire trace
    whiskdata.amplitude = amplitude  # amplitude entire trace
    whiskdata.spikes = spikes  # empty here
    whiskdata.setpoint = setpoint  # setpoint entire trace
    whiskdata.angle = angle
    whiskdata.fangle = filtered_sig  # filtered angle_trace
    whiskdata.tops = tops
    whiskdata.iswhisking = iswhisking
    whiskdata.isw_ifrm = isw_ifrm  # id frames while whisking
    whiskdata.isw_itop = isw_itop  # id frames while whisking and top position (phase calculated)
    whiskdata.isw_angl = isw_angl  # angle while whisking
    whiskdata.isw_phs = isw_phs  # phase while whisking
    whiskdata.isw_amp = isw_amp  # amplitude while whisking
    whiskdata.isw_spt = isw_spt  # setpoint while whisking
    # whiskdata.isw_sam = isw_sam
    whiskdata.event_times = event_times  # times of whisking onset
    # whiskdata.ACall   = ACall
    # whiskdata.ACisw   = ACisw

    return whiskdata


def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:
        exit("Too few arguments calling script")

    exptn = sys.argv[1]


def setpoint_lowpass(data, order, Wn, sr):
    ''' Compute setpoint by lowpass filtering angle instead of
    Mary's get_slow_var method
    '''
    b, a = butter(order, Wn, btype='lowpass', fs=sr)
    spt = filtfilt(b, a, data)
    return spt
