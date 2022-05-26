#!/usr/bin/env python
"""
Python implementation of the following psychoacoustic features for perceptual audio quality assessment.
a) noise to mask ratio (NMR), 1992, Brandenburg and Sporer
b) cepstral correlation (Cep), 2014, Kates and Arehart
c) perceptual similarity (PSMt), 1997, Dau and Kollmeier
d) neurongram similarity index measure (NSIM), 2012 Hines and Harte
"""
import multiprocessing as mp
from abc import ABCMeta, abstractmethod

import librosa as lr
import numpy as np
import pkg_resources
import scipy.signal as sigs
from numba import njit
from pyACA import ToolGammatoneFb


class BaseFeatureExtractor(metaclass=ABCMeta):
    """
    base class for audio feature extractor
    """

    m_tar_sig = []
    m_ref_sig = []

    def __init__(self, tar_sig, ref_sig, fs):
        self.m_tar_sig = tar_sig
        self.m_ref_sig = ref_sig
        self.m_fs = fs  # assume 48kHz most of the time; need to implement a sanity check?

    def temporal_align_signals(self):
        # make sure audio signals are well-aligned
        # shift target signal to match the reference signal
        lag = BaseFeatureExtractor.estimate_xcorr_lag(self.m_tar_sig, self.m_ref_sig)
        self.m_tar_sig = BaseFeatureExtractor.shift_signal(self.m_tar_sig, lag)

    @staticmethod
    def shift_signal(array_tar, lag):
        """
        shift array_tar given the lag
        :param array_tar: float, ndarray (num_sample_1,), target signal
        :param lag: int, lag between two signals. if lag > 0: tar needs to be zero-padded; vice versa
        :return: array_tar_shifted
        """
        if lag > 0:
            array_tar_shifted = np.concatenate((np.zeros(lag, ), array_tar), axis=0)
        else:
            array_tar_shifted = array_tar[abs(lag):]
        return array_tar_shifted

    @staticmethod
    def estimate_xcorr_lag(array_tar, array_ref):
        """
        calculates the lag between two signals by finding the maximum peak of cross-correlation function
        :param array_tar: float, ndarray (num_sample_1,), target signal
        :param array_ref: float, ndarray (num_sample_2,), reference signal
        :return: max_lag: int, lag
        """
        # ==== zero pad the shorter signal before xcorr()
        if len(array_tar) > len(array_ref):
            tmp = BaseFeatureExtractor.zero_pad_audio_segment(array_ref,
                                                              abs(len(array_tar) - len(array_ref)), "back")
            r = sigs.correlate(tmp, array_tar, mode="full", method="fft")
        else:  # len(array_ref) > len(array_target):
            tmp = BaseFeatureExtractor.zero_pad_audio_segment(array_tar,
                                                              abs(len(array_tar) - len(array_ref)), "back")
            r = sigs.correlate(array_ref, tmp, mode="full", method="fft")
        lag = np.arange(-len(r) // 2 + 1, len(r) // 2 + 1)
        max_lag = lag[int(np.argmax(r))]
        return max_lag

    @staticmethod
    def zero_pad_audio_segment(segment, num_zeros, location):
        """
        simple function to concatenate zeros to a segment of audio
        :param segment: ndarray, (num_sample, ) audio segment
        :param num_zeros: int, number of zeros to concatenate
        :param location: string, "front" or "back"
        :return: segment_zp: ndarray, (num_sample + num_zeros, ) audio segment
        """
        zp = np.zeros(num_zeros)
        if location == "front":
            segment_zp = np.concatenate((zp, segment), axis=0)
        elif location == "back":
            segment_zp = np.concatenate((segment, zp), axis=0)
        else:
            raise ValueError("location -- %s is undefined" % location)
        return segment_zp

    @staticmethod
    def _truncate_signals(sig_ref, sig_tar):
        # make sure the audio duration of target and reference signals are matched
        # truncate the longer one to match the shorter one
        len_tar = len(sig_tar)
        len_ref = len(sig_ref)
        min_len = min(len_ref, len_tar)
        sig_ref = sig_ref[0:min_len]
        sig_tar = sig_tar[0:min_len]
        return sig_ref, sig_tar

    def preprocess_signals(self):
        self.temporal_align_signals()
        self.m_ref_sig, self.m_tar_sig = self._truncate_signals(self.m_ref_sig, self.m_tar_sig)

    @abstractmethod
    def extract_features(self):
        raise NotImplementedError


"""
===========================================================
===============  Derived classes CepCorr ==================
===========================================================
"""


class CepcorrFeatureExtractor(BaseFeatureExtractor):
    """
    CepCorr is a selected feature from HASQI/HAAQI for determining perceptual quality of speech/audio
    the basic concept is described in [1][2]
    this feature represents the averaged correlation coefficient of cepstral coefficients of higher modulation bands
    for more information, see
    [1] Kates, J. M., and Areharts, K. H., The hearing-aid speech quality index (HASQI) version 2,
        JAES, Vol.62, No.3, 2014
    [2] Kates, J. M., and Areharts, K. H., The hearing-aid audio quality index (HAAQI),
        IEEE/ACM TASLP, 24(2), 354-365, 2015
    This is a re-implementation of the CepCorr feature.
    For more specific details of the implementation, please contact Dr. Kates for MATLAB reference code

    The entry function is "self.extract_features()"
    """
    m_nchan = 32
    m_fs_24 = 24000
    m_level1 = 65  # default SPL
    m_ref_sig_24 = []
    m_tar_sig_24 = []
    m_feature = 0.0
    m_seg_win_size_sec = 0.008  # size of overlapping segment for env smoothing (8 msec)

    def compute_earmodel(self):
        """
        this processing block simulates the middle ear and the auditory filterbank (gammatone).
        the original implementation in HAAQI also includes simulation of inner hair cell (IHC) and
        outer hair cell (OHC) damages; these steps were removed (or simplified) due to our focus on normal hearing
        """
        assert(len(self.m_ref_sig) == len(self.m_tar_sig))
        # ==== resampling (already aligned broadband signals)
        self.m_ref_sig_24 = lr.core.resample(self.m_ref_sig, orig_sr=self.m_fs, target_sr=self.m_fs_24)
        self.m_tar_sig_24 = lr.core.resample(self.m_tar_sig, orig_sr=self.m_fs, target_sr=self.m_fs_24)
        # ==== NAL-R EQ
        # bypass this step because we do not focus on hearing loss (HL) simulation
        # when no HL is applied, NAL-R filter is just a delay oSSf 140 samples
        # ==== middle ear
        self._compute_middle_ear()
        # ==== get loss params
        # since no hearing loss is simulated, the essential params can be retrieved via look-up tables
        self._get_loss_params()
        # ==== gammatone filterbank
        self._compute_auditory_filterbank()

    def compute_env_smooth(self):
        """
        this processing block smooths the representations (envelope and basilar membrane motion)
        returned by the ear model
        """
        self.ref_db_smooth = self._compute_smoothing(self.ref_db_fb, self.m_seg_win_size_sec,
                                                     self.m_nchan, self.m_fs_24)
        self.tar_db_smooth = self._compute_smoothing(self.tar_db_fb, self.m_seg_win_size_sec,
                                                     self.m_nchan, self.m_fs_24)

    def compute_cep_corr_high(self):
        """
        this processing block takes smoothed version of filterbank representation and computes the cepCorr
        at higher modulation frequency
        """
        # ==== prepare mel cepstrum basis functions
        num_basis = 6
        num_bands = self.m_nchan
        cep_mat = self._prepare_cep_basis(num_basis, num_bands)
        # ==== silence removal
        x_ns, y_ns = self._remove_silence_add_noise(self.ref_db_smooth, self.tar_db_smooth, num_bands)
        # ==== compute mel cepstrum coef
        x_cep, y_cep = self._compute_mel_cep_coef(x_ns, y_ns, cep_mat)
        # ==== modulation filtering
        x_cep_mod, y_cep_mod = self._compute_modulation_filtering(self.m_seg_win_size_sec, x_cep, y_cep)
        # ==== compute cross-covariance matrix
        cov_mat = self._compute_cross_covariance_matrix(x_cep_mod, y_cep_mod)
        # ==== average over higher modulation bands
        cep_corr_high = np.sum(cov_mat[4:, 1:])
        cep_corr_high = cep_corr_high/(4 * (num_basis-1))
        self.m_feature = cep_corr_high

    @staticmethod
    def _compute_cross_covariance_matrix(x_cep_mod, y_cep_mod):
        """
        compute cross-covariance matrix using the modulated mel cepstral coefficients
        :param x_cep_mod: num_mod by num_basis modulated mel coefficients of reference signal
        :param y_cep_mod: num_mod by num_basis modulated mel coefficients of target signal
        :return: cov_mat --> num_mod by num_basis cross-covariance matrix
        """
        small = 1e-30
        num_mod, num_basis, num_blocks = np.shape(x_cep_mod)
        cov_mat = np.zeros((num_mod, num_basis))
        for m in range(0, num_mod):
            for k in range(0, num_basis):
                c_x = x_cep_mod[m][k]
                c_x = c_x - np.mean(c_x)
                sum_x = np.sum(np.power(c_x, 2))  # variance?
                c_y = y_cep_mod[m][k]
                c_y = c_y - np.mean(c_y)
                sum_y = np.sum(np.power(c_y, 2))
                if (sum_x < small) | (sum_y < small):
                    cov_mat[m, k] = 0
                else:
                    cov_mat[m, k] = np.abs(np.dot(c_x, c_y))/np.sqrt(sum_x * sum_y)
        return cov_mat

    @staticmethod
    def _compute_modulation_filtering(seg_win_size_sec, x_cep, y_cep):
        """
        apply modulation filterbank on mel coefficients per block
        :param seg_win_size_sec: segment window size in sec
        :param x_cep: num_basis by num_blocks -->  mel coefficients of reference signal (per block)
        :param y_cep: num_basis by num_blocks -->  mel coefficients of target signal (per block)
        :return: x_cep_mod, y_cep_mod --> num_mod by num_basis modulated mel coefficients
        """
        seg_win_size_ms = seg_win_size_sec * 1000
        fs_sub = 1000.0/(0.5 * seg_win_size_ms)  # envelope sub-sampling rate
        edge_cf = [4.0, 8.0, 12.5, 20.0, 32.0, 50.0, 80.0]  # modulation frequencies
        num_mod = 1 + len(edge_cf)
        num_basis, num_blocks = np.shape(x_cep)
        assert(np.size(x_cep, 0) == np.size(y_cep, 0))
        n_fir = round(128 * ((fs_sub * 0.5)/125))
        n_fir = int(2 * np.floor(n_fir/2))
        n_fir2 = int(n_fir/2)
        b = [sigs.firwin(n_fir + 1, edge_cf[0], window="hann", fs=fs_sub)]
        for m in range(1, num_mod-1):
            b.append(sigs.firwin(n_fir+1, [edge_cf[m-1], edge_cf[m]], pass_zero=False, window="hann", fs=fs_sub))
        b.append(sigs.firwin(n_fir+1, edge_cf[num_mod-2], pass_zero=False, window="hann", fs=fs_sub))
        x_cep_mod = []
        y_cep_mod = []
        for m in range(0, num_mod):
            x_tmp = []
            y_tmp = []
            for k in range(0, num_basis):
                c_x = sigs.convolve(b[m], x_cep[k, :])
                c_y = sigs.convolve(b[m], y_cep[k, :])
                x_tmp.append(c_x[n_fir2:n_fir2+num_blocks+1])
                y_tmp.append(c_y[n_fir2:n_fir2+num_blocks+1])
            x_cep_mod.append(x_tmp)
            y_cep_mod.append(y_tmp)
        return x_cep_mod, y_cep_mod

    @staticmethod
    def _compute_mel_cep_coef(x_ns, y_ns, cep_mat):
        """
        compute the mel cepstrum coefficients using non-silent parts of the signals
        :param x_ns: num_bands by num_blocks, non-silent reference signal matrix
        :param y_ns: num_bands by num_blocks, non-silent target signal matrix
        :param cep_mat: num_bands by num_basis, cepstrum basis functions
        :return: x_cep, y_cep: num_basis by num_blocks, mel cep coefficient per block (for reference and target signal)
        """
        num_bands, num_basis = np.shape(cep_mat)
        num_blocks = np.size(x_ns, 1)
        assert(np.size(x_ns, 1) == np.size(y_ns, 1))
        x_cep = np.zeros((num_basis, num_blocks))
        y_cep = np.zeros((num_basis, num_blocks))
        for n in range(0, num_blocks):
            for k in range(0, num_basis):
                x_cep[k, n] = np.dot(x_ns[:, n], cep_mat[:, k])
                y_cep[k, n] = np.dot(y_ns[:, n], cep_mat[:, k])
        for k in range(0, num_basis):
            x_cep[k, :] = np.subtract(x_cep[k, :], np.mean(x_cep[k, :]))
            y_cep[k, :] = np.subtract(y_cep[k, :], np.mean(y_cep[k, :]))
        return x_cep, y_cep

    @staticmethod
    def _remove_silence_add_noise(x_db, y_db, num_bands):
        """
        remove samples that are below a specific threshold
        optionally, a gaussian random noise could be added by specifying its gain
        :param x_db: num_bands by num_blocks, smoothed & filtered reference signals (in dB SPL)
        :param y_db: num_bands by num_blocks, smoothed & filtered target signals (in dB SPL)
        :param num_bands: int, number of auditory filter bands
        :return: x_ns, y_ns --> non-silent version of x_db and y_db
        """
        thr = 2.5  # silence threshold hard-coded in HAAQI
        x_lin = np.power(10, np.divide(x_db, 20.0))
        x_sum = 20 * np.log10(np.divide(np.sum(x_lin, axis=0), num_bands))
        index = [i for i in range(0, len(x_sum)) if x_sum[i] > thr]
        x_ns = np.zeros(np.shape(x_db))
        y_ns = np.zeros(np.shape(y_db))
        if len(index) <= 1:
            print("CepcorrFeatureExtractor._remove_silence_add_noise(): tracks cannot be completely silent")
            return x_ns, y_ns
        x_ns = x_db[:, index]
        y_ns = y_db[:, index]
        # ==== Note: in HAAQI, the param is hardcoded to 0.0, which adds nothing to the signal
        # add_noise = 0.0
        # x_ns = np.add(x_ns, add_noise * np.random.randn(np.size(x_ns, 0), np.size(x_ns, 1)))
        # y_ns = np.add(y_ns, add_noise * np.random.randn(np.size(y_ns, 0), np.size(y_ns, 1)))
        return x_ns, y_ns

    @staticmethod
    def _prepare_cep_basis(num_basis, num_bands):
        """
        this function prepares the basis functions for computing cepstral coefficients
        :param num_basis: int, number of basis functions
        :param num_bands: int, number of frequency bands (of the gammatone filterbank)
        :return: cep_mat --> a matrix of cepstral basis functions
        """
        freq = np.arange(0, num_basis)
        k = np.arange(0, num_bands)
        cep_mat = np.zeros((num_bands, num_basis))
        for i in range(0, num_basis):
            basis = np.cos(np.divide(np.multiply(freq[i] * np.pi, k), num_bands - 1))
            cep_mat[:, i] = np.divide(basis, np.linalg.norm(basis))
        return cep_mat

    @staticmethod
    def _compute_smoothing(sig_mat, seg_win_size_sec, nchan, fs):
        """
        performing cosine window smoothing using predefined block size and overlap and add technique
        :param sig_mat: nchan by num_samples, signal matrix returned by the auditory filterbank
        :param seg_win_size_sec: float, default number in HAAQI is 0.008 sec (8 msec)
        :param nchan: int, default number in HAAQI is 32
        :param fs: int, default number in HAAQI is 24000
        :return: sig_mat_smooth --> smoothed signal matrix
        """
        # ==== init params
        seg_win_size = round(seg_win_size_sec * fs)
        seg_win_size += np.mod(seg_win_size, 2)  # make sure the win size is always even number
        seg_hop_size = int(seg_win_size/2)
        # ==== init cosine window params
        cosin_window = np.hanning(seg_win_size)
        win_sum = np.sum(cosin_window)
        half_cosin_window = cosin_window[seg_hop_size:]
        half_win_sum = np.sum(half_cosin_window)
        num_samples = np.size(sig_mat, 1)
        num_seg = int(1 + np.floor(num_samples/seg_win_size) + np.floor(((num_samples - seg_hop_size)/seg_win_size)))
        # ==== smoothing
        sig_mat_smooth = np.zeros((nchan, num_seg))
        for i in range(0, nchan):
            r = sig_mat[i, :]
            # - first hop, apply second half of the smoothing window
            istart = 0
            sig_mat_smooth[i, 0] = np.sum(np.dot(r[istart:seg_hop_size], half_cosin_window))/half_win_sum
            # - middle hop, apply full smoothing window
            for j in range(1, num_seg-1):
                istart += seg_hop_size
                iend = istart + seg_win_size
                sig_mat_smooth[i, j] = np.sum(np.dot(r[istart:iend], cosin_window))/win_sum
            # - last hop, apply first half of the smoothing window
            istart += seg_hop_size
            iend = istart + seg_hop_size
            sig_mat_smooth[i, num_seg-1] = np.sum(np.dot(r[istart:iend], cosin_window[0:seg_hop_size])) / half_win_sum
        return sig_mat_smooth

    def _compute_auditory_filterbank(self):
        """
        gammatone filtering happens in this step
        :return:
        """
        # init matrices for storing filter bank (fb) output
        ref_db_fb = np.zeros((self.m_nchan, len(self.m_ref_sig_24)))
        tar_db_fb = np.zeros((self.m_nchan, len(self.m_tar_sig_24)))
        # ref_bm_fb = np.zeros((self.m_nchan, len(self.m_ref_sig_24)))
        # tar_bm_fb = np.zeros((self.m_nchan, len(self.m_tar_sig_24)))
        bw_x = np.zeros((self.m_nchan, ))
        bw_y = np.zeros((self.m_nchan, ))

        # ==== loop through each band of filter
        for i in range(0, self.m_nchan):
            # - gammatone_env2
            [x_control, y_control] = self._compute_gammatone_env2_bm2(self.m_ref_sig_24, self.bw_1[i],
                                                                      self.m_tar_sig_24, self.bw_1[i],
                                                                      self.m_fs_24, self.center_freq[i],
                                                                      return_bm2_flag=False)
            # - bw_adjust
            bw_x[i] = self._adjust_bw(x_control, self.bw_min[i], self.bw_1[i], self.m_level1)
            bw_y[i] = self._adjust_bw(y_control, self.bw_min[i], self.bw_1[i], self.m_level1)
            # - gammatone_bm2
            [env_x, bm_x, env_y, bm_y] = self._compute_gammatone_env2_bm2(self.m_ref_sig_24, bw_x[i],
                                                                          self.m_tar_sig_24, bw_y[i],
                                                                          self.m_fs_24, self.center_freq[i],
                                                                          return_bm2_flag=True)
            # - RMS levels
            # for CepCorr, the xave, xcave, yave, ycave are not required
            # skipped these variables for simplicity
            # - env_compress_bm
            env_x_comp, bm_x_comp = self._compute_env_compress_bm(env_x, bm_x, x_control, self.attn_ohc[i],
                                                                  self.low_knee[i], self.comp_ratio[i],
                                                                  self.m_fs_24, self.m_level1)
            env_y_comp, bm_y_comp = self._compute_env_compress_bm(env_y, bm_y, y_control, self.attn_ohc[i],
                                                                  self.low_knee[i], self.comp_ratio[i],
                                                                  self.m_fs_24, self.m_level1)
            # - env_align
            # in the original HAAQI code, the target signal was delayed by 2 msec. Additionally, the NAL-R filter
            # has a 140 sample delay. Here, we are only concerned about normal hearing, no additional delay was applied
            # therefore, drop the realignment step
            # - env_sl2
            env_x_comp_sl, bm_x_comp_sl = self._compute_env_sl2(env_x_comp, bm_x_comp, self.attn_ihc[i], self.m_level1)
            env_y_comp_sl, bm_y_comp_sl = self._compute_env_sl2(env_y_comp, bm_y_comp, self.attn_ihc[i], self.m_level1)
            # - ihc_adapt
            # delta = 2.0  # same hard-coded param as in HAAQI: amount of overshoot
            # ref_db_fb[i, :], bm_x_comp_sl = self._adapt_ihc(env_x_comp_sl, bm_x_comp_sl, delta, self.m_fs_24)
            # tar_db_fb[i, :], bm_y_comp_sl = self._adapt_ihc(env_y_comp_sl, bm_y_comp_sl, delta, self.m_fs_24)
            ref_db_fb[i, :] = env_x_comp_sl
            tar_db_fb[i, :] = env_y_comp_sl
            # - bm_add_noise
            # ihc_thr = -10.0  # additive noise level, dB re: auditory threshold
            # ref_bm_fb[i, :] = self._add_noise_bm(bm_x_comp_sl, ihc_thr, self.m_level1)
            # tar_bm_fb[i, :] = self._add_noise_bm(bm_y_comp_sl, ihc_thr, self.m_level1)

        # ==== group_delay_comp
        # Note: for the fb output of a processing step, the variable names are changed back to tar/ref
        self.ref_db_fb = self._comp_group_delay(ref_db_fb, bw_x, self.center_freq, self.m_fs_24)
        self.tar_db_fb = self._comp_group_delay(tar_db_fb, bw_x, self.center_freq, self.m_fs_24)
        # Note: the following two filtered signal are not used to generate CepCorr
        # self.ref_bm_fb = self._comp_group_delay(ref_bm_fb, bw_x, self.center_freq, self.m_fs_24)
        # self.tar_bm_fb = self._comp_group_delay(tar_bm_fb, bw_x, self.center_freq, self.m_fs_24)

    @staticmethod
    def _comp_group_delay(sig_mat, bw, cf, fs):
        """
        this function compensates the group delay of each filter band at its center frequency
        :param sig_mat: nchan by num_sample signal matrix
        :param bw: gammatone filter bandwidth
        :param cf: center frequencies of the bands
        :param fs: sampling rate in Hz
        :return: sig_mat_comp --> signal matrix with group delay compensation
        """
        nchan = len(bw)
        ear_q = 9.26449
        min_bw = 24.7
        erb = min_bw + np.divide(cf, ear_q)
        tpt = 2 * np.pi / fs
        [a1, a2, a3, a4, a5, _] = CepcorrFeatureExtractor._get_gammatone_filter_coefficients(tpt, bw, erb)
        group_delay = np.zeros((nchan,))
        for i in range(0, nchan):
            dump, group_delay[i] = sigs.group_delay(([1, a1[i], a5[i]], [1, -a1[i], -a2[i], -a3[i], -a4[i]]), 1)
        group_delay = np.round(group_delay)
        group_delay = np.subtract(group_delay, min(group_delay))
        correct = np.subtract(max(group_delay), group_delay)
        sig_mat_comp = np.zeros(np.shape(sig_mat))
        for i in range(0, nchan):
            r = sig_mat[i, :]
            num_samples = len(r)
            r = CepcorrFeatureExtractor.zero_pad_audio_segment(r, int(correct[i]), "front")[0:num_samples]
            sig_mat_comp[i, :] = r
        return sig_mat_comp

    @staticmethod
    def _add_noise_bm(bm_x, thr, level1):
        """
        add random gaussian noise to a signal
        :param bm_x: signal (envelope or basilar membrane motion), np.shape(bm_x) = (num_sample, )
        :param thr: auditory threshold
        :param level1: reference level dB SPL at rms = 1
        :return: bm_x_noise --> signal with additive noise
        """
        gain = np.power(10, (thr - level1)/20.0)
        noise = gain * np.random.randn(len(bm_x),)
        bm_x_noise = np.add(bm_x, noise)
        return bm_x_noise

    @staticmethod
    def _adapt_ihc(env_x, bm_x, delta, fs):
        """
        Note: this is the SLOWEST function according profiler
        this function is a re-implementation of eb_IHCadapt()
        it adjusts the gain of both envelope and basilar membrane motion signals based on neural firing rate
        please see [1], section 2, last paragraph for more explanation
        :param env_x: signal envelope of one freq band (dB)
        :param bm_x: basilar membrane motion
        :param delta: overshoot factor (with respect to steady-state)
        :param fs: sampling rate in Hz
        :return: env_x_adapted, bm_x_adapted
        """
        delta = np.max((delta, 1.0001))

        # init time constants (in seconds)
        tau1 = 0.001 * 2.0
        tau2 = 0.001 * 60.0

        # equivalent circuit params
        T = 1.0/fs
        R1 = 1/delta
        R2 = 0.5 * (1 - R1)
        R3 = 0.5 * (1 - R1)
        C1 = tau1 * (R1 + R2) / (R1 * R2)
        C2 = tau2 / ((R1 + R2) * R3)

        a11 = R1 + R2 + R1 * R2 * (C1 / T)
        a12 = -R1
        a21 = -R3
        a22 = R2 + R3 + R2 * R3 * (C2 / T)
        denom = 1.0/(a11 * a22 - a21 * a12)

        R1_inv = 1.0/R1
        R12_C1 = R1 * R2 * (C1 / T)
        R23_C2 = R2 * R3 * (C2 / T)

        num_samples = len(env_x)
        gain = np.ones((num_samples,))
        env_x_adapted = np.zeros((num_samples,))
        V1 = 0.0
        V2 = 0.0
        small = 1e-30

        for i in range(0, num_samples):
            V0 = env_x[i]
            b1 = V0 * R2 + R12_C1 * V1
            b2 = R23_C2 * V2
            V1 = denom * (a22 * b1 - a12 * b2)
            V2 = denom * (-a21 * b1 + a11 * b2)
            out = (V0 - V1) * R1_inv
            out = np.maximum(out, 0.0)  # no drop below threshold
            env_x_adapted[i] = out
            gain[i] = np.divide((out + small), (V0 + small))

        bm_x_adapted = np.multiply(gain, bm_x)
        return env_x_adapted, bm_x_adapted

    @staticmethod
    def _compute_env_sl2(env_sig, bm_sig, attn_ihc, level1):
        small = 1e-30
        env_sig_sl = 20 * np.log10(env_sig + small) - attn_ihc + level1
        env_sig_sl = np.maximum(env_sig_sl, 0.0)
        gain = np.divide(env_sig_sl + small, env_sig + small)
        bm_sig_sl = np.multiply(gain, bm_sig)
        return env_sig_sl, bm_sig_sl

    @staticmethod
    def _compute_env_compress_bm(env_sig, bm_sig, control_sig, attn_ohc, thr_low, cr, fs, level1):
        """
        this function computes the gain for compressing envelope and basilar membrane motion
        :param env_sig: envelope signal from the gammatone filter bank
        :param bm_sig: basilar membrane motion output from gaamatone filter bank
        :param control_sig: control envelope from control path filter bank
        :param attn_ohc: OHC attenuation (0 in the case of normal hearing)
        :param thr_low: knee point for linear amplification
        :param cr: compression ratio
        :param fs: sampling rate in Hz
        :param level1: reference level in dB
        :return: env_sig_comp: compressed version of the signal env
                 bm_sig_comp: compressed version of the basilar membrane motion
        """
        # internal function that implements eb_EnvCompressBM() in HAAQI
        thr_high = 100.0
        small = 1e-30
        log_env = np.maximum(control_sig, small)
        log_env = level1 + 20 * np.log10(log_env)
        log_env = np.minimum(log_env, thr_high)
        log_env = np.maximum(log_env, thr_low)

        gain = np.subtract(-attn_ohc, np.multiply(np.subtract(log_env, thr_low), (1 - (1/cr))))
        gain = np.power(10, (np.divide(gain, 20)))
        flp = 800
        [b, a] = sigs.butter(1, flp/(0.5 * fs))
        gain = sigs.lfilter(b, a, gain)
        env_sig_comp = np.multiply(gain, env_sig)
        bm_sig_comp = np.multiply(gain, bm_sig)
        return env_sig_comp, bm_sig_comp

    @staticmethod
    def _adjust_bw(sig_control, bw_min, bw_max, level1):
        # adjust the bandwidth for the averaged signal level...
        c_rms = np.sqrt(np.mean(np.power(sig_control, 2)))
        c_db = 20 * np.log10(c_rms) + level1
        if c_db < 50:
            bw = bw_min
        elif c_db > 100:
            bw = bw_max
        else:
            bw = bw_min + ((c_db - 50)/50) * (bw_max - bw_min)  # linear interpolation
        return bw

    @staticmethod
    def _compute_gammatone_env2_bm2(sig_x, bw_x, sig_y, bw_y, fs, cf, return_bm2_flag):
        """
        @ when return_bm2_flag == False
        this function is a re-implementation of eb_GammatoneEnv2() from HAAQI
        the variable names generally follow the original naming convention with minor changes
        given the center frequency and bandwidth, this function filters the signals
        with a gammatone filter and computes the corresponding envelopes
        @ when return_bm2_flag == true
        this function becomes eb_GammatoneBM2() from HAAQI
        it takes the adjusted bw information and apply the gammatone filter on the middle-ear filtered signals
        in addition to computing the envelope, this function computes the basilar membrane motion of both signals
        :param sig_x: first sequence to be filtered (reference signal)
        :param bw_x: bandwidth of x
        :param sig_y: second sequence to be filtered (target signal)
        :param bw_y: bandwidth of y
        :param fs: sampling rate in Hz
        :param cf: filter center frequency in Hz
        :param return_bm2_flag: True or False, this determines the output of this function
        :return: env_x, env_y: envelopes of x and y signal
                 bm_x, bm_y: basilar membrane motions of x and y signal
        """
        ear_q = 9.26449
        min_bw = 24.7
        erb = min_bw + (cf/ear_q)
        tpt = 2 * np.pi / fs
        assert(len(sig_x) == len(sig_y))
        num_samples = len(sig_x)
        cos_cf, sin_cf = CepcorrFeatureExtractor._init_complex_demodulation(num_samples, tpt, cf)

        # ==== filter reference signal
        [a1_x, a2_x, a3_x, a4_x, a5_x, gain_x] = CepcorrFeatureExtractor._get_gammatone_filter_coefficients(tpt,
                                                                                                            bw_x, erb)
        u_real_x = sigs.lfilter([1, a1_x, a5_x], [1, -a1_x, -a2_x, -a3_x, -a4_x], np.multiply(sig_x, cos_cf))
        u_imag_x = sigs.lfilter([1, a1_x, a5_x], [1, -a1_x, -a2_x, -a3_x, -a4_x], np.multiply(sig_x, sin_cf))
        env_x = gain_x * np.sqrt(np.add(np.multiply(u_real_x, u_real_x), np.multiply(u_imag_x, u_imag_x)))

        # ==== filter target signal
        [a1_y, a2_y, a3_y, a4_y, a5_y, gain_y] = CepcorrFeatureExtractor._get_gammatone_filter_coefficients(tpt,
                                                                                                            bw_y, erb)
        u_real_y = sigs.lfilter([1, a1_y, a5_y], [1, -a1_y, -a2_y, -a3_y, -a4_y], np.multiply(sig_y, cos_cf))
        u_imag_y = sigs.lfilter([1, a1_y, a5_y], [1, -a1_y, -a2_y, -a3_y, -a4_y], np.multiply(sig_y, sin_cf))
        env_y = gain_y * np.sqrt(np.add(np.multiply(u_real_y, u_real_y), np.multiply(u_imag_y, u_imag_y)))

        if return_bm2_flag:
            bm_x = gain_x * np.add(np.multiply(u_real_x, cos_cf), np.multiply(u_imag_x, sin_cf))
            bm_y = gain_y * np.add(np.multiply(u_real_y, cos_cf), np.multiply(u_imag_y, sin_cf))
            return env_x, bm_x, env_y, bm_y
        else:
            return env_x, env_y

    @staticmethod
    @njit
    def _init_complex_demodulation(num_samples, tpt, cf):
        """
        Note: this function is also pretty slow according profiler
        :param num_samples: int, number of samples to generate for the demodulation function
        :param tpt: 2*pi/fs --> a constant depending on sampling rate
        :param cf: center frequency of the filter bank in Hz
        :return: cos_cf, sin_cf --> demodulated cos and sin components
        """
        cn = np.cos(tpt * cf)
        sn = np.sin(tpt * cf)
        cos_cf = np.zeros((num_samples, ))
        sin_cf = np.zeros((num_samples, ))
        c_old = 1
        s_old = 0
        cos_cf[0] = c_old
        sin_cf[0] = s_old
        for i in range(1, num_samples):
            arg = c_old * cn + s_old * sn
            s_old = s_old * cn - c_old * sn
            c_old = arg
            cos_cf[i] = c_old
            sin_cf[i] = s_old
        return cos_cf, sin_cf

    @staticmethod
    def _get_gammatone_filter_coefficients(tpt, bw, erb):
        tpt_bw = np.multiply(bw, tpt * erb * 1.019)
        a = np.exp(-tpt_bw)
        a1 = 4.0 * a
        a2 = -6.0 * np.power(a, 2)
        a3 = 4.0 * np.power(a, 3)
        a4 = -np.power(a, 4)
        a5 = 4.0 * np.power(a, 2)
        gain = 2.0 * (1 - a1 - a2 - a3 - a4) / (1 + a1 + a5)
        return a1, a2, a3, a4, a5, gain

    @staticmethod
    def resample_to_24khz(sig, original_fs):
        target_fs = 24000
        target_fs_khz = 24
        original_fs_khz = round(original_fs/1000)

        if original_fs == target_fs:  # do nothing
            sig_24 = sig
        elif original_fs < target_fs:  # upsampling
            sig_24 = lr.core.resample(sig, orig_sr=original_fs, target_sr=target_fs)
            sig_24 = CepcorrFeatureExtractor._match_rms_y_to_x(sig, sig_24)
        else:  # downsampling
            sig_24 = lr.core.resample(sig, orig_sr=original_fs, target_sr=target_fs)
            [b_in, a_in] = sigs.cheby2(7, 30, 21/original_fs_khz)
            sig_filt = sigs.lfilter(b_in, a_in, sig)
            [b_out, a_out] = sigs.cheby2(7, 30, 21/target_fs_khz)
            sig_24_filt = sigs.lfilter(b_out, a_out, sig_24)
            sig_24 = CepcorrFeatureExtractor._match_rms_y_to_x(sig_filt, sig_24_filt)
        return sig_24

    @staticmethod
    def _match_rms_y_to_x(sig_x, sig_y):
        # y = (xRMS/yRMS) * y
        x_rms = np.sqrt(np.mean(np.power(sig_x, 2)))
        y_rms = np.sqrt(np.mean(np.power(sig_y, 2)))
        sig_y_matched = np.multiply(x_rms / y_rms, sig_y)
        return sig_y_matched

    @staticmethod
    def apply_middle_ear_filters(sig, fs):
        #  In HAAQI, middle ear model is implemented by combining a LP filter with a HP filter (butterworth)
        [b_lp, a_lp] = sigs.butter(1, 5000 / (0.5 * fs))
        sig_lp = sigs.lfilter(b_lp, a_lp, sig)
        [b_hp, a_hp] = sigs.butter(2, 350 / (0.5 * fs), "highpass")
        sig_lp_hp = sigs.lfilter(b_hp, a_hp, sig_lp)
        return sig_lp_hp

    def _compute_middle_ear(self):
        self.m_ref_sig_24 = self.apply_middle_ear_filters(self.m_ref_sig_24, self.m_fs_24)
        self.m_tar_sig_24 = self.apply_middle_ear_filters(self.m_tar_sig_24, self.m_fs_24)

    def _get_loss_params(self):
        # in the case of normal hearing, HAAQI can be simplified by hard-coding the following params
        self.bw_1 = [1.28096378060800, 1.34463367600863, 1.40430675025603, 1.46192770111434, 1.51915683759555,
                     1.57734190779408, 1.63752299850873, 1.70045880652767, 1.76666378924627, 1.83644871838567,
                     1.90996001038767, 1.98721534334793, 2.06813449183959, 2.15256517637044, 2.24030420428322,
                     2.33111441100562, 2.42473799259552, 2.52090681759646, 2.61935025946742, 2.71980102514491,
                     2.82199938493744, 2.92569614162676, 3.03065461604316, 3.13665187384759, 3.24347937385736,
                     3.35094318140336, 3.45886386002423, 3.56707613033287, 3.67542836520979, 3.78378197475444,
                     3.89201072193444, 4]  # maximum bw for the control
        self.center_freq = [80.0000000000001, 114.496607321504, 152.846482559522, 195.480035391681, 242.875752217653,
                            295.565566324662, 354.140827899982, 419.258940893300, 491.650741216002, 572.128699084646,
                            661.596037565470, 761.056869659483, 871.627467699052, 994.548791535073, 1131.20041612150,
                            1283.11601480981, 1452.00057212610, 1639.74951921372, 1848.47000670326, 2080.50455376008,
                            2338.45733872837, 2625.22342643773, 2944.02126019725, 3298.42878314247, 3692.42359433343,
                            4130.42759028527, 4617.35659295376, 5158.67552116236, 5760.45972467264, 6429.46316926481,
                            7173.19423808563, 8000]
        # ttnOHCy,BWminy,lowkneey,CRy,attnIHCy <-- output from eb_LossParameters()
        self.attn_ohc = np.zeros((self.m_nchan,))
        self.attn_ihc = np.zeros((self.m_nchan,))
        self.bw_min = np.ones((self.m_nchan,))
        self.low_knee = np.multiply(30.0, np.ones((self.m_nchan,)))
        self.comp_ratio = [1.25000000000000, 1.32258064516129, 1.39516129032258, 1.46774193548387, 1.54032258064516,
                           1.61290322580645, 1.68548387096774, 1.75806451612903, 1.83064516129032, 1.90322580645161,
                           1.97580645161290, 2.04838709677419, 2.12096774193548, 2.19354838709677, 2.26612903225806,
                           2.33870967741936, 2.41129032258065, 2.48387096774194, 2.55645161290323, 2.62903225806452,
                           2.70161290322581, 2.77419354838710, 2.84677419354839, 2.91935483870968, 2.99193548387097,
                           3.06451612903226, 3.13709677419355, 3.20967741935484, 3.28225806451613, 3.35483870967742,
                           3.42741935483871, 3.50000000000000]

    def extract_features(self):
        self.preprocess_signals()  # in HAAQI, the target signal was delayed for 2 msec for dispersion
        # print("computing ear model")
        self.compute_earmodel()
        # print("smoothing envelopes")
        self.compute_env_smooth()
        # print("computing cepcorr feature")
        self.compute_cep_corr_high()
        # print("finished!")
        # ==== mel_corr
        return self.m_feature


"""
===========================================================
===============  Derived classes NMR ======================
===========================================================
"""


class NmrFeatureExtractor(BaseFeatureExtractor):
    """
    NMR is a selected feature from PEAQ for determining how significant the presence of noise is by calculating
    the noise vs the masking threshold. For more information, please see the following publications:
    [1] Brandenburg, K. and Sporer, T., "NMR" and "Masking flag": evaluation of quality using perceptual criteria,
        in Proc. AES international conference, 1992
    [2] ITU-R Rec. BS. 1387, "Method for objective measurements of perceived audio quality," international
        telecommunications union, Geneva, Switzerland, 1998
    [3] Thiede, T. et al., PEAQ - the ITU standard for objective measurement of perceived audio quality,
        JAES, vol.48, no.1/2, 2000
    [4] Kabal, P., An examination and interpretation of ITU-R BS.1387: perceptual evaluation of audio quality,
        tech rep., TSP lab, McGill University, 2002
        http://www-mmsp.ece.mcgill.ca/Documents/Software/, last access: 2019.12
    This is a re-implementation of NMR feature only.

    The entry function is "self.extract_features()"
    """
    m_feature = 0.0
    m_winsize = 2048
    m_hopsize = 512
    m_spec_diff = []
    m_spec_ref = []
    m_cb_diff = []
    m_cb_ref = []
    m_fc = []
    m_mask_patterns = []

    def compute_diff_stft(self):
        """
        this processing block prepares the spectrogram of diff (noise) signal and the reference signal
        Note: in [1], the diff signal is computed in the time domain, whereas in [2] it's done in freq domain
              freq domain ignores tiny phase shift and could be
        """
        spec_ref = lr.core.stft(self.m_ref_sig, n_fft=self.m_winsize, hop_length=self.m_hopsize, window='hann')
        spec_tar = lr.core.stft(self.m_tar_sig, n_fft=self.m_winsize, hop_length=self.m_hopsize, window='hann')
        self.m_spec_diff = np.abs(spec_ref) - np.abs(spec_tar)  # magnitude (linear)
        self.m_spec_ref = np.abs(spec_ref)  # magnitude (linear)

    def compute_critical_banding(self):
        """
        This processing block computes the averaged power within each critical band. For definition of critical band,
        please refer to [1] Table 1.
        """
        u_n, s_n, fc = self._get_critical_band_tables(self.m_fs, self.m_winsize)
        assert(len(u_n) == len(s_n))
        num_bands = len(u_n)
        num_blocks = np.size(self.m_spec_diff, axis=1)
        cb_diff = np.zeros((num_bands, num_blocks))
        cb_ref = np.zeros((num_bands, num_blocks))
        for t in range(num_blocks):
            for n in range(num_bands):
                istart = u_n[n]
                iend = istart + s_n[n]
                cb_diff[n, t] = np.mean(np.power(self.m_spec_diff[istart:iend, t], 2))
                cb_ref[n, t] = np.mean(np.power(self.m_spec_ref[istart:iend, t], 2))
        self.m_cb_diff = cb_diff  # power (linear); forcing minimum values to be 1e-12 causes bugs
        self.m_cb_ref = np.maximum(cb_ref, 1e-12)  # power (linear)
        self.m_fc = fc

    def compute_masking_patterns(self):
        """
        This processing block generates the masking patterns by convolving spreading functions with the spectrum.
        Note: the spreading functions are precomputed by assuming a fixed SPL (65 dB); they are computed in dB scale
              and later converted back to linear scale. The resulting masking patterns are in the power domain (linear)
        """
        num_bands, num_blocks = np.shape(self.m_cb_diff)
        mask_patterns = np.zeros((num_bands, num_blocks))
        spl = 65  # same default value as in HAAQI
        for i in range(0, num_blocks):
            spread_func_mat_lin = self._get_spreading_function(self.m_fc, spl)
            mask_patterns[:, i] = np.dot(spread_func_mat_lin, self.m_cb_ref[:, i])
        self.m_mask_patterns = mask_patterns  # power (linear)

    def compute_nmr(self):
        """
        This processing block computes the noise-to-mask ratio NMR feature based on the definition in [1].
        """
        small = 1e-30  # to avoid -Inf
        raw_nmr_per_band = np.mean(np.divide(self.m_cb_diff, self.m_mask_patterns), axis=0)
        nmr_total = 10 * np.log10(np.mean(raw_nmr_per_band) + small)
        self.m_feature = nmr_total

    @staticmethod
    def _get_spreading_function(fc, spl):
        """
        given a list of center frequency and sound pressure level in dB, return a matrix of
        spreading functions with all possible shifts
        Note: reference of the spreading function can be found in [2]
        :param fc: num_center_freq by 1, float vector of all center frequencies of critical bands
        :param spl: float, sound pressure level of current block
        :return: spread_func_mat_lin --> num_center_freq by num_shifts, spreading function matrix
        """
        # pre-compute spreading function in dB
        spread_func_mat = np.zeros((len(fc), len(fc)))
        for i in range(0, len(fc)):
            for p in range(0, len(fc)):
                if i <= p:
                    spread_func_mat[i, p] = np.maximum(27 * (i - p), -100)
                else:
                    spread_func_mat[i, p] = np.maximum((-24 - 230/fc[i] + 0.2*spl) * (i - p), -100)
        # convert to linear
        spread_func_mat_lin = np.power(10, np.divide(spread_func_mat, 10))
        return spread_func_mat_lin

    @staticmethod
    def _get_critical_band_tables(fs, windowsize):
        """
        this function returns the information for critical banding as described in Table 1 of [1]
        only accounts for fs = 48kHz
        Note: this table assumes 1024 freq. bins, which correspond to 2048 FFT size & 512 hop size
        :param fs: sampling rate in Hz; assume to be 48kHz
        :return: lower_edge_bin, size_bin --> frequency bin of the lower edge and size of critical bands
                 center_frequencies --> center frequencies of the critical bands in Hz
        """
        # ==== hard-coded at 48kHz
        assert(fs == 48000)
        lower_edge_bin = [0, 4, 8, 12, 16, 20, 24, 28, 32, 38, 44, 50, 58, 66, 76,
                          88, 102, 118, 138, 162, 192, 230, 278, 342, 426, 542, 684]
        size_bin = [4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 8, 10, 12,
                    14, 16, 20, 24, 30, 38, 48, 64, 84, 116, 142, 340]
        df = fs/windowsize
        center_frequencies = np.multiply(np.add(lower_edge_bin, np.divide(size_bin, 2)), df)
        return lower_edge_bin, size_bin, center_frequencies

    def extract_features(self):
        self.preprocess_signals()
        # ==== prepare STFT
        self.compute_diff_stft()
        # ==== Critical banding
        self.compute_critical_banding()
        # ==== spreading function
        self.compute_masking_patterns()
        # ==== pre/postmasking and NMR
        self.compute_nmr()
        return self.m_feature


"""
===========================================================
===============  Derived classes NSIM =====================
===========================================================
"""


class NsimFeatureExtractor(BaseFeatureExtractor):
    """
    NSIM is a selected feature from VISQOLAudio for determining perceptual quality of speech/audio
    the basic concept is described in [1][2]
    this feature represents the averaged correlation coefficient of cepstral coefficients of higher modulation bands
    for more information, see
    [1] Hines, A., Gillen, E., Kelly, D., Skoglund, J., Kokaram, A. and Harte, N., ViSQOLAudio: an objective audio
        quality metric for low bitrate codecs, JASA, Vol.137, No.6, 2015
    [2] Hines, A., Skoglund, J., Kokaram, A. and Harte, N., ViSQOL: an objective speech quality model,
        EURASIP, 2015:13, 2015
    [3] http://www.mee.tcd.ie/~sigmedia/Resources/ViSQOLAudio, last access: 2019.12
    [4] Hines, A., Harte, N., Speech intelligibility prediction using a neurogram similarity index measure,
        JSPECOM, vol. 54, no.2, 306-320, 2012
    This is a re-implementation of the NSIM feature. The original matlab implementation can be found in [3]

    The entry function is "self.extract_features()"
    """
    m_feature = 0.0
    m_patch_size = 30  # 30 frames per patch, hard-coded as in [1]
    m_warp = 1  # no warping for VISQOLAudio
    m_num_bands = 0
    m_spectrogram_tar = []
    m_spectrogram_ref = []
    m_dynamic_range = 0  # dynamic range in dB
    m_patches_ref = []
    m_patch_indices_ref = []
    m_patch_indices_tar = []

    def compute_both_sig_spectrograms(self):
        """
        this processing block computes power spectrogram of both target and reference signal
        neurograms were created by selecting specific frequency bins (as defined below)
        the reference spectrogram is normalized to have min power = 0, and the target spectrogram is adjusted
        accordingly.
        """
        # ==== initialize parameters
        bfs = [50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900,
               3400, 4000, 4800, 5800, 7000, 8500, 10500, 13500, 16000]
        self.m_num_bands = len(bfs)
        window_size = int(np.round((self.m_fs / 8000) * 256))  # 256 for 8k, 1536 for 48kHz sampling rate
        overlap = int(0.5 * window_size)
        # ==== scale target signal to match reference signal
        ref_spl = 20 * np.log10(np.sqrt(np.mean(np.power(self.m_ref_sig, 2)))/20e-6)
        self.m_tar_sig = self._scale_signal(self.m_tar_sig, ref_spl)
        # ==== compute spectrogram
        self.m_spectrogram_ref = self._get_normalized_power_spectrogram(self.m_ref_sig, window_size,
                                                                        overlap, self.m_fs, bfs)
        self.m_spectrogram_tar = self._get_normalized_power_spectrogram(self.m_tar_sig, window_size,
                                                                        overlap, self.m_fs, bfs)
        # ==== handle noise floor and dynamic range
        ref_floor = np.min(self.m_spectrogram_ref)
        tar_floor = np.min(self.m_spectrogram_tar)
        low_floor = np.minimum(ref_floor, tar_floor)
        self.m_spectrogram_ref -= low_floor
        self.m_spectrogram_tar -= low_floor
        self.m_dynamic_range = np.max(self.m_spectrogram_ref)

    def create_reference_patches(self):
        """
        this processing block breaks the reference spectrogram into non-overlapping "patches" of small images
        these images will be used as the basis of quality measurement
        """
        patch_indices_ref = np.arange(self.m_patch_size/2, np.size(self.m_spectrogram_ref, 1) - self.m_patch_size,
                                      self.m_patch_size).astype('int64')
        if len(patch_indices_ref) == 0:  # in case the audio clip is shorter than 30 frames
            patch_indices_ref = [0]
        num_patches = len(patch_indices_ref)
        patches = []
        for i in range(0, num_patches):
            istart = patch_indices_ref[i]
            iend = patch_indices_ref[i] + self.m_patch_size
            patch = self.m_spectrogram_ref[:, istart:iend]
            assert(self.m_warp == 1)  # no warping --> another dimension for warp parameters is not necessary
            patches.append(patch)
        self.m_patches_ref = patches
        self.m_patch_indices_ref = patch_indices_ref

    def align_degarded_patches(self):
        """
        this processing block finds the most correlated target patch for each reference patch
        """
        max_slide_offset = max(np.size(self.m_spectrogram_tar, 1) - self.m_patch_size, 1)  # should not be negative
        num_patches = len(self.m_patches_ref)
        patch_corr = np.zeros((max_slide_offset, num_patches))  # weird implementation --> memory-wise
        patch_indices_tar = np.zeros((num_patches,))

        istart = 0
        for i in range(0, num_patches):
            if i > 0:
                istart = int(self.m_patch_indices_ref[i - 1] + self.m_patch_size/2)
            if i < num_patches - 1:
                iend = int(self.m_patch_indices_ref[i + 1] - self.m_patch_size/2)
            else:
                iend = max_slide_offset
            slide_offset = istart  # for each starting point, slide through all possible sample delay
            for j in range(istart, iend):
                patch_ref = self.m_patches_ref[i]  # no warping, only one patch
                patch_tar = self.m_spectrogram_tar[:, slide_offset:slide_offset+self.m_patch_size]
                patch_corr[j, i] = self._compute_nsim(patch_ref, patch_tar, self.m_dynamic_range)
                slide_offset += 1
            patch_indices_tar[i] = np.argmax(patch_corr[:, i])  # for each patch, save the corresponding max index
        # ==== assign output to class attributes
        self.m_patch_indices_tar = patch_indices_tar.astype("int64")

    def calculate_patch_similarity(self):
        """
        this processing block take the adjusted indices of target signal and compute aggregated NSIM value
        for the entire file (averaged)
        """
        # force align patches with large deviation
        for i in range(0, len(self.m_patch_indices_ref)):
            if abs(self.m_patch_indices_ref[i] - self.m_patch_indices_tar[i]) > 30:
                self.m_patch_indices_tar[i] = self.m_patch_indices_ref[i]
        # patch_deltas = np.subtract(self.m_patch_indices_ref, self.m_patch_indices_tar)
        num_patches = len(self.m_patches_ref)
        nsim_per_patch = np.zeros((num_patches,))
        for j in range(0, num_patches):
            # Note: since no warping, the inner for loop can be simplified
            istart = self.m_patch_indices_tar[j]
            iend = istart + self.m_patch_size
            if iend > np.size(self.m_spectrogram_tar, 1):
                nsim_per_patch[j] = 0
            else:
                patch_ref = self.m_patches_ref[j]
                patch_tar = self.m_spectrogram_tar[:, istart:iend]
                nsim_per_patch[j] = self._compute_nsim(patch_ref, patch_tar, self.m_dynamic_range)
        avg_nsim = np.mean(nsim_per_patch)
        self.m_feature = avg_nsim

    @staticmethod
    def _compute_nsim(patch_ref, patch_tar, dynamic_range):
        """
        this is the core of VISQOLAudio; given two patches (images), compute the neurogram similarity
        using an equation that is inspired by SSIM; the local similarity is computed within a gaussian window
        as a final step, all the local similarities are averaged to get one aggregated output
        Note: differ from SSIM, NSIM ignores "contrast" in the calculation of similarity. This is based on the
        subjective experiment results in [4]
        :param patch_ref: num_bands by num_frames patch of reference signal
        :param patch_tar: num_bands by num_frames patch of target signal
        :param dynamic_range: float, dynamic range of the reference signal (dB)
        :return: nsim --> float, the averaged similarity between two patches across different frequency bands
        """
        gaussian_window = np.asarray([[0.0113, 0.0838, 0.0113],
                                      [0.0838, 0.6193, 0.0838],
                                      [0.0113, 0.0838, 0.0113]])
        gaussian_window = np.divide(gaussian_window, np.sum(gaussian_window))  # normalize as a pdf
        k = [0.01, 0.03]
        c1 = np.power(k[0] * dynamic_range, 2)  # noise floor for luminescence
        c2 = np.power(k[1] * dynamic_range, 2) / 2  # noise floor for structure
        mu_ref = sigs.convolve2d(patch_ref, np.rot90(gaussian_window, k=2), mode="valid")
        mu_tar = sigs.convolve2d(patch_tar, np.rot90(gaussian_window, k=2), mode="valid")
        mu_sq_ref = np.power(mu_ref, 2)
        mu_sq_tar = np.power(mu_tar, 2)
        mu_product = np.multiply(mu_ref, mu_tar)
        sigma_sq_ref = sigs.convolve2d(np.power(patch_ref, 2), np.rot90(gaussian_window, k=2), mode="valid") - mu_sq_ref
        sigma_sq_tar = sigs.convolve2d(np.power(patch_tar, 2), np.rot90(gaussian_window, k=2), mode="valid") - mu_sq_tar
        sigma_sq_product = sigs.convolve2d(np.multiply(patch_ref, patch_tar),
                                           np.rot90(gaussian_window, k=2), mode="valid") - mu_product
        sigma_ref = np.multiply(np.sign(sigma_sq_ref), np.sqrt(np.abs(sigma_sq_ref)))
        sigma_tar = np.multiply(np.sign(sigma_sq_tar), np.sqrt(np.abs(sigma_sq_tar)))
        lumine = np.divide((2 * mu_product + c1), (mu_sq_ref + mu_sq_tar + c1))
        struct = np.divide((sigma_sq_product + c2), (np.multiply(sigma_ref, sigma_tar) + c2))
        term1 = np.multiply(np.sign(lumine), np.abs(lumine))
        term2 = np.multiply(np.sign(struct), np.abs(struct))
        nmap = np.multiply(term1, term2)
        patch_avg_sim_per_band = np.mean(nmap, axis=1)
        # the following part handles the situation when low frequency part has perfect match
        # but the high frequency part is missing!
        # this is exactly why VISQOLAudio showed a hard cutoff for LP anchor files
        if max(patch_avg_sim_per_band >= 0.999) == 1:
            non_perfect_avg_sim = patch_avg_sim_per_band[patch_avg_sim_per_band <= 0.999]
            if len(non_perfect_avg_sim) == 0:
                patch_avg_sim_per_band = np.ones(np.shape(patch_avg_sim_per_band))  # perfect score
            else:
                patch_avg_sim_per_band = np.append(non_perfect_avg_sim, 1)
        nsim = np.mean(patch_avg_sim_per_band)
        return nsim

    @staticmethod
    def _get_normalized_power_spectrogram(sig, window_size, overlap, fs, bfs):
        """
        from time-domain signal, compute spectrogram with selected frequency bins, and normalize the power
        :param sig: float vector, input signal
        :param window_size: int, window size for computing STFT
        :param overlap: int, overlap size between consecutive windows
        :param fs: float, sampling rate in Hz
        :param bfs: float vector, center frequencies of the selected frequency bins
        :return: spec_db --> power spectrogram of the selected frequency bins (dB)
        """
        # f, t, spec = sigs.spectrogram(sig, fs, np.hamming(window_size), nfft=window_size, noverlap=overlap)
        spec = lr.core.stft(sig, n_fft=window_size, hop_length=window_size-overlap, window='hamming')
        df = fs / window_size
        bfs_bin_index = np.divide(bfs, df).astype('int64')
        spec = abs(spec[bfs_bin_index, :])
        spec = np.maximum(spec, 2.22e-16)
        spec = np.divide(spec, np.max(spec))  # normalized to 1
        spec_db = 20 * np.log10(spec)
        return spec_db

    @staticmethod
    def _scale_signal(sig, spl_required):
        """
        scale signal based on the required SPL (dB) and its current SPL
        Note: this function is similar to CepcorrFeatureExtractor._match_rms_y_to_x()
        :param sig: vector, input signal
        :param spl_required: required SPL in dB
        :return: vector, a linearly scaled signal
        """
        spl_reference = 20 * np.log10(np.sqrt(np.mean(np.power(sig, 2))) / 20e-6)
        scale_factor = np.power(10, (spl_required - spl_reference)/20)
        sig_scaled = np.multiply(sig, scale_factor)
        return sig_scaled

    def extract_features(self):
        self.preprocess_signals()
        # ==== get spectrograms
        self.compute_both_sig_spectrograms()
        # ==== create reference signal patches
        self.create_reference_patches()
        # ==== align degraded patches
        self.align_degarded_patches()
        # ==== calculate patch similarity
        self.calculate_patch_similarity()
        return self.m_feature


"""
===========================================================
===============  Derived classes PSMt =====================
===========================================================
"""


class PsmtFeatureExtractor(BaseFeatureExtractor):
    """
    Psmt is a selected feature from PEMO-Q for determining perceptual quality of speech/audio.
    the basic concept is described in [1][2]
    this feature represents correlation between internal representations of target and reference signals;
    the internal representation is a modulation filterbank based tensor that highlights the temporal fluctuations
    within the signals
    [1] Dau, T., and Kollmeier, B., Modelling auditory processing of amplitude modulation I. Detection and masking,
        with narrow-band carriers, JASA, Vol.102, Issue 5, p.2892, 1997
    [2] Dau, T., and Kollmeier, B., Modelling auditory processing of amplitude modulation II. Spectral and temporal
        integration, JASA, Vol.102, Issue 5, p.2906, 1997
    [3] Huber, R., and Kollmeier B., PEMO-Q -- A new method for objective audio quality assessment using a model,
        of auditory perception, IEEE/ACM TASLP, 14(6), 1902-1911, 2006
    This is a python implementation is based on the PEASS matlab implementation
    [4] http://bass-db.gforge.inria.fr/peass/PEASS-Software.html, last access: 2019.12

    The entry function is "self.extract_features()"
    """
    m_sig_internal_ref = []
    m_sig_internal_tar = []
    m_fs_internal = 800  # same as in [4]
    m_psmt_local = []
    m_feature = 0.0

    def compute_both_internal_representations(self):
        """
        this processing block takes both target and reference signal and compute their internal representations,
        respectively.
        """
        assert (len(self.m_ref_sig) == len(self.m_tar_sig))
        self.m_sig_internal_ref = self._compute_pemo_internal(self.m_ref_sig, self.m_fs, self.m_fs_internal)
        self.m_sig_internal_tar = self._compute_pemo_internal(self.m_tar_sig, self.m_fs, self.m_fs_internal)

    def compute_pemo_metric(self):
        """
        this processing block takes two internal representations and compute thee cross-correlation as in [3]
        both per frame (psmt_local) and per file feature are stored as class attributes
        """
        # ==== assimilation
        self.m_sig_internal_tar = self._compute_assimilation(self.m_sig_internal_ref, self.m_sig_internal_tar)
        # ==== local PSMt
        self.m_psmt_local = self._get_local_psmt(self.m_sig_internal_ref, self.m_sig_internal_tar, self.m_fs_internal)
        # ==== global PSMt
        self.m_feature = self._get_global_psmt(self.m_psmt_local, self.m_sig_internal_tar, self.m_fs_internal)

    @staticmethod
    def _compute_assimilation(sig_internal_ref, sig_internal_tar):
        """
        assimilation is based on the assumption that "additional content is less annoying than missing content"
        whenever the degraded signal has a lower value, use the actual value from the reference signal to smooth
        out the entry
        Note: in [3], assimilation scaling factor is [0.5, 0.5], whereas here it is [0.25, 0.75]
        :param sig_internal_ref: num_bands by num_samples by num_mod, internal representation of reference signal
        :param sig_internal_tar: num_bands by num_samples by num_mod, internal representation of target signal
        :return: sig_internal_tar --> assimilated internal representation of target signal
        """
        assim_loc = (np.abs(sig_internal_tar) < np.abs(sig_internal_ref))
        sig_internal_tar[assim_loc] = 0.25 * sig_internal_ref[assim_loc] + 0.75 * sig_internal_tar[assim_loc]
        return sig_internal_tar

    @staticmethod
    def _get_local_psmt(sig_internal_ref, sig_internal_tar, fs):
        """
        this function computes the per frame cross-correlation on the internal representations
        :param sig_internal_ref: num_bands by num_samples by num_mod, internal representation of reference signal
        :param sig_internal_tar: num_bands by num_samples by num_mod, internal representation of target signal
        :param fs: sampling rate in Hz
        :return: psmt_local --> perceptual similarity measure per frame
        """
        # ==== initialization
        num_bands, num_samples_mod, num_mod = np.shape(sig_internal_ref)
        frame_size = int(min(num_samples_mod, 0.1 * fs))
        num_frames = int(num_samples_mod / frame_size)
        num_samples_mod = int(num_frames * frame_size)
        # ==== truncate representations
        sig_internal_ref = sig_internal_ref[:, 0:num_samples_mod, :]
        sig_internal_tar = sig_internal_tar[:, 0:num_samples_mod, :]
        psmt_local = np.zeros((num_frames,))
        l_psm = np.zeros((num_mod, ))
        l_nms = np.zeros((num_mod, ))
        # ==== compute cross-correlation
        for t in range(0, num_frames):
            for m in range(0, num_mod):
                istart = int(t * frame_size)
                iend = int(istart + frame_size)
                l_ref = sig_internal_ref[:, istart:iend, m]
                l_ref = np.ndarray.flatten(l_ref) - np.mean(l_ref)
                l_tar = sig_internal_tar[:, istart:iend, m]
                l_tar = np.ndarray.flatten(l_tar)
                l_nms[m] = np.dot(l_tar, l_tar)
                l_tar = l_tar - np.mean(l_tar)
                # check the validity of denominator to avoid invalid value from division
                if np.sqrt(np.multiply(np.dot(l_ref, l_ref), np.dot(l_tar, l_tar))) != 0:
                    l_psm[m] = np.divide(np.dot(l_ref, l_tar),
                                         np.sqrt(np.multiply(np.dot(l_ref, l_ref), np.dot(l_tar, l_tar))))
            if not np.isnan(np.dot(l_psm, l_nms) / np.sum(l_nms)):  # handling the NAN problem here
                psmt_local[t] = np.dot(l_psm, l_nms) / np.sum(l_nms)
        return psmt_local

    @staticmethod
    def _get_global_psmt(psmt_local, sig_internal_tar, fs):
        """
        this function computes the aggregated PSMT using weighted moving average
        :param psmt_local: vector, num_frames by 1, perceptual similarity measure per frame
        :param sig_internal_tar: num_bands by num_samples by num_mod, internal representation of target signal
        :param fs: sampling rate in Hz
        :return: psmt_global --> perceptual similarity measure of the entire audio clip
        """
        num_bands, num_samples_mod, num_mod = np.shape(sig_internal_tar)
        frame_size = int(min(num_samples_mod, 0.1 * fs))
        i_len = fs
        num_frames = len(psmt_local)
        sig_internal_tar = np.sum(np.sum(np.power(sig_internal_tar, 2), axis=0), axis=1)
        rms = np.zeros((num_frames,))
        for t in range(0, num_frames):
            istart = int(max(0, (t - 0.5) * frame_size - 0.5 * i_len))  # don't understand this adjustment, why 0.5 sec?
            iend = int(min(num_samples_mod, (t - 0.5) * frame_size + 0.5 * i_len))  # it is computing rms within 1 sec?
            l_test = sig_internal_tar[istart:iend]
            rms[t] = np.mean(l_test)
        index = np.argsort(psmt_local)
        rms = rms[index]
        rms = np.cumsum(rms)
        index = index[(rms >= 0.5 * rms[-1])]
        psmt_global = psmt_local[index[0]]
        return psmt_global

    @staticmethod
    def _compute_pemo_internal(sig, fs, fs_internal):
        """
        this function computes PEMO internal representation given an input signal (mono). Note that this implementation
        is based on [1][3] and the result may vary from the matlab implementation in [4]. For more accurate results,
        please refer to [3] and [4] for more details
        :return: sig_internal --> internal representation of input signal, num_bands by num_samples by num_mod tensor
        """
        # ==== scaling
        sig = 10 * sig
        cfs, bws = PsmtFeatureExtractor._get_cfs_and_bws()
        sig_gammatone = PsmtFeatureExtractor._apply_gammatone_filterbank(sig, fs, cfs, bws)
        # ==== haircell model
        sig_haircell = PsmtFeatureExtractor._apply_haircell_model(sig_gammatone, fs)
        # ==== hearing threshold + adaptation loops
        sig_adapt = PsmtFeatureExtractor._apply_thres_and_adaptation(sig_haircell, fs)
        # ==== modulation filtering
        sig_internal = PsmtFeatureExtractor._compute_modulation_filterbank(sig_adapt, fs, fs_internal)
        return sig_internal

    @staticmethod
    def _compute_modulation_filterbank(sig_mat, fs, fs_internal):
        """
        this function filters the entire signal matrix using different modulation frequencies as defined in [3][4]
        :param sig_mat: num_bands by num_samples matrix of filterbank signals
        :param fs: sampling rate in Hz
        :return: sig_mod_env --> num_bands by num_samples_mod by num_mod, modulation filtered signal tensor
        """
        # ==== initialization
        sig_resamp = PsmtFeatureExtractor._resample_sig_mat(sig_mat, fs_internal, fs)
        cfs_mod = [0, 5, 10, 16.6667, 27.7778, 46.2963, 77.1605, 128.6008]
        bws_mod = [5, 5, 5, 8.3333, 13.8889, 23.1481, 38.5802, 64.3004]
        num_mod = len(cfs_mod)
        num_bands = np.size(sig_mat, 0)
        num_samples_mod = np.size(sig_resamp, 1)
        sig_mod_complex = np.zeros((num_bands, num_samples_mod, num_mod), dtype=complex)
        sig_mod_env = np.zeros((num_bands, num_samples_mod, num_mod))
        # ==== modulation filtering
        for i in range(0, num_mod):
            gain = np.exp(-np.pi * bws_mod[i] / fs_internal)
            sig_mod_complex[:, :, i] = sigs.lfilter([1 - gain],
                                                    [1, -gain * np.exp(2j * np.pi * cfs_mod[i] / fs_internal)],
                                                    sig_resamp)
            if cfs_mod[i] > 10:
                sig_mod_env[:, :, i] = np.abs(sig_mod_complex[:, :, i])
            else:
                sig_mod_env[:, :, i] = np.real(sig_mod_complex[:, :, i])
        return sig_mod_env

    @staticmethod
    def _resample_sig_mat(sig_mat, target_fs, original_fs):
        """
        this function resamples multiple signals from the gammatone filterbank in time axis
        :param sig_mat: num_bands by num_samples matrix of filterbank signals
        :param target_fs: target sampling rate in Hz
        :param original_fs: original sampling rate in Hz
        :return: sig_resamp --> num_bands by num_samples_new, resampled signal matrix
        """
        num_bands = np.size(sig_mat, 0)
        sig_resamp = []
        for i in range(0, num_bands):
            sig_resamp.append(lr.core.resample(sig_mat[i, :], orig_sr=original_fs, target_sr=target_fs, res_type='kaiser_fast'))
        return np.asarray(sig_resamp)

    @staticmethod
    def _get_cfs_and_bws():
        """
        this function returns the center frequency and bandwidth of gammatone filters that are close to the
        definition in PEMO-Q matlab implementation; as specified in [3] and [4], the f_min and f_max are
        235 Hz and 14500 Hz, respectively. All the associated values are precomputed and hard-coded
        """
        ear_q = 9.26449
        min_bw = 24.7
        order = 4
        cfs = np.asarray([236.338327906829, 289.356816182397, 348.417991091494, 414.210556951023, 487.501711910993,
                          569.146094147655, 660.095747682888, 761.411224039868, 874.273949189948, 1000.00000000000,
                          1140.05545082553, 1296.07346920405, 1469.87335999932, 1663.48178006761, 1879.15637082724,
                          2119.41208430771, 2387.05050966279, 2685.19254212010, 3017.31477531529, 3387.29004137697,
                          3799.43257149467, 4258.54830358123, 4769.99092366019, 5339.72429446944, 5974.39199925237,
                          6681.39481167720, 7468.97699525087, 8346.32243855292, 9323.66174730949, 10412.3915420933,
                          11625.2073527641, 12976.2516593132, 14481.2788053967])
        bws = 1.019 * 2 * np.pi * (((cfs / ear_q)**order + min_bw**order)**(1 / order))
        return cfs, bws

    @staticmethod
    def _apply_gammatone_filterbank(sig, fs, cfs, bws):
        """
        this function implements the gammatone filterbank using functions in pyACA
        for more information, please see: https://github.com/alexanderlerch/pyACA/tree/master/pyACA
        :param sig: a vector of input signal to be filtered
        :param fs: sampling rate in Hz
        :param cfs: a vector (ndarray) of center frequencies of the filterbank
        :param bws: a vector (ndarray) of bandwidth of the filterbank
        :return: sig_mat --> num_bands by num_samples matrix of filterbank signals
        """
        # ==== init params
        num_bands = len(cfs)
        num_samples = len(sig)
        T = 1 / fs
        sig_mat = np.zeros((num_bands, num_samples))

        # ==== get filter coefficients
        [coef_b, coef_a] = ToolGammatoneFb.getCoeffs(cfs, bws, T)

        # ==== cascaded filtering
        for k in range(0, num_bands):
            sig_mat[k, :] = sig
            for j in range(0, 4):
                sig_mat[k, :] = sigs.lfilter(coef_b[j, :, k], coef_a[j, :, k], sig_mat[k, :])
        return sig_mat

    @staticmethod
    def _apply_thres_and_adaptation(sig_mat, fs):
        """
        this function re-implements the adaptation for loops in pemo_internal.m from PEASS software [4]
        Note: this is super slow again... sample by sample operation is very slow in python
        Update: after using njit from Numba, it becomes fast again!
        :param sig_mat: sig_mat: num_bands by num_samples, matrix of gammatone filtered signal
        :param fs: sampling rate
        :return: sig_mat --> thresholded and adapted signal matrix
        """
        num_bands, num_samples = np.shape(sig_mat)
        db_range = 100
        thres = np.power(10, (-db_range/20))
        bw = np.divide(1.0, np.multiply(np.pi, [0.005, 0.05, 0.129, 0.253, 0.5]))
        sig_mat = np.maximum(sig_mat, thres)
        sig_mat = PsmtFeatureExtractor._use_jit_adaptation(sig_mat, bw, thres, db_range, num_bands, num_samples, fs)
        return sig_mat

    @staticmethod
    @njit
    def _use_jit_adaptation(sig_mat, bw, thres, db_range, num_bands, num_samples, fs):
        s_thres = thres
        for b in range(0, num_bands):
            s_thres = thres
            for s in range(0, 5):
                gain = np.exp(-np.pi * bw[s]/fs)
                s_thres = np.sqrt(s_thres)
                factor = s_thres
                for t in range(0, num_samples):
                    sig_mat[b, t] = np.divide(sig_mat[b, t], factor)
                    factor = max((1 - gain) * sig_mat[b, t] + gain * factor, s_thres)
        sig_mat = np.multiply(db_range/(1 - s_thres), np.subtract(sig_mat, s_thres))
        return sig_mat

    @staticmethod
    def _apply_haircell_model(sig_mat, fs):
        """
        apply half-wave rectification and 1k lowpass (1st order IIR) on each channel of gammatone filtered signal
        :param sig_mat: num_bands by num_samples, matrix of gammatone filtered signal
        :param fs: sampling rate
        :return: sig_mat --> processed signal matrix
        """
        num_bands = np.size(sig_mat, 0)
        for i in range(0, num_bands):
            # ==== half-wave rectification
            sig = np.maximum(sig_mat[i, :], 0.0)
            # ==== 1k lowpass filtering
            f_cutoff = 1000
            [b_lp, a_lp] = sigs.butter(1, f_cutoff / (0.5 * fs))
            sig_mat[i, :] = sigs.lfilter(b_lp, a_lp, sig)
        return sig_mat

    def extract_features(self):
        self.preprocess_signals()
        # ==== compute internal representations
        self.compute_both_internal_representations()
        # ==== compute psmt
        self.compute_pemo_metric()
        return self.m_feature


"""
===========================================================
===============  End of Derived classes ===================
===========================================================
"""


class SmaqFeatureExtractor:
    """
    This class extracts all relevant features for SMAQ predictor. Given two audio signals (target and reference),
    this class will return a vector of different audio features that correspond to the perceptual audio quality of the
    target signal. The signals should be mono (single channel), and the sampling rate is 48kHz.

    The entry function is "self.extract_all_features()"
    """
    m_all_features = []
    m_feature_extractors = []

    def __init__(self, tar_sig, ref_sig, fs):
        assert(fs == 48000)
        self.m_tar_sig = tar_sig
        self.m_ref_sig = ref_sig
        self.m_fs = fs

    def extract_all_features(self):
        """
        This processing block instantiate all selected features and subsequently combine them into a final feature
        vector
        """
        self._instantiate_feature_extractors()
        self._stack_all_features()
        return self.m_all_features

    def _stack_all_features(self):
        """
        This function combines all features into a np.shape = (num_features, ) ndarray vector
        """
        pool = mp.Pool(mp.cpu_count())
        list_of_extractors = self.m_feature_extractors
        all_features = pool.map(self._get_feature, (extractor for extractor in list_of_extractors))
        pool.close()
        self.m_all_features = np.asarray(all_features)

    @staticmethod
    def _get_feature(extractor):
        return extractor.extract_features()

    def _instantiate_feature_extractors(self):
        """
        This function prepares a list of feature objects. To add new features, simply add additional feature object
        to the list
        """
        cepcorr = CepcorrFeatureExtractor(self.m_tar_sig, self.m_ref_sig, self.m_fs)
        nmr = NmrFeatureExtractor(self.m_tar_sig, self.m_ref_sig, self.m_fs)
        psmt = PsmtFeatureExtractor(self.m_tar_sig, self.m_ref_sig, self.m_fs)
        nsim = NsimFeatureExtractor(self.m_tar_sig, self.m_ref_sig, self.m_fs)
        self.m_feature_extractors = [cepcorr, nmr, psmt, nsim]


if __name__ == "__main__":
    print("running main() directly")
    import time
    tic = time.time()
    # ==== quick and dirty way of testing the implementation
    # load audio signals for testing
    TARPATH = pkg_resources.resource_filename(__name__, "data/tar.wav")
    REFPATH = pkg_resources.resource_filename(__name__, "data/ref.wav")
    FS = 48000
    target_sig, sr = lr.load(TARPATH, sr=FS, mono=True)
    reference_sig, _ = lr.load(REFPATH, sr=FS, mono=True)
    smaq_feat = SmaqFeatureExtractor(target_sig, reference_sig, FS)
    feature_vec = smaq_feat.extract_all_features()
    print(feature_vec)
    print(np.shape(feature_vec))
    toc = time.time() - tic
    print("The entire process took %f seconds" % toc)
