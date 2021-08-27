#!/usr/bin/env python
"""
Predictor should be straightforward & easy to use
for example, in python:
predictor = smaqPredictor()
smaq_score, raw_scores, raw_features = predictor.predict(tar_path, ref_path)
"""
import filecmp
import json
import os
import time
import warnings

import librosa as lr
import numpy as np
import pkg_resources
import runez
import soundfile as sf
from scipy.stats import gmean
from sklearn.externals import joblib
from tensorflow.keras.models import load_model

from smaq_cli.feature_extractor import SmaqFeatureExtractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduce tf warning messages


class SmaqPredictor:
    """
    entry function is self.predict()
    """
    MODEL_PATH = pkg_resources.resource_filename(__name__, "data/smaq_mtl_model.h5")
    SCALER_PATH = pkg_resources.resource_filename(__name__, "data/minmax_scaler.save")
    RESOURCE_TAR = pkg_resources.resource_filename(__name__, "data/tar.wav")
    RESOURCE_REF = pkg_resources.resource_filename(__name__, "data/ref.wav")

    def __init__(self):
        """
        attempt to load pre-trained model and scaler during init stage.
        user may override the preset by invoking self.set_model_scaler_path()
        """
        self.m_segment_size_sec = 20  # default to 20 seconds --> can be changed in the future
        self.m_duration_thres_sec = 60  # need to figure out the best threshold
        self.m_fs = 48000
        self.m_tar_info = None
        self.m_ref_info = None
        self.m_tar_path = None
        self.m_ref_path = None
        self.m_smaq_model = None
        self.m_smaq_scaler = None
        self.IS_SAME_FILE = None
        self.IS_TOO_LONG = None
        self._reset_scores()
        self._load_model_and_scalers()

    def _reset_scores(self):
        # ==== aggregated output attributes
        self.m_smaq_final = None
        self.m_smaq_raw = []
        self.m_smaq_features = []
        # ==== time series output attributes
        self.m_smaq_series = []
        self.m_smaq_series_time_stamp = []
        self.m_smaq_raw_series = []
        self.m_smaq_features_series = []
        # ==== derived stats
        self.m_smaq_max = None
        self.m_smaq_max_time_stamp = None
        self.m_smaq_min = None
        self.m_smaq_min_time_stamp = None
        self.m_smaq_std = None

    def check_audio_info(self, tar_path, ref_path):
        """
        check basic information of the provided audio signals and display warnings if necessary
        checklist:
        - sampling rate (48kHz)
        - formats (wave)
        - channels (mono)
        - durations?
        """
        self.m_ref_path = ref_path
        self.m_tar_path = tar_path
        tar_info = sf.info(self.m_tar_path)
        ref_info = sf.info(self.m_ref_path)
        self.IS_TOO_LONG = False
        self.IS_SAME_FILE = False
        if tar_info.samplerate != 48000:
            warnings.warn("target signal sampling rate is not 48kHz, it will be resampled")
        if ref_info.samplerate != 48000:
            warnings.warn("reference signal sampling rate is not 48kHz, it will be resampled")
        if tar_info.format != ref_info.format:
            warnings.warn("Target and reference signals are in different formats")
        if tar_info.channels >= 2:
            warnings.warn("target signal has channel number >= 2, it will be downmixed to mono")
        if ref_info.channels >= 2:
            warnings.warn("reference signal has channel number >= 2, it will be downmixed to mono")
        if tar_info.duration != ref_info.duration:
            warnings.warn("Target and reference signals have different durations; longer one will be truncated")
        if min(tar_info.duration, ref_info.duration) >= self.m_duration_thres_sec:
            self.IS_TOO_LONG = True
            warnings.warn("Files are longer than %d seconds; using segmental SMAQ" % self.m_duration_thres_sec)
        if min(tar_info.duration, ref_info.duration) <= 0.48:  # minimum duration due to VISQOL patch size
            warnings.warn("Files should be at least 0.48 second; unreliable results might be returned")
        if filecmp.cmp(self.m_tar_path, self.m_ref_path, shallow=False):
            self.IS_SAME_FILE = True
            warnings.warn("Files are bit-for-bit identical")
        print("=============================================================")
        self.m_tar_info = tar_info
        self.m_ref_info = ref_info

    def set_model_scaler_path(self, new_model_filepath, new_scaler_filepath):
        """
        set the model and feature scaler filepath when testing different pre-trained models
        subsequently load up the model and scaler
        :param new_model_filepath: str, absolute path to the pre-trained model .h5 file
        :param new_scaler_filepath: str, aboslute path to the pre-trained feature scaler .save file
        :return:
        """
        self.MODEL_PATH = new_model_filepath
        self.SCALER_PATH = new_scaler_filepath
        self._load_model_and_scalers()

    def set_segment_size(self, new_segment_size_sec):
        """
        set the segment size for computing segmental SMAQ score (non-overlapping window for now)
        :param new_segment_size_sec: int, the duration of a unit audio segment
        """
        self.m_segment_size_sec = new_segment_size_sec

    def set_duration_threshold(self, new_duration_thres_sec):
        """
        set the threshold for computing segmental SMAQ; when the audio signal is longer than the threshold,
        use segmental SMAQ instead
        """
        self.m_duration_thres_sec = new_duration_thres_sec

    def _load_model_and_scalers(self):
        """
        load pre-trained model and feature scaler (min-max scaler)
        """
        self.m_smaq_model = load_model(self.MODEL_PATH)
        self.m_smaq_scaler = joblib.load(self.SCALER_PATH)

    def compute_smaq_score(self, feature_vec):
        """
        given a feature vector, compute SMAQ output and final mos score
        :param feature_vec: float, ndarray, four SMAQ features (Cepcorr, NMR, PSMt, NSIM)
        :return: final_mos --> SMAQ score, a continuous value between 1 and 5 (MOS scale)
                 raw_output--> six output values from the SMAQ multitask learning model
        1) HAAQI predictor: a value between 0 and 1 representing the simulated HAAQI score
        2) PEAQ predictor: a value between 0 and 1 representing the simulated PEAQ score
        3) PEMO-Q predictor: a value between 0 and 1 representing the simulated PEMO-Q score
        4) VISQOLAudio predictor: a value between 0 and 1 representing the simulated VISQOLAudio score
        5) Bitrate indicator: a value between 0 and 1 (0: lower than 48kbps, 1: higher than 48kbps)
        6) Minimum score indicator:  a value between 0 and 1 (0: less than 3.0, 1: greater or equal to 3.0)
        """
        feature_vec = np.expand_dims(feature_vec, axis=0)
        feature_vec = self.m_smaq_scaler.transform(feature_vec)
        raw_output = self.m_smaq_model.predict(feature_vec)
        final = gmean(raw_output[0, 0:4])  # TODO: investigate a better way to aggregate all learned tasks
        final_mos = min(final * 4.0 + 1.0, 5.0)
        return final_mos, raw_output

    @staticmethod
    def _is_perfect_silence(sig):
        """
        a simple silence detection that checks the mean & std of a signal
        :param sig: float vector, (num_samples, )
        :return: bool
        """
        mu = np.mean(sig)
        std = np.std(sig)
        if mu == 0.0 and std == 0.0:
            return True
        else:
            return False

    @staticmethod
    def extract_features(tar_sig, ref_sig, fs):
        """
        extract four SMAQ features from a pair of target and reference signals
        :param tar_sig: float vector, (num_samples, ), target (compressed) signal
        :param ref_sig: float vector, (num_samples, ), reference (uncompressed) signal
        :param fs: int, sampling rate, default value = 48000Hz
        :return: smaq_features --> float, ndarray, four SMAQ features (Cepcorr, NMR, PSMt, NSIM)
        """
        smaq_extractor = SmaqFeatureExtractor(tar_sig, ref_sig, fs)
        smaq_features = smaq_extractor.extract_all_features()
        return smaq_features

    def _predict_entire_file(self):
        """
        this processing block computes the overall SMAQ value from the entire audio clip;
        """
        minimum_duration = min(self.m_tar_info.duration, self.m_ref_info.duration)
        tar_sig, sr = lr.core.load(self.m_tar_path, mono=True, offset=0, duration=minimum_duration, sr=self.m_fs)
        ref_sig, sr = lr.core.load(self.m_ref_path, mono=True, offset=0, duration=minimum_duration, sr=self.m_fs)
        if self._is_perfect_silence(tar_sig) or self._is_perfect_silence(ref_sig):
            warnings.warn("perfect silence detected, calculation skipped")
        else:
            raw_features = self.extract_features(tar_sig, ref_sig, self.m_fs)
            final_mos, raw_output = self.compute_smaq_score(raw_features)
            self.m_smaq_features_series.append(raw_features)
            self.m_smaq_raw_series.append(raw_output)
            self.m_smaq_series.append(final_mos)
            self.m_smaq_series_time_stamp.append(0.0)
            self._aggregate_smaq_time_series()  # in this case, there will only be one value in the lists

    def _predict_segmental(self):
        """
        For longer audio signals, a time-series of SMAQ score with a selected segment size will be computed
        (default = 20 seconds)
        """
        minimum_duration = min(self.m_tar_info.duration, self.m_ref_info.duration)
        num_segments = int(np.divide(np.floor(minimum_duration), self.m_segment_size_sec))
        for i in range(0, num_segments):
            print("processing segment %d..." % i)
            istart = i * self.m_segment_size_sec
            tar_buffer, sr = lr.core.load(self.m_tar_path, mono=True, offset=istart, duration=self.m_segment_size_sec,
                                          sr=self.m_fs)
            ref_buffer, sr = lr.core.load(self.m_ref_path, mono=True, offset=istart, duration=self.m_segment_size_sec,
                                          sr=self.m_fs)
            if self._is_perfect_silence(tar_buffer) or self._is_perfect_silence(ref_buffer):
                warnings.warn("segmental: perfect silence detected, calculation skipped")
            else:
                raw_features = self.extract_features(tar_buffer, ref_buffer, self.m_fs)
                final_mos, raw_output = self.compute_smaq_score(raw_features)
                self.m_smaq_features_series.append(raw_features)
                self.m_smaq_raw_series.append(raw_output)
                self.m_smaq_series.append(final_mos)
                self.m_smaq_series_time_stamp.append(istart)
        self._aggregate_smaq_time_series()

    def _aggregate_smaq_time_series(self):
        """
        For longer files, a SMAQ score per 20 sec segment will be computed. To get a final score that represents
        the whole file, a simple aggregation (average) method is currently used
        TODO: investigate the best temporal aggregation strategy
        """
        self.m_smaq_final = np.mean(self.m_smaq_series, axis=0)
        self.m_smaq_raw = np.mean(self.m_smaq_raw_series, axis=0)
        self.m_smaq_features = np.mean(self.m_smaq_features_series, axis=0)
        self.m_smaq_min = np.min(self.m_smaq_series)
        self.m_smaq_min_time_stamp = self.m_smaq_series_time_stamp[int(np.argmin(self.m_smaq_series))]
        self.m_smaq_max = np.max(self.m_smaq_series)
        self.m_smaq_max_time_stamp = self.m_smaq_series_time_stamp[int(np.argmax(self.m_smaq_series))]
        self.m_smaq_std = np.std(self.m_smaq_series)

    def predict(self, tar_path, ref_path):
        """
        predict the SMAQ score of the given audio clips
        :param tar_path: str, target signal filepath
        :param ref_path: str, reference signal filepath
        :return: m_smaq_final --> SMAQ score, a continuous value between 1 and 5 (MOS scale)
                 m_smaq_raw --> six output values from the SMAQ multitask learning model
                 m_smaq_features --> raw audio features for predicting smaq scores (cepcorr, nmr, psmt, nsim)
        """
        # ==== reset class attributes
        self._reset_scores()
        # ==== check audio and set flags
        self.check_audio_info(tar_path, ref_path)
        if self.IS_SAME_FILE:
            self.m_smaq_series.append(5.0)
            self.m_smaq_series_time_stamp.append(0.0)
            self.m_smaq_raw_series.append([1.0, 1.0, 1.0, 1.0])
            self.m_smaq_features_series.append([1.0, 1.0, 1.0, 1.0])
            self._aggregate_smaq_time_series()
        else:
            # ==== invoke predict based on audio duration
            if self.IS_TOO_LONG:
                self._predict_segmental()
            else:
                self._predict_entire_file()
        # ==== return final scores
        return self.m_smaq_final, self.m_smaq_raw, self.m_smaq_features

    def _generate_output_report(self):
        """
        generate a document dictionary using json template file
        """
        boiler_plate_path = pkg_resources.resource_filename(__name__, "data/document_template.json")
        with open(boiler_plate_path, 'r') as reader:
            self.m_report = json.loads(reader.read())
            # ==== modify values based on template
            self.m_report['metadata']['smaqVersion'] = runez.get_version("smaq_cli")
            self.m_report['tracks'][0]['overall_smaq_score'] = self.m_smaq_final
            self.m_report['tracks'][0]['segmental_smaq_scores'] = self.m_smaq_series
            self.m_report['tracks'][0]['segmental_time_stamps'] = self.m_smaq_series_time_stamp
            self.m_report['tracks'][0]['segmental_min_score'] = self.m_smaq_min
            self.m_report['tracks'][0]['segmental_min_time_stamp'] = self.m_smaq_min_time_stamp
            self.m_report['tracks'][0]['segmental_max_score'] = self.m_smaq_max
            self.m_report['tracks'][0]['segmental_max_time_stamp'] = self.m_smaq_max_time_stamp
            self.m_report['tracks'][0]['segmental_std'] = self.m_smaq_std

    def write_json_output(self, output_json_path):
        """
        This function writes the analysis report to a json file
        """
        self._generate_output_report()
        json_str = json.dumps(self.m_report, indent=4, sort_keys=True)
        fp = open(output_json_path, "w")
        fp.write(json_str)
        fp.close()


if __name__ == "__main__":
    print("running main() directly")
    tic = time.time()
    # ==== quick and dirty way of testing the implementation
    smaq_predictor = SmaqPredictor()
    smaq_predictor.set_duration_threshold(10)
    smaq_predictor.set_segment_size(1)
    tar_filepath = smaq_predictor.RESOURCE_TAR
    ref_filepath = smaq_predictor.RESOURCE_REF
    final_score, raw_score, features = smaq_predictor.predict(tar_filepath, ref_filepath)
    smaq_predictor.write_json_output('tmp.json')
    print("=============================================================")
    print("SMAQ score = %s" % final_score)
    print("raw score = %s" % raw_score)
    print("raw features = %s" % features)
    toc = time.time() - tic
    print("The entire process took %f seconds" % toc)
