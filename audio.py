from dataclasses import dataclass, field
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile
from librosa.filters import mel
from scipy.fft import fft
from scipy.signal.windows import get_window

from deep_learning.util import JSONObject


@dataclass
class FrameParameter(JSONObject):
    frame_len: int
    frame_shift: int
    padding: bool = field(default=True)

    @classmethod
    def from_ms(
        cls, fs: float, frame_len_ms: float, frame_shift_ms: float, padding: bool = True
    ) -> "FrameParameter":
        frame_len = int(fs * frame_len_ms * 1000)
        frame_shift = int(fs * frame_shift_ms * 1000)

        return cls(frame_len, frame_shift, padding=padding)

    def frame_len_ms(self, fs: float) -> float:
        return self.frame_len / fs / 1000.0

    def frame_shift_ms(self, fs: float) -> float:
        return self.frame_shift / fs / 1000.0


@dataclass
class SpectrumParameter(JSONObject):
    fft_point: int
    window: str

    @classmethod
    def from_ms(
        cls, fs: float, fft_point_ms: float, window: str
    ) -> "SpectrumParameter":
        fft_point = int(fs * fft_point_ms * 1000)

        return cls(fft_point, window)

    def get_window(self, frame_len: int):
        return get_window(self.window, frame_len)

    def fft_point_ms(self, fs: float) -> float:
        return self.fft_point / fs / 1000.0


class FrameSeries:
    feature_names = [
        "waveform",
        "spectrum",
        "log spectrum",
        "mel frequency",
        "mel spectrum",
    ]

    def __init__(self, frame_series: np.ndarray) -> None:
        self.__frame_series = frame_series

    @property
    def frame_series(self) -> np.ndarray:
        return self.__frame_series

    @property
    def num_frame(self) -> int:
        return self.__frame_series.shape[0]

    @classmethod
    def from_param(
        cls, time_series: np.ndarray, param: FrameParameter, dtype=np.float32
    ) -> "FrameSeries":
        if param.padding:
            pad = param.frame_len // 2
            time_series = np.pad(time_series, pad, mode="reflect")

        num_frame = 1 + (len(time_series) - param.frame_len) // param.frame_shift
        frames = []
        for i in range(num_frame):
            start, end = cls.edge_point(i, param.frame_len, param.frame_shift)
            frames.append(time_series[start:end])

        frames_np: np.ndarray = np.array(frames, dtype=dtype)

        return cls(frames_np)

    @staticmethod
    def edge_point(index: int, frame_len: int, frame_shift: int) -> Tuple[int, int]:
        start = index * frame_shift
        end = start + frame_len
        return start, end

    @staticmethod
    def feature_series(
        feature_name: str,
        series: "FrameSeries",
        include_nyquist=True,
        spectrum_param: Optional[SpectrumParameter] = None,
        fs: Optional[int] = None,
        mel_bins: Optional[int] = None,
    ) -> "FrameSeries":
        if feature_name not in FrameSeries.feature_names:
            raise RuntimeError(
                "feature_name is one of {}.".format(FrameSeries.feature_names)
            )

        if feature_name == "waveform":
            return series

        if spectrum_param is None:
            raise RuntimeError("spectrum_param must not be None.")

        if feature_name == "spectrum":
            return series.to_amplitude_spectrum(
                spectrum_param, include_nyquist=include_nyquist
            )

        if feature_name == "log spectrum":
            return series.to_amplitude_spectrum(
                spectrum_param, include_nyquist=include_nyquist
            ).to_dB()

        if feature_name == "mel frequency":
            return series.to_amplitude_spectrum(
                spectrum_param, include_nyquist=include_nyquist
            ).to_mel_frequency()

        if fs is None:
            raise RuntimeError("fs must not be None.")
        if mel_bins is None:
            raise RuntimeError("mel_bins must not be None.")

        # メルスペクトルが最後に残る
        assert feature_name == "mel spectrum"
        return series.to_mel_spectrum(spectrum_param, fs, mel_bins)

    def to_amplitude_spectrum(
        self, param: SpectrumParameter, include_nyquist=True
    ) -> "FrameSeries":
        window_func = param.get_window(self.frame_series.shape[1])
        frames = window_func * self.frame_series
        to_spectrum = lambda frame: np.abs(fft(frame, n=param.fft_point))
        spectrum = to_spectrum(frames)
        if include_nyquist:
            # ナイキスト周波数 + 1点分を返す
            return FrameSeries(spectrum[:, : param.fft_point // 2 + 1])
        else:
            return FrameSeries(spectrum[:, : param.fft_point // 2])

    def to_dB(self) -> "FrameSeries":
        to_dB = lambda frame: 20 * np.log10(
            frame, out=np.zeros_like(frame), where=frame != 0
        )

        return FrameSeries(to_dB(self.frame_series))

    def to_mel_frequency(self) -> "FrameSeries":
        to_mel = lambda freq: 2595.0 * np.log(freq / 700.0 + 1)

        return FrameSeries(to_mel(self.frame_series))

    def to_mel_spectrum(
        self, param: SpectrumParameter, fs: float, mel_bins: int
    ) -> "FrameSeries":
        spectrum = self.to_amplitude_spectrum(param).frame_series
        power = spectrum ** 2

        filter = mel(sr=fs, n_fft=param.fft_point, n_mels=mel_bins)

        return FrameSeries(np.dot(filter, power.T).T)

    def to_image(self, time_len: int) -> np.ndarray:
        padded :np.ndarray = np.pad(
            self.frame_series,
            ((0, (time_len - (self.frame_series.shape[0] % time_len))), (0, 0)),
        )

        return padded.reshape(padded.shape[0] // time_len, time_len, padded.shape[1])


class WavFile:
    def __init__(self, data: np.ndarray, fs: int, dtype=np.float32) -> None:
        self.fs = fs
        self.data = np.array(data, dtype=dtype)

    @classmethod
    def read(cls, path: str, fs=None, dtype=np.float32) -> "WavFile":
        data, fs = librosa.core.load(path, sr=fs)
        return cls(data, fs, dtype=dtype)

    def resample(self, target_fs: int) -> "WavFile":
        if self.fs == target_fs:
            return self

        resampled: np.ndarray = librosa.core.resample(
            y=self.data, orig_sr=self.fs, target_sr=target_fs
        )

        return WavFile(resampled, target_fs, dtype=resampled.dtype)

    def save(self, path: str, target_bits: int = 16) -> None:
        soundfile.write(path, self.data, self.fs, "PCM_{}".format(target_bits))
