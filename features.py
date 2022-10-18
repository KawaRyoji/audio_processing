from typing import Optional, Tuple, Union

import librosa
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal.windows import get_window


class FrameSeries:
    def __init__(
        self,
        frame_series: np.ndarray,
        frame_length: int,
        frame_shift: int,
        dB: Optional[bool] = None,
        power: Optional[bool] = None,
    ) -> None:
        self.__frame_series = frame_series
        self.__frame_length = frame_length
        self.__frame_shift = frame_shift
        self.__dB = dB
        self.__power = power

    @property
    def frame_length(self) -> int:
        return self.__frame_length

    @property
    def frame_shift(self) -> int:
        return self.__frame_shift

    @property
    def shape(self) -> Tuple:
        return self.__frame_series.shape

    @property
    def data(self) -> np.ndarray:
        return self.__frame_series

    @property
    def dB(self) -> Optional[bool]:
        return self.__dB

    @property
    def power(self) -> Optional[bool]:
        return self.__power

    @classmethod
    def edge_point(
        cls, index: int, frame_length: int, frame_shift: int
    ) -> Tuple[int, int]:
        start = index * frame_shift
        end = start + frame_length
        return start, end

    def to_patches(self, frames: int) -> np.ndarray:
        padded: np.ndarray = np.pad(
            self.data,
            ((0, (frames - (self.data.shape[0] % frames))), (0, 0)),
        )

        return padded.reshape(padded.shape[0] // frames, frames, padded.shape[1])

    def linear_to_dB(self) -> "FrameSeries":
        if self.dB is None:
            print("This feature can't be a dB value.")
            return self
        elif self.dB:
            print("This feature is already a dB value.")
            return self

        if self.power:
            dB_func = lambda frame: 10 * np.log10(
                frame, out=np.zeros_like(frame), where=frame != 0
            )
        else:
            dB_func = lambda frame: 20 * np.log10(
                frame, out=np.zeros_like(frame), where=frame != 0
            )

        return FrameSeries(
            dB_func(self.data),
            self.frame_length,
            self.frame_shift,
            dB=True,
            power=self.power,
        )

    def dB_to_linear(self) -> "FrameSeries":
        if self.dB is None or not self.dB:
            print("This feature is already a linear value.")
            return self

        if self.power:
            linear_func = lambda x: np.power(x / 10, 10)
        else:
            linear_func = lambda x: np.power(x / 20, 10)

        return FrameSeries(
            linear_func(self.data),
            self.frame_length,
            self.frame_shift,
            dB=False,
            power=self.power,
        )

    def linear_to_power(self) -> "FrameSeries":
        if self.power is None:
            print("This feature can't convert to power.")
            return self
        elif self.power:
            print("This feature is already converted to power.")
            return self
        elif self.dB:
            print("This feature is already a dB value.")
            return self

        return FrameSeries(
            np.power(self.data, 2),
            self.frame_length,
            self.frame_shift,
            dB=self.dB,
            power=True,
        )

    def power_to_linear(self) -> "FrameSeries":
        if self.power is None or not self.power:
            print("This feature is already linear value.")
            return self
        elif self.dB is None or self.dB:
            print("This feature is a dB value.")
            return self

        return FrameSeries(
            np.sqrt(self.data),
            self.frame_length,
            self.frame_shift,
            dB=self.dB,
            power=False,
        )

    def plot(
        self,
        up_to_nyquist=True,
        show=True,
        save_fig_path: Optional[str] = None,
        color_map: str = "magma",
    ) -> None:
        if up_to_nyquist:
            show_data: np.ndarray = self.data[:, : self.shape[1] // 2 + 1]
        else:
            show_data = self.data

        fig, ax = plt.subplots(
            dpi=100,
            figsize=(show_data.shape[0] / 100, show_data.shape[1] / 100),
        )

        ax.pcolor(show_data.T, cmap=color_map)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        if show:
            plt.show()

        if save_fig_path is not None:
            plt.savefig(save_fig_path)

        plt.close()

    def __len__(self) -> int:
        return self.__frame_series.__len__()


class Waveform(FrameSeries):
    @classmethod
    def from_param(
        cls,
        time_series: np.ndarray,
        frame_length: int,
        frame_shift: int,
        padding: bool = True,
        padding_mode: str = "reflect",
        dtype=np.float32,
    ) -> "Waveform":
        if padding:
            pad = frame_length // 2
            time_series = np.pad(time_series, pad_width=pad, mode=padding_mode)

        num_frame = 1 + (len(time_series) - frame_length) // frame_shift
        frames = []
        for i in range(num_frame):
            start, end = cls.edge_point(i, frame_length, frame_shift)
            frames.append(time_series[start:end])

        frames_np: np.ndarray = np.array(frames, dtype=dtype)

        return cls(frames_np, frame_length, frame_shift, dB=None, power=None)

    def to_spectrum(
        self,
        fft_point: Optional[int] = None,
        window: Union[str, np.ndarray] = "hann",
    ) -> "Spectrum":
        if type(window) is str:
            window_func: np.ndarray = get_window(window, self.frame_length)
        elif type(window) is np.ndarray:
            window_func = window
        else:
            raise TypeError("Type of window is str or np.ndarray")

        if fft_point is None:
            fft_point = self.shape[1]

        to_spectrum = lambda frame: np.fft.fft(frame, n=fft_point)
        spectrum = to_spectrum(self.data * window_func)

        return Spectrum(
            spectrum, self.frame_length, self.frame_shift, dB=None, power=None
        )


class Spectrum(FrameSeries):
    def to_amplitude(self) -> "AmplitudeSpectrum":
        return AmplitudeSpectrum(
            np.abs(self.data),
            self.frame_length,
            self.frame_shift,
            dB=False,
            power=False,
        )

    def to_phase(self) -> "PhaseSpectrum":
        return PhaseSpectrum(
            np.degrees(self.data),
            self.frame_length,
            self.frame_shift,
            dB=None,
            power=None,
        )


class AmplitudeSpectrum(FrameSeries):
    def to_cepstrum(self) -> "Cepstrum":
        if self.power or self.dB:
            raise RuntimeError("This feature can't convert to cepstrum.")

        return Cepstrum(
            np.real(
                np.fft.ifft(
                    np.log(
                        self.data, out=np.zeros_like(self.data), where=self.data != 0
                    )
                )
            ),
            self.frame_length,
            self.frame_shift,
            dB=None,
            power=None,
        )

    def to_mel(self, fs: int, bins: int) -> "MelSpectrum":
        filter = librosa.filters.mel(sr=fs, n_fft=self.shape[1], n_mels=bins)

        return MelSpectrum(
            np.dot(filter, self.data[:, : self.shape[1] // 2 + 1].T).T,
            self.frame_length,
            self.frame_shift,
            dB=self.dB,
            power=self.power,
        )


class PhaseSpectrum(FrameSeries):
    pass


class MelSpectrum(FrameSeries):
    def to_cepstrum(self) -> "MelCepstrum":
        if self.power or self.dB:
            raise RuntimeError("This feature can't convert to cepstrum.")

        return MelCepstrum(
            np.real(
                np.fft.ifft(
                    np.log(
                        self.data, out=np.zeros_like(self.data), where=self.data != 0
                    )
                )
            ),
            self.frame_length,
            self.frame_shift,
            dB=None,
            power=None,
        )


class Cepstrum(FrameSeries):
    def to_spectrum(self) -> "AmplitudeSpectrum":
        return AmplitudeSpectrum(
            np.exp(np.real(np.fft.fft(self.data))),
            self.frame_length,
            self.frame_shift,
            dB=False,
            power=False,
        )

    def lifter(self, order: int, high_quefrency=False) -> "Cepstrum":
        if high_quefrency:
            pad = np.zeros(
                (self.shape[0], self.shape[1] - 2 * order),
                dtype=self.data.dtype,
            )
            liftered = np.concatenate(
                (self.data[:, :order], pad, self.data[:, -order:]), axis=1
            )
        else:
            pad = np.zeros((self.shape[0], order), dtype=self.data.dtype)
            liftered = np.concatenate((pad, self.data[:, order:-order], pad), axis=1)

        return Cepstrum(
            liftered,
            self.frame_length,
            self.frame_shift,
            dB=self.dB,
            power=self.power,
        )

    def to_mel_cepstrum(self, bins: int, alpha: float) -> "MelCepstrum":
        # NOTE: 処理時間がかなりかかる
        return MelCepstrum(
            self.__freqt(self.data, self.shape[1] // 2, bins, alpha),
            self.frame_length,
            self.frame_shift,
            dB=self.dB,
            power=self.power,
        )

    # TODO: 高速化
    @staticmethod
    def __freqt(cepstrum: np.ndarray, n: int, bins: int, alpha: float):
        beta = 1 - alpha ** 2
        c: np.ndarray = cepstrum[:, : n + 1]
        h_mem: np.ndarray = np.zeros((cepstrum.shape[0], bins + 1))
        h: np.ndarray = np.zeros((cepstrum.shape[0], bins + 1))

        for k in range(n, -1, -1):  # [n 0]
            h[:, 0] = c[:, k] + alpha * h_mem[:, 0]
            h[:, 1] = beta * h_mem[:, 0] + alpha * h_mem[:, 1]
            for i in range(2, bins + 1):
                h[:, i] = h_mem[:, i - 1] + alpha * (h_mem[:, i] - h[:, i - 1])
            h_mem = np.copy(h)

        return np.concatenate([h, np.fliplr(h)[:, 1:-1]])


class MelCepstrum(FrameSeries):
    def to_spectrum(self, fft_point: Optional[int] = None) -> "MelSpectrum":
        if fft_point is None:
            fft_point = self.shape[1]

        to_spectrum = lambda frame: np.fft.fft(frame, n=fft_point)
        spectrum = np.exp(np.real(to_spectrum(self.data)))

        return MelSpectrum(
            spectrum,
            self.frame_length,
            self.frame_shift,
            dB=False,
            power=False,
        )
