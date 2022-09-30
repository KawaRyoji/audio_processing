from typing import Tuple, Union
import numpy as np
from scipy.signal.windows import get_window


class FrameSeries:
    def __init__(
        self,
        frame_series: np.ndarray,
        frame_length: int,
        frame_shift: int,
    ) -> None:
        self.__frame_series = frame_series
        self.__frame_length = frame_length
        self.__frame_shift = frame_shift

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

    def edge_point(self, index: int) -> Tuple[int, int]:
        start = index * self.frame_shift
        end = start + self.frame_len
        return start, end

    def to_dB(self, power=False) -> "FrameSeries":
        if power:
            to_dB = lambda frame: 10 * np.log10(
                frame, out=np.zeros_like(frame), where=frame != 0
            )
        else:
            to_dB = lambda frame: 20 * np.log10(
                frame, out=np.zeros_like(frame), where=frame != 0
            )
        return FrameSeries(to_dB(self.data))

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
        dtype=np.float32,
    ) -> "FrameSeries":
        if padding:
            pad = frame_length // 2
            time_series = np.pad(time_series, pad, mode="reflect")

        num_frame = 1 + (len(time_series) - frame_length) // frame_shift
        frames = []
        for i in range(num_frame):
            start, end = cls.edge_point(i, frame_length, frame_shift)
            frames.append(time_series[start:end])

        frames_np: np.ndarray = np.array(frames, dtype=dtype)

        return cls(frames_np, frame_length, frame_shift)

    def to_spectrum(
        self,
        fft_point: int,
        window: Union[str, np.ndarray],
    ) -> "Spectrum":
        if type(window) is str:
            window_func: np.ndarray = get_window(window, self.frame_length)
        elif type(window) is np.ndarray:
            window_func: np.ndarray = window
        else:
            raise TypeError("Type of window is str or np.ndarray")

        to_spectrum = lambda frame: np.fft.fft(frame, n=fft_point)
        spectrum = to_spectrum(self.data * window_func)

        return Spectrum(spectrum, self.frame_length, self.frame_shift, power=False)


class Spectrum(FrameSeries):
    def __init__(
        self,
        frame_series: np.ndarray,
        frame_length: int,
        frame_shift: int,
        power: bool = False,
    ) -> None:
        super().__init__(frame_series, frame_length, frame_shift)
        self.power = power

    def to_amplitude(self) -> "Spectrum":
        return Spectrum(
            np.abs(self.data), self.frame_length, self.frame_shift, power=self.power
        )
