from typing import Callable, Optional, Union

import numpy as np
from scipy.signal.windows import get_window
from features import Spectrum, Waveform


def to_spectrum(
    fft_point: Optional[int] = None, window: Union[str, np.ndarray, None] = "hann"
) -> Callable[[Waveform], Spectrum]:
    def inner(waveform: Waveform) -> Spectrum:
        if window is None:
            window_func = np.ones(waveform.frame_length)
        elif type(window) is str:
            window_func: np.ndarray = get_window(window, waveform.frame_length)
        elif type(window) is np.ndarray:
            window_func = window
        else:
            raise TypeError("窓関数はstrもしくはnp.ndarrayでなければいけません.")

        if fft_point is None:
            fft_point = waveform.shape[1]

        to_spectrum = lambda frame: np.fft.fft(frame, n=fft_point)
        spectrum = to_spectrum(waveform.data * window_func)

        return Spectrum(
            spectrum,
            waveform.frame_length,
            waveform.frame_shift,
            fft_point,
            waveform.fs,
        )

    return inner

a = to_spectrum(fft_point=256)
