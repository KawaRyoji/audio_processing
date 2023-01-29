from typing import Optional, Union

import librosa
import numpy as np
from scipy.signal.windows import get_window
from typing_extensions import override

from audio_processing.base import (
    FrameSeries,
    FreqDomainFrameSeries,
    TimeDomainFrameSeries,
)


class Waveform(TimeDomainFrameSeries):
    """
    時間波形のフレームの系列を扱うクラスです.
    """

    @classmethod
    def create(
        cls,
        time_series: np.ndarray,
        frame_length: int,
        frame_shift: int,
        fs: int,
        padding: bool = True,
        padding_mode: str = "reflect",
        dtype=np.float32,
    ) -> "Waveform":
        """
        時間波形から各種パラメータを使ってフレームの系列に変換し, `Waveform`インスタンスを生成します.

        Args:
            time_series (np.ndarray): 時間波形(1次元)
            frame_length (int): フレーム長
            frame_shift (int): シフト長
            fs (int): サンプリング周波数
            padding (bool, optional): 始めのフレームが分析窓の中心にするようにパディングするかどうか
            padding_mode (str, optional): パディングの手法
            dtype (_type_, optional): 保持するデータの型

        Returns:
            Waveform: 時間波形のフレームの系列
        """
        if padding:
            pad = frame_length // 2
            time_series = np.pad(time_series, pad_width=pad, mode=padding_mode)

        num_frame = 1 + (len(time_series) - frame_length) // frame_shift
        frames = []
        for i in range(num_frame):
            start, end = cls.edge_point(i, frame_length, frame_shift)
            frames.append(time_series[start:end])

        frames_np: np.ndarray = np.array(frames, dtype=dtype)

        return cls(frames_np, frame_length, frame_shift, fs)

    def to_spectrum(
        self,
        fft_point: Optional[int] = None,
        window: Union[str, np.ndarray, None] = "hann",
    ) -> "Spectrum":
        """
        時間波形のフレームの系列をスペクトル(位相情報含む)に変換します.

        Args:
            fft_point (Optional[int], optional): FFTポイント数. `None`の場合, フレーム長が使用されます
            window (Union[str, np.ndarray, None], optional): 使用する窓関数. scipyの関数名での指定か自身で作成した窓関数を指定します. `None`の場合矩形窓を使用します

        Raises:
            TypeError: 窓関数の指定が`str`または`np.ndarray`でない場合

        Returns:
            Spectrum: スペクトル
        """
        if window is None:
            window_func = np.ones(self.frame_length)
        elif type(window) is str:
            window_func: np.ndarray = get_window(window, self.frame_length)
        elif type(window) is np.ndarray:
            window_func = window
        else:
            raise TypeError("窓関数はstrもしくはnp.ndarrayでなければいけません.")

        if fft_point is None:
            fft_point = self.shape[1]

        to_spectrum = lambda frame: np.fft.fft(frame, n=fft_point)
        spectrum = to_spectrum(self.data * window_func)

        return Spectrum(
            spectrum,
            self.frame_length,
            self.frame_shift,
            fft_point,
            self.fs,
        )


class Spectrum(FreqDomainFrameSeries):
    """
    スペクトル(位相情報含む)のフレームの系列を扱うクラスです.
    """

    def __init__(
        self,
        frame_series: np.ndarray,
        frame_length: int,
        frame_shift: int,
        fft_point: int,
        fs: int,
    ) -> None:
        super().__init__(
            frame_series, frame_length, frame_shift, fft_point, fs, dB=False, dB=False
        )

    @override
    def linear_to_dB(self) -> "Spectrum":
        print("スペクトルはdB値に変換できません")
        return self

    @override
    def dB_to_linear(self) -> "Spectrum":
        print("このスペクトルはすでに線形値です")
        return self

    @override
    def linear_to_power(self) -> "Spectrum":
        print("パワースペクトルを求めたい場合、代わりに spectrum.to_amplitude().to_power() を使用してください")
        return self

    @override
    def power_to_linear(self) -> "Spectrum":
        print("このスペクトルはすでに線形値です")
        return self

    def to_amplitude(self) -> "AmplitudeSpectrum":
        """
        スペクトルを振幅スペクトルに変換します.

        Returns:
            AmplitudeSpectrum: 振幅スペクトル
        """
        return AmplitudeSpectrum(
            np.abs(self.data),
            self.frame_length,
            self.frame_shift,
            self.fft_point,
            self.fs,
            dB=False,
            power=False,
        )

    def to_phase(self, rad=True) -> "PhaseSpectrum":
        """
        スペクトルを位相スペクトルに変換します.

        Args:
            rad (bool, optional): 位相をradで扱うかどうか

        Returns:
            PhaseSpectrum: 位相スペクトル
        """
        return PhaseSpectrum(
            np.angle(self.data, deg=not rad),
            self.frame_length,
            self.frame_shift,
        )


class AmplitudeSpectrum(FreqDomainFrameSeries):
    """
    振幅スペクトルのフレームの系列を扱うクラスです.
    """

    def to_cepstrum(self) -> "Cepstrum":
        """
        振幅スペクトルをケプストラムに変換します.

        Raises:
            ValueError: この系列がパワーもしくはdB値になっている場合

        Returns:
            Cepstrum: ケプストラム
        """
        if self.power or self.dB:
            raise ValueError("この振幅スペクトルはケプストラムに変換できません.")

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
            self.fs,
        )

    def to_mel(self, bins: int) -> "MelSpectrum":
        """
        振幅スペクトルをメルスペクトルに変換します.

        Args:
            bins (int): メルのビン数

        Returns:
            MelSpectrum: メルスペクトル
        """
        filter = librosa.filters.mel(sr=self.fs, n_fft=self.fft_point, n_mels=bins)
        melspectrum = np.dot(filter, self.data[:, : self.shape[1] // 2 + 1].T).T
        return MelSpectrum(
            FreqDomainFrameSeries.to_symmetry(melspectrum),
            self.frame_length,
            self.frame_shift,
            self.fft_point,
            self.fs,
            dB=self.dB,
            power=self.power,
        )


class PhaseSpectrum(FrameSeries):
    """
    位相スペクトルのフレームの系列を扱うクラスです.
    """

    pass


class MelSpectrum(FreqDomainFrameSeries):
    """
    メルスペクトルのフレームの系列を扱うクラスです.
    """

    def to_cepstrum(self) -> "MelCepstrum":
        """
        メルスペクトルをメルケプストラムに変換します.

        Raises:
            ValueError: この値がパワーまたはdB値の場合

        Returns:
            MelCepstrum: メルケプストラム
        """
        if self.power or self.dB:
            raise ValueError("このメルスペクトルはメルケプストラムに変換できません.")

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
            self.fs,
        )


class Cepstrum(TimeDomainFrameSeries):
    """
    ケプストラムのフレームの系列を扱うクラスです.
    """

    def to_spectrum(self, fft_point: Optional[int] = None) -> "AmplitudeSpectrum":
        """
        ケプストラムから振幅スペクトルに変換します.

        Args:
            fft_point (Optional[int], optional): FFTポイント数

        Returns:
            AmplitudeSpectrum: 振幅スペクトル
        """
        if fft_point is None:
            fft_point = self.data.shape[1]

        return AmplitudeSpectrum(
            np.exp(np.real(np.fft.fft(self.data, n=fft_point))),
            self.frame_length,
            self.frame_shift,
            fft_point,
            self.fs,
            dB=False,
            power=False,
        )

    def lifter(self, order: int, high_quefrency=False) -> "Cepstrum":
        """
        リフタリング処理を行ったケプストラムを返します.

        Args:
            order (int): リフタリング次数
            high_quefrency (bool, optional): `True`の場合高ケフレンシー領域を残し, `False`の場合は低ケフレンシー領域を残します.

        Returns:
            Cepstrum: リフタリングされたケプストラム
        """
        if high_quefrency:
            pad = np.zeros(
                (self.shape[0], self.shape[1] - 2 * order),
                dtype=self.data.dtype,
            )
            liftered = np.concatenate(
                (self.data[:, :order], pad, self.data[:, -order:]), axis=1
            )
        else:
            liftered = np.pad(
                self.data[:, order:-order],
                ((0, 0), (order, order)),
                "constant",
                constant_values=0,
            )

        return Cepstrum(
            liftered,
            self.frame_length,
            self.frame_shift,
            self.fs,
        )

    def to_mel_cepstrum(self, bins: int, alpha: float) -> "MelCepstrum":
        """
        ケプストラムからメルケプストラムに変換します.
        この処理は処理時間がかかるので注意してください.

        Args:
            bins (int): メルのビン数
            alpha (float): 伸縮率 (alpha > 0)

        Returns:
            MelCepstrum: メルケプストラム
        """
        melcepstrum = self._freqt(self.data, self.shape[1] // 2 + 1, bins, alpha)

        # NOTE: 処理時間がかなりかかる
        return MelCepstrum(
            FreqDomainFrameSeries.to_symmetry(melcepstrum),
            self.frame_length,
            self.frame_shift,
            self.fs,
        )

    # TODO: 高速化
    @staticmethod
    def _freqt(cepstrum: np.ndarray, n: int, bins: int, alpha: float) -> np.ndarray:
        """
        ケプストラムを伸縮するアルゴリズム.

        Args:
            cepstrum (np.ndarray): 変換元のケプストラム
            n (int): 使用するケプストラムの次数
            bins (int): メルのビン数
            alpha (float): 伸縮率

        Returns:
            np.ndarray: 伸縮されたケプストラム
        """
        beta = 1 - alpha**2
        c: np.ndarray = cepstrum[:, :n]
        h_mem: np.ndarray = np.zeros((cepstrum.shape[0], bins + 1))
        h: np.ndarray = np.zeros((cepstrum.shape[0], bins + 1))

        for k in range(n, -1, -1):  # [n 0]
            h[:, 0] = c[:, k] + alpha * h_mem[:, 0]
            h[:, 1] = beta * h_mem[:, 0] + alpha * h_mem[:, 1]
            for i in range(2, bins + 1):
                h[:, i] = h_mem[:, i - 1] + alpha * (h_mem[:, i] - h[:, i - 1])
            h_mem = np.copy(h)

        return h


class MelCepstrum(TimeDomainFrameSeries):
    """
    メルケプストラムのフレームの系列を扱うクラスです.
    """

    def to_spectrum(self, fft_point: Optional[int] = None) -> "MelSpectrum":
        """
        メルケプストラムをメルスペクトルに変換します.

        Args:
            fft_point (Optional[int], optional): FFTポイント数. `None`の場合フレーム長を使用します

        Returns:
            MelSpectrum: メルスペクトル
        """
        if fft_point is None:
            fft_point = self.shape[1]

        to_spectrum = lambda frame: np.fft.fft(frame, n=fft_point)
        spectrum = np.exp(np.real(to_spectrum(self.data)))

        return MelSpectrum(
            spectrum,
            self.frame_length,
            self.frame_shift,
            fft_point,
            self.fs,
            dB=False,
            power=False,
        )

    def to_cepstrum(self, alpha: int) -> "Cepstrum":
        """
        メルケプストラムをケプストラムに変換します.

        Args:
            alpha (int): メルケプストラムを生成した伸縮率 (alpha > 0)

        Returns:
            Cepstrum: ケプストラム
        """
        cepstrum = Cepstrum._freqt(
            self.data, self.shape[1] // 2 + 1, self.shape[1] // 2 + 1, -alpha
        )
