from typing import Any, Callable, Optional, Union

import librosa
import numpy as np
from base import FrameSeries, FreqDomainFrameSeries, TimeDomainFrameSeries
from scipy.signal.windows import get_window
from typing_extensions import override


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
        dtype: np.dtype = np.float32,
    ) -> "Waveform":
        """
        時間波形から各種パラメータを使ってフレームの系列に変換し, `Waveform`インスタンスを生成します.

        Args:
            time_series (np.ndarray): 時間波形(1次元)
            frame_length (int): フレーム長
            frame_shift (int): シフト長
            fs (int): サンプリング周波数
            padding (bool, optional): フレームが分析窓の中心にするようにパディングするかどうか
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
        spectrum = to_spectrum(self.frame_series * window_func)

        return Spectrum(
            spectrum,
            self.frame_length,
            self.frame_shift,
            fft_point,
            self.fs,
        )

    # 以下継承したメソッド

    def apply(
        self, func: Callable[[np.ndarray], np.ndarray], axis: int = 1
    ) -> "Waveform":
        return super().apply(func, axis)

    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
        frame_length: Optional[int] = None,
        frame_shift: Optional[int] = None,
        fs: Optional[int] = None,
    ) -> "Waveform":
        """
        引数の値を使って自身のインスタンスをコピーします.

        Args:
            frame_series (Optional[np.ndarray], optional): フレーム単位の系列
            frame_length (Optional[int], optional): フレーム長
            frame_shift (Optional[int], optional): フレームシフト
            fs (Optional[int], optional): サンプリング周波数

        Returns:
            Waveform: コピーしたインスタンス
        """
        frame_series = self.frame_series if frame_series is None else frame_series
        frame_length = self.frame_length if frame_length is None else frame_length
        frame_shift = self.frame_shift if frame_shift is None else frame_shift
        fs = self.fs if fs is None else fs

        return Waveform(frame_series, frame_length, frame_shift, fs)

    def save(
        self, path: str, compress: bool = False, overwrite: bool = False
    ) -> "Waveform":
        return super().save(path, compress=compress, overwrite=overwrite)

    @classmethod
    def from_npz(cls, path: str) -> "Waveform":
        return super().from_npz(path)

    def plot(
        self, show=True, save_fig_path: Optional[str] = None, color_map: str = "magma"
    ) -> "Waveform":
        return super().plot(
            show=show,
            save_fig_path=save_fig_path,
            color_map=color_map,
        )

    def dump(self) -> "Waveform":
        return super().dump()

    def __add__(self, other: Any) -> "Waveform":
        return super().__add__(other)

    def __sub__(self, other: Any) -> "Waveform":
        return super().__sub__(other)

    def __mul__(self, other: Any) -> "Waveform":
        return super().__mul__(other)

    def __truediv__(self, other: Any) -> "Waveform":
        return super().__truediv__(other)


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
            frame_series,
            frame_length,
            frame_shift,
            fft_point,
            fs,
            dB=False,
            power=False,
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
            np.abs(self.frame_series),
            self.frame_length,
            self.frame_shift,
            self.fft_point,
            self.fs,
            dB=False,
            power=False,
        )

    def to_phase(self) -> "PhaseSpectrum":
        """
        スペクトルを位相スペクトルに変換します.

        Returns:
            PhaseSpectrum: 位相スペクトル
        """
        return PhaseSpectrum(
            np.angle(self.frame_series),
            self.frame_length,
            self.frame_shift,
        )

    def to_waveform(self) -> "Waveform":
        return Waveform(
            np.real(np.fft.ifft(self.frame_series)),
            self.frame_length,
            self.frame_shift,
            self.fs,
        )

    @classmethod
    def restore(
        cls, amplitude: "AmplitudeSpectrum", phase: "PhaseSpectrum"
    ) -> "Spectrum":
        if amplitude.dB or amplitude.power:
            raise ValueError("振幅スペクトルを線形値にしてください.")

        if amplitude.shape != phase.shape:
            raise ValueError(
                "振幅スペクトルと位相スペクトルの次元が異なっています. amplitude:{} phase:{}".format(
                    amplitude.shape, phase.shape
                )
            )

        if amplitude.frame_length != phase.frame_length:
            raise ValueError(
                "振幅スペクトルと位相スペクトルのフレーム長が異なっています. amplitude:{} phase:{}".format(
                    amplitude.frame_length, phase.frame_length
                )
            )

        if amplitude.frame_shift != phase.frame_shift:
            raise ValueError(
                "振幅スペクトルと位相スペクトルのフレームシフトが異なっています. amplitude:{} phase:{}".format(
                    amplitude.frame_shift, phase.frame_shift
                )
            )

        return Spectrum(
            np.multiply(amplitude, np.exp(1j * phase)),
            amplitude.frame_length,
            amplitude.frame_shift,
            amplitude.fft_point,
            amplitude.fs,
        )

    # 以下継承したメソッド

    def apply(
        self, func: Callable[[np.ndarray], np.ndarray], axis: int = 1
    ) -> "Spectrum":
        return super().apply(func, axis)

    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
        frame_length: Optional[int] = None,
        frame_shift: Optional[int] = None,
        fft_point: Optional[int] = None,
        fs: Optional[int] = None,
        dB: Optional[bool] = None,
        power: Optional[bool] = None,
    ) -> "Spectrum":
        """
        引数の値を使って自身のインスタンスをコピーします.

        Args:
            frame_series (Optional[np.ndarray], optional): フレーム単位の系列
            frame_length (Optional[int], optional): フレーム長
            frame_shift (Optional[int], optional): フレームシフト
            fft_point (Optional[int], optional): FFTポイント数
            fs (Optional[int], optional): サンプリング周波数
            dB (Optional[bool], optional): dB値であるか
            power (Optional[bool], optional): パワー値であるか

        Returns:
            Spectrum: コピーしたインスタンス
        """
        frame_series = self.frame_series if frame_series is None else frame_series
        frame_length = self.frame_length if frame_length is None else frame_length
        frame_shift = self.frame_shift if frame_shift is None else frame_shift
        fft_point = self.fft_point if fft_point is None else fft_point
        fs = self.fs if fs is None else fs
        dB = self.dB if dB is None else dB
        power = self.power if power is None else power

        return Spectrum(
            frame_series, frame_length, frame_shift, fft_point, fs, dB=dB, power=power
        )

    def save(
        self, path: str, compress: bool = False, overwrite: bool = False
    ) -> "Spectrum":
        return super().save(path, compress=compress, overwrite=overwrite)

    @classmethod
    def from_npz(cls, path: str) -> "Spectrum":
        return super().from_npz(path)

    def plot(
        self,
        up_to_nyquist=True,
        show=True,
        save_fig_path: Optional[str] = None,
        color_map: str = "magma",
    ) -> "Spectrum":
        return super().plot(
            up_to_nyquist=up_to_nyquist,
            show=show,
            save_fig_path=save_fig_path,
            color_map=color_map,
        )

    def dump(self) -> "Spectrum":
        return super().dump()

    def __add__(self, other: Any) -> "Spectrum":
        return super().__add__(other)

    def __sub__(self, other: Any) -> "Spectrum":
        return super().__sub__(other)

    def __mul__(self, other: Any) -> "Spectrum":
        return super().__mul__(other)

    def __truediv__(self, other: Any) -> "Spectrum":
        return super().__truediv__(other)


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
                        self.frame_series,
                        out=np.zeros_like(self.frame_series),
                        where=self.frame_series != 0,
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
        melspectrum = np.dot(filter, self.frame_series[:, : self.shape[1] // 2 + 1].T).T
        return MelSpectrum(
            FreqDomainFrameSeries.to_symmetry(melspectrum),
            self.frame_length,
            self.frame_shift,
            self.fft_point,
            self.fs,
            dB=self.dB,
            power=self.power,
        )

    # 以下継承したメソッド

    def linear_to_dB(self) -> "AmplitudeSpectrum":
        return super().linear_to_dB()

    def dB_to_linear(self) -> "AmplitudeSpectrum":
        return super().dB_to_linear()

    def linear_to_power(self) -> "AmplitudeSpectrum":
        return super().linear_to_power()

    def power_to_linear(self) -> "AmplitudeSpectrum":
        return super().power_to_linear()

    def apply(
        self, func: Callable[[np.ndarray], np.ndarray], axis: int = 1
    ) -> "AmplitudeSpectrum":
        return super().apply(func, axis)

    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
        frame_length: Optional[int] = None,
        frame_shift: Optional[int] = None,
        fft_point: Optional[int] = None,
        fs: Optional[int] = None,
        dB: Optional[bool] = None,
        power: Optional[bool] = None,
    ) -> "AmplitudeSpectrum":
        """
        引数の値を使って自身のインスタンスをコピーします.

        Args:
            frame_series (Optional[np.ndarray], optional): フレーム単位の系列
            frame_length (Optional[int], optional): フレーム長
            frame_shift (Optional[int], optional): フレームシフト
            fft_point (Optional[int], optional): FFTポイント数
            fs (Optional[int], optional): サンプリング周波数
            dB (Optional[bool], optional): dB値であるか
            power (Optional[bool], optional): パワー値であるか

        Returns:
            AmplitudeSpectrum: コピーしたインスタンス
        """
        frame_series = self.frame_series if frame_series is None else frame_series
        frame_length = self.frame_length if frame_length is None else frame_length
        frame_shift = self.frame_shift if frame_shift is None else frame_shift
        fft_point = self.fft_point if fft_point is None else fft_point
        fs = self.fs if fs is None else fs
        dB = self.dB if dB is None else dB
        power = self.power if power is None else power

        return AmplitudeSpectrum(
            frame_series, frame_length, frame_shift, fft_point, fs, dB=dB, power=power
        )

    def save(
        self, path: str, compress: bool = False, overwrite: bool = False
    ) -> "AmplitudeSpectrum":
        return super().save(path, compress=compress, overwrite=overwrite)

    @classmethod
    def from_npz(cls, path: str) -> "AmplitudeSpectrum":
        return super().from_npz(path)

    def plot(
        self,
        up_to_nyquist: bool = True,
        show: bool = True,
        save_fig_path: Optional[str] = None,
        color_map: str = "magma",
    ) -> "AmplitudeSpectrum":
        return super().plot(
            up_to_nyquist=up_to_nyquist,
            show=show,
            save_fig_path=save_fig_path,
            color_map=color_map,
        )

    def dump(self) -> "AmplitudeSpectrum":
        return super().dump()

    def __add__(self, other: Any) -> "AmplitudeSpectrum":
        return super().__add__(other)

    def __sub__(self, other: Any) -> "AmplitudeSpectrum":
        return super().__sub__(other)

    def __mul__(self, other: Any) -> "AmplitudeSpectrum":
        return super().__mul__(other)

    def __truediv__(self, other: Any) -> "AmplitudeSpectrum":
        return super().__truediv__(other)


class PhaseSpectrum(FrameSeries):
    """
    位相スペクトルのフレームの系列を扱うクラスです.
    """

    # 以下継承したメソッド

    def apply(
        self, func: Callable[[np.ndarray], np.ndarray], axis: int = 1
    ) -> "PhaseSpectrum":
        return super().apply(func, axis)

    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
        frame_length: Optional[int] = None,
        frame_shift: Optional[int] = None,
    ) -> "PhaseSpectrum":
        """
        引数の値を使って自身のインスタンスをコピーします.

        Args:
            frame_series (Optional[np.ndarray], optional): フレーム単位の系列
            frame_length (Optional[int], optional): フレーム長
            frame_shift (Optional[int], optional): フレームシフト

        Returns:
            PhaseSpectrum: コピーしたインスタンス
        """
        frame_series = self.frame_series if frame_series is None else frame_series
        frame_length = self.frame_length if frame_length is None else frame_length
        frame_shift = self.frame_shift if frame_shift is None else frame_shift

        return PhaseSpectrum(frame_series, frame_length, frame_shift)

    def save(
        self, path: str, compress: bool = False, overwrite: bool = False
    ) -> "PhaseSpectrum":
        return super().save(path, compress=compress, overwrite=overwrite)

    @classmethod
    def from_npz(cls, path: str) -> "PhaseSpectrum":
        return super().from_npz(path)

    def plot(
        self,
        show: bool = True,
        save_fig_path: Optional[str] = None,
        color_map: str = "magma",
    ) -> "PhaseSpectrum":
        return super().plot(
            show=show,
            save_fig_path=save_fig_path,
            color_map=color_map,
        )

    def dump(self) -> "PhaseSpectrum":
        return super().dump()

    def __add__(self, other: Any) -> "PhaseSpectrum":
        return super().__add__(other)

    def __sub__(self, other: Any) -> "PhaseSpectrum":
        return super().__sub__(other)

    def __mul__(self, other: Any) -> "PhaseSpectrum":
        return super().__mul__(other)

    def __truediv__(self, other: Any) -> "PhaseSpectrum":
        return super().__truediv__(other)


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
                        self.frame_series,
                        out=np.zeros_like(self.frame_series),
                        where=self.frame_series != 0,
                    )
                )
            ),
            self.frame_length,
            self.frame_shift,
            self.fs,
        )

    # 以下継承したメソッド

    def linear_to_dB(self) -> "MelSpectrum":
        return super().linear_to_dB()

    def dB_to_linear(self) -> "MelSpectrum":
        return super().dB_to_linear()

    def linear_to_power(self) -> "MelSpectrum":
        return super().linear_to_power()

    def power_to_linear(self) -> "MelSpectrum":
        return super().power_to_linear()

    def apply(
        self, func: Callable[[np.ndarray], np.ndarray], axis: int = 1
    ) -> "MelSpectrum":
        return super().apply(func, axis)

    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
        frame_length: Optional[int] = None,
        frame_shift: Optional[int] = None,
        fft_point: Optional[int] = None,
        fs: Optional[int] = None,
        dB: Optional[bool] = None,
        power: Optional[bool] = None,
    ) -> "MelSpectrum":
        """
        引数の値を使って自身のインスタンスをコピーします.

        Args:
            frame_series (Optional[np.ndarray], optional): フレーム単位の系列
            frame_length (Optional[int], optional): フレーム長
            frame_shift (Optional[int], optional): フレームシフト
            fft_point (Optional[int], optional): FFTポイント数
            fs (Optional[int], optional): サンプリング周波数
            dB (Optional[bool], optional): dB値であるか
            power (Optional[bool], optional): パワー値であるか

        Returns:
            MelSpectrum: コピーしたインスタンス
        """
        frame_series = self.frame_series if frame_series is None else frame_series
        frame_length = self.frame_length if frame_length is None else frame_length
        frame_shift = self.frame_shift if frame_shift is None else frame_shift
        fft_point = self.fft_point if fft_point is None else fft_point
        fs = self.fs if fs is None else fs
        dB = self.dB if dB is None else dB
        power = self.power if power is None else power

        return MelSpectrum(
            frame_series, frame_length, frame_shift, fft_point, fs, dB=dB, power=power
        )

    def save(
        self, path: str, compress: bool = False, overwrite: bool = False
    ) -> "MelSpectrum":
        return super().save(path, compress=compress, overwrite=overwrite)

    @classmethod
    def from_npz(cls, path: str) -> "MelSpectrum":
        return super().from_npz(path)

    def plot(
        self,
        up_to_nyquist: bool = True,
        show: bool = True,
        save_fig_path: Optional[str] = None,
        color_map: str = "magma",
    ) -> "MelSpectrum":
        return super().plot(
            up_to_nyquist=up_to_nyquist,
            show=show,
            save_fig_path=save_fig_path,
            color_map=color_map,
        )

    def dump(self) -> "MelSpectrum":
        return super().dump()

    def __add__(self, other: Any) -> "MelSpectrum":
        return super().__add__(other)

    def __sub__(self, other: Any) -> "MelSpectrum":
        return super().__sub__(other)

    def __mul__(self, other: Any) -> "MelSpectrum":
        return super().__mul__(other)

    def __truediv__(self, other: Any) -> "MelSpectrum":
        return super().__truediv__(other)


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
            fft_point = self.frame_series.shape[1]

        return AmplitudeSpectrum(
            np.exp(np.real(np.fft.fft(self.frame_series, n=fft_point))),
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
                dtype=self.frame_series.dtype,
            )
            liftered = np.concatenate(
                (self.frame_series[:, :order], pad, self.frame_series[:, -order:]),
                axis=1,
            )
        else:
            liftered = np.pad(
                self.frame_series[:, order:-order],
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
        melcepstrum = self._freqt(
            self.frame_series, self.shape[1] // 2 + 1, bins, alpha
        )

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

    # 以下継承したメソッド

    def apply(
        self, func: Callable[[np.ndarray], np.ndarray], axis: int = 1
    ) -> "Cepstrum":
        return super().apply(func, axis)

    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
        frame_length: Optional[int] = None,
        frame_shift: Optional[int] = None,
        fs: Optional[int] = None,
    ) -> "Cepstrum":
        """
        引数の値を使って自身のインスタンスをコピーします.

        Args:
            frame_series (Optional[np.ndarray], optional): フレーム単位の系列
            frame_length (Optional[int], optional): フレーム長
            frame_shift (Optional[int], optional): フレームシフト
            fs (Optional[int], optional): サンプリング周波数

        Returns:
            Cepstrum: コピーしたインスタンス
        """
        frame_series = self.frame_series if frame_series is None else frame_series
        frame_length = self.frame_length if frame_length is None else frame_length
        frame_shift = self.frame_shift if frame_shift is None else frame_shift
        fs = self.fs if fs is None else fs

        return Cepstrum(frame_series, frame_length, frame_shift, fs)

    def save(
        self, path: str, compress: bool = False, overwrite: bool = False
    ) -> "Cepstrum":
        return super().save(path, compress=compress, overwrite=overwrite)

    @classmethod
    def from_npz(cls, path: str) -> "Cepstrum":
        return super().from_npz(path)

    def plot(
        self,
        show: bool = True,
        save_fig_path: Optional[str] = None,
        color_map: str = "magma",
    ) -> "Cepstrum":
        return super().plot(
            show=show,
            save_fig_path=save_fig_path,
            color_map=color_map,
        )

    def dump(self) -> "Cepstrum":
        return super().dump()

    def __add__(self, other: Any) -> "Cepstrum":
        return super().__add__(other)

    def __sub__(self, other: Any) -> "Cepstrum":
        return super().__sub__(other)

    def __mul__(self, other: Any) -> "Cepstrum":
        return super().__mul__(other)

    def __truediv__(self, other: Any) -> "Cepstrum":
        return super().__truediv__(other)


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
        spectrum = np.exp(np.real(to_spectrum(self.frame_series)))

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
            self.frame_series, self.shape[1] // 2 + 1, self.shape[1] // 2 + 1, -alpha
        )

        return Cepstrum(
            cepstrum,
            self.frame_length,
            self.frame_shift,
            self.fs,
        )

    # 以下継承したメソッド

    def apply(
        self, func: Callable[[np.ndarray], np.ndarray], axis: int = 1
    ) -> "MelCepstrum":
        return super().apply(func, axis)

    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
        frame_length: Optional[int] = None,
        frame_shift: Optional[int] = None,
        fs: Optional[int] = None,
    ) -> "MelCepstrum":
        """
        引数の値を使って自身のインスタンスをコピーします.

        Args:
            frame_series (Optional[np.ndarray], optional): フレーム単位の系列
            frame_length (Optional[int], optional): フレーム長
            frame_shift (Optional[int], optional): フレームシフト
            fs (Optional[int], optional): サンプリング周波数

        Returns:
            MelCepstrum: コピーしたインスタンス
        """
        frame_series = self.frame_series if frame_series is None else frame_series
        frame_length = self.frame_length if frame_length is None else frame_length
        frame_shift = self.frame_shift if frame_shift is None else frame_shift
        fs = self.fs if fs is None else fs

        return MelCepstrum(frame_series, frame_length, frame_shift, fs)

    def save(
        self, path: str, compress: bool = False, overwrite: bool = False
    ) -> "MelCepstrum":
        return super().save(path, compress=compress, overwrite=overwrite)

    @classmethod
    def from_npz(cls, path: str) -> "MelCepstrum":
        return super().from_npz(path)

    def plot(
        self, show=True, save_fig_path: Optional[str] = None, color_map: str = "magma"
    ) -> "MelCepstrum":
        return super().plot(
            show=show,
            save_fig_path=save_fig_path,
            color_map=color_map,
        )

    def dump(self) -> "MelCepstrum":
        return super().dump()

    def __add__(self, other: Any) -> "MelCepstrum":
        return super().__add__(other)

    def __sub__(self, other: Any) -> "MelCepstrum":
        return super().__sub__(other)

    def __mul__(self, other: Any) -> "MelCepstrum":
        return super().__mul__(other)

    def __truediv__(self, other: Any) -> "MelCepstrum":
        return super().__truediv__(other)
