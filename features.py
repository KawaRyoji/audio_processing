from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

import librosa
import numpy as np
from scipy.signal.windows import get_window
from typing_extensions import Self, override

from audio_processing.base import (
    FrameSeries,
    FreqDomainFrameSeries,
    TimeDomainFrameSeries,
)

if TYPE_CHECKING:
    from audio_processing.fileio import WavFile


class WaveformFrameSeries(TimeDomainFrameSeries):
    """
    時間波形のフレームの系列を扱うクラスです.
    """

    @classmethod
    def from_waveform(
        cls,
        waveform: np.ndarray,
        frame_length: int,
        frame_shift: int,
        fs: int,
        padding: bool = True,
        padding_mode: str = "reflect",
        dtype: np.dtype = np.float32,
    ) -> Self:
        """
        時間波形から各種パラメータを使ってフレームの系列に変換し, `Waveform`インスタンスを生成します.

        Args:
            waveform (np.ndarray): 時間波形(1次元)
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
            waveform = np.pad(waveform, pad_width=pad, mode=padding_mode)

        num_frame = 1 + (len(waveform) - frame_length) // frame_shift
        frames = np.array(
            [
                waveform[slice(*cls.edge_point(i, frame_length, frame_shift))]
                for i in range(num_frame)
            ],
            dtype=dtype,
        )

        return cls(frames, frame_length, frame_shift, fs)

    def to_spectrum(
        self,
        fft_point: Optional[int] = None,
        window: Union[str, np.ndarray, None] = "hann",
    ) -> Spectrum:
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
            window_func = get_window(window, self.frame_length)
        elif type(window) is np.ndarray:
            window_func = window
        else:
            raise TypeError("窓関数はstrもしくはnp.ndarrayでなければいけません.")

        fft_point = self.shape[1] if fft_point is None else fft_point

        return Spectrum(
            np.fft.fft(self.frame_series * window_func),
            self.frame_length,
            self.frame_shift,
            fft_point,
            self.fs,
        )

    def to_wav_file(self) -> WavFile:
        """
        フレームの系列の時間波形を1次元の時間波形に再構成し, WavFileに変換します.

        Returns:
            WavFile: 再構成した時間波形のWavFile
        """
        from audio_processing.fileio import WavFile  # 循環参照対策

        # TODO: 高速化
        # x = [[0,1,2],[3,4,5],[6,7,8]], シフト長1の場合
        # 1回目
        # [0,1,2,0] +   :シフト長分右にゼロを埋める
        # [0,3,4,5] =   :上([0,1,2])の要素数(3) + シフト長(1) - 自身の要素数(3)分左にゼロを埋める
        # [0,4,6,5]
        # 2回目
        # [0,4,6,5,0] + :シフト長分右にゼロを埋める
        # [0,0,6,7,8] = :上([0,4,6,5])の要素数(4) + シフト長(1) - 自身の要素数(3)分左にゼロを埋める
        # [0,4,12,12,8] :解
        # istftでやっていることと同じ
        return WavFile(
            data=self.reduce(
                lambda x, y: np.pad(x, (0, self.frame_shift))
                + np.pad(y, (len(x) + self.frame_shift - len(y), 0)),
            ),
            fs=self.fs,
            dtype=self.frame_series.dtype,
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
            frame_series,
            frame_length,
            frame_shift,
            fft_point,
            fs,
            dB=False,
            power=False,
        )

    @override
    def linear_to_dB(self) -> Self:
        print("スペクトルはdB値に変換できません")
        return self

    @override
    def dB_to_linear(self) -> Self:
        print("このスペクトルはすでに線形値です")
        return self

    @override
    def linear_to_power(self) -> Self:
        print("パワースペクトルを求めたい場合、代わりに spectrum.to_amplitude().to_power() を使用してください")
        return self

    @override
    def power_to_linear(self) -> Self:
        print("このスペクトルはすでに線形値です")
        return self

    def to_amplitude(self) -> AmplitudeSpectrum:
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

    def to_phase(self) -> PhaseSpectrum:
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

    def to_waveform(self) -> WaveformFrameSeries:
        """
        スペクトルを時間波形に変換します.

        Returns:
            Waveform: 変換した時間波形
        """
        return WaveformFrameSeries(
            np.fft.ifft(self.frame_series).real,
            self.frame_length,
            self.frame_shift,
            self.fs,
        )

    def decompose(self) -> Tuple[AmplitudeSpectrum, PhaseSpectrum]:
        """
        スペクトルを振幅スペクトルと位相スペクトルに分解します.

        Returns:
            Tuple[AmplitudeSpectrum, PhaseSpectrum]: 振幅スペクトルと位相スペクトル
        """
        return self.to_amplitude(), self.to_phase()

    @classmethod
    def restore(cls, amplitude: AmplitudeSpectrum, phase: PhaseSpectrum) -> Spectrum:
        """
        振幅スペクトルと位相スペクトルからスペクトルを復元します.

        Args:
            amplitude (AmplitudeSpectrum): 振幅スペクトル
            phase (PhaseSpectrum): 位相スペクトル

        Raises:
            ValueError: 振幅スペクトルがdB値またはパワー値となっている場合
            ValueError: 振幅スペクトルと位相スペクトルの次元が異なっている場合
            ValueError: 振幅スペクトルと位相スペクトルを生成したプロパティが異なっている場合

        Returns:
            Spectrum: 復元したスペクトル
        """
        if amplitude.dB or amplitude.power:
            raise ValueError("振幅スペクトルを線形値にしてください.")

        if amplitude.shape != phase.shape:
            raise ValueError(
                "振幅スペクトルと位相スペクトルの次元が異なっています. amplitude:{} phase:{}".format(
                    amplitude.shape, phase.shape
                )
            )

        return Spectrum(
            np.multiply(amplitude.frame_series, np.exp(1j * phase.frame_series)),
            amplitude.frame_length,
            amplitude.frame_shift,
            amplitude.fft_point,
            amplitude.fs,
        )

    # 以下継承したメソッド
    @override
    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
        frame_length: Optional[int] = None,
        frame_shift: Optional[int] = None,
        fft_point: Optional[int] = None,
        fs: Optional[int] = None,
    ) -> Self:
        """
        引数の値を使って自身のインスタンスをコピーします.

        Args:
            frame_series (Optional[np.ndarray], optional): フレーム単位の系列
            frame_length (Optional[int], optional): フレーム長
            frame_shift (Optional[int], optional): フレームシフト
            fft_point (Optional[int], optional): FFTポイント数
            fs (Optional[int], optional): サンプリング周波数

        Returns:
            Spectrum: コピーしたインスタンス
        """
        frame_series = self.frame_series if frame_series is None else frame_series
        frame_length = self.frame_length if frame_length is None else frame_length
        frame_shift = self.frame_shift if frame_shift is None else frame_shift
        fft_point = self.fft_point if fft_point is None else fft_point
        fs = self.fs if fs is None else fs

        return Spectrum(frame_series, frame_length, frame_shift, fft_point, fs)


class AmplitudeSpectrum(FreqDomainFrameSeries):
    """
    振幅スペクトルのフレームの系列を扱うクラスです.
    """

    def to_wav_file(self, iterations: int = 20) -> WavFile:
        """
        振幅スペクトルから位相復元をしてwavファイルを生成します.
        位相復元アルゴリズムはlibrosaのgiriffinlimを使用します.

        Args:
            iterations (int, optional): griffinlimアルゴリズムの反復回数

        Returns:
            WavFile: 位相復元したスペクトルから生成したwavファイル
        """
        from audio_processing.fileio import WavFile  # 循環参照対策
        
        if self.dB:
            print("振幅スペクトルを線形値に変換してください.")

        return WavFile(
            data=librosa.griffinlim(
                self.frame_series.T[: self.fft_point // 2 + 1, :],
                hop_length=self.frame_shift,
                win_length=self.frame_length,
                n_fft=self.fft_point,
                n_iter=iterations,
            ),
            fs=self.fs,
            dtype=self.dtype,
        )

    def to_cepstrum(self) -> Cepstrum:
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
            np.fft.ifft(
                np.log(
                    np.where(  # 0をepsで置換している
                        self.frame_series == 0,
                        np.full_like(
                            self.frame_series, np.finfo(self.frame_series.dtype).eps
                        ),
                        self.frame_series,
                    ),
                )
            ).real,
            self.frame_length,
            self.frame_shift,
            self.fs,
        )

    def to_mel(self, bins: int) -> MelSpectrum:
        """
        振幅スペクトルをメルスペクトルに変換します.

        Args:
            bins (int): メルのビン数

        Returns:
            MelSpectrum: メルスペクトル
        """
        filter = librosa.filters.mel(
            sr=self.fs, n_fft=self.fft_point, n_mels=bins, dtype=self.frame_series.dtype
        )
        melspectrum = np.dot(self.frame_series[:, : self.shape[1] // 2 + 1].T, filter)
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

    @classmethod
    def griffin_lim(cls, spectrum: AmplitudeSpectrum, iterations: int = 200) -> Self:
        if spectrum.dB:
            print("振幅スペクトルを線形値に変換してください.")

        return cls(
            librosa.griffinlim(
                spectrum.frame_series.T[: spectrum.fft_point // 2 + 1, :],
                hop_length=spectrum.frame_shift,
                win_length=spectrum.frame_length,
                n_fft=spectrum.fft_point,
                n_iter=iterations,
            )
        )


class MelSpectrum(FreqDomainFrameSeries):
    """
    メルスペクトルのフレームの系列を扱うクラスです.
    """

    def to_mel_cepstrum(self) -> MelCepstrum:
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
            np.fft.ifft(
                np.log(
                    np.where(  # 0をepsに置換している
                        self.frame_series == 0,
                        np.full_like(
                            self.frame_series, np.finfo(self.frame_series.dtype).eps
                        ),
                        self.frame_series,
                    )
                ),
            ).real,
            self.frame_length,
            self.frame_shift,
            self.fs,
        )


class Cepstrum(TimeDomainFrameSeries):
    """
    ケプストラムのフレームの系列を扱うクラスです.
    """

    def to_spectrum(self, fft_point: Optional[int] = None) -> AmplitudeSpectrum:
        """
        ケプストラムから振幅スペクトルに変換します.

        Args:
            fft_point (Optional[int], optional): FFTポイント数

        Returns:
            AmplitudeSpectrum: 振幅スペクトル
        """
        fft_point = self.shape[1] if fft_point is None else fft_point

        return AmplitudeSpectrum(
            np.exp(np.fft.fft(self.frame_series, n=fft_point).real),
            self.frame_length,
            self.frame_shift,
            fft_point,
            self.fs,
            dB=False,
            power=False,
        )

    def lifter(self, order: int, high_quefrency=False) -> Self:
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

    def to_mel_cepstrum(self, alpha: float, bins: Optional[int] = None) -> MelCepstrum:
        """
        ケプストラムからメルケプストラムに変換します.
        この処理は処理時間がかかるので注意してください.

        Args:
            alpha (float): 伸縮率 (alpha > 0)
            bins (Optional[int], optional): メルのビン数. 指定しない場合

        Returns:
            MelCepstrum: メルケプストラム
        """
        assert alpha > 0
        bins = self.shape[1] // 2 + 1 if bins is None else bins

        mel_cepstrum = self._freqt(
            self.frame_series, self.shape[1] // 2 + 1, bins, alpha
        )

        # NOTE: 処理時間がかなりかかる
        return MelCepstrum(
            FreqDomainFrameSeries.to_symmetry(mel_cepstrum),
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
        h_mem: np.ndarray = np.zeros((cepstrum.shape[0], bins), dtype=cepstrum.dtype)
        h: np.ndarray = np.zeros((cepstrum.shape[0], bins), dtype=cepstrum.dtype)

        for k in range(n - 1, -1, -1):  # [n-1 0]
            h[:, 0] = c[:, k] + alpha * h_mem[:, 0]
            h[:, 1] = beta * h_mem[:, 0] + alpha * h_mem[:, 1]
            for i in range(2, bins):
                h[:, i] = h_mem[:, i - 1] + alpha * (h_mem[:, i] - h[:, i - 1])
            h_mem = np.copy(h)

        return h


class MelCepstrum(TimeDomainFrameSeries):
    """
    メルケプストラムのフレームの系列を扱うクラスです.
    """

    def to_spectrum(self, fft_point: Optional[int] = None) -> MelSpectrum:
        """
        メルケプストラムをメルスペクトルに変換します.

        Args:
            fft_point (Optional[int], optional): FFTポイント数. `None`の場合フレーム長を使用します

        Returns:
            MelSpectrum: メルスペクトル
        """
        fft_point = self.shape[1] if fft_point is None else fft_point

        return MelSpectrum(
            np.exp(np.fft.fft(self.frame_series, n=fft_point).real),
            self.frame_length,
            self.frame_shift,
            fft_point,
            self.fs,
            dB=False,
            power=False,
        )

    def to_cepstrum(self, alpha: int) -> Cepstrum:
        """
        メルケプストラムをケプストラムに変換します.

        Args:
            alpha (int): メルケプストラムを生成した伸縮率 (alpha > 0)

        Returns:
            Cepstrum: ケプストラム
        """
        assert alpha > 0
        cepstrum = Cepstrum._freqt(
            self.frame_series, self.shape[1] // 2 + 1, self.shape[1] // 2 + 1, -alpha
        )

        return Cepstrum(
            FreqDomainFrameSeries.to_symmetry(cepstrum),
            self.frame_length,
            self.frame_shift,
            self.fs,
        )
