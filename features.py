from typing import Optional, Tuple, Union

import librosa
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal.windows import get_window


class FrameSeries:
    """
    フレーム単位の情報を扱うクラスです.
    """

    def __init__(
        self,
        frame_series: np.ndarray,
        frame_length: int,
        frame_shift: int,
        dB: Optional[bool] = None,
        power: Optional[bool] = None,
    ) -> None:
        """
        Args:
            frame_series (np.ndarray): フレーム単位の系列(2次元を想定)
            frame_length (int): フレーム長
            frame_shift (int): シフト長
            dB (Optional[bool], optional): この系列がdB値であるか. `None`の場合dB値に変換できないことを表します.
            power (Optional[bool], optional): この系列がパワーであるか. `None`の場合パワーに変換できないことを表します.
        """
        self.__frame_series = frame_series
        self.__frame_length = frame_length
        self.__frame_shift = frame_shift
        self.__dB = dB
        self.__power = power

    @property
    def frame_length(self) -> int:
        """
        フレーム長を返します.

        Returns:
            int: フレーム長
        """
        return self.__frame_length

    @property
    def frame_shift(self) -> int:
        """
        シフト長を返します.

        Returns:
            int: シフト長
        """
        return self.__frame_shift

    @property
    def shape(self) -> Tuple:
        """
        フレームの系列の形状を返します.

        Returns:
            Tuple: フレームの系列の形状
        """
        return self.__frame_series.shape

    @property
    def data(self) -> np.ndarray:
        """
        フレームの系列を返します.

        Returns:
            np.ndarray: フレームの系列
        """
        return self.__frame_series

    @property
    def dB(self) -> Optional[bool]:
        """
        この系列がdB値であるかを返します.
        dB値に変換できない系列の場合Noneを返します.

        Returns:
            Optional[bool]: この系列がdB値であるかどうか. または`None`
        """
        return self.__dB

    @property
    def power(self) -> Optional[bool]:
        """
        この系列がパワーであるかを返します.
        パワーに変換できない系列の場合Noneを返します.

        Returns:
            Optional[bool]: この系列がパワーであるかどうか. または`None`
        """
        return self.__power

    @classmethod
    def edge_point(
        cls, index: int, frame_length: int, frame_shift: int
    ) -> Tuple[int, int]:
        """
        `index`番目のフレームの両端のサンプル数を返します.

        Args:
            index (int): フレーム番号
            frame_length (int): フレーム長
            frame_shift (int): シフト長

        Returns:
            Tuple[int, int]: フレームの両端のサンプル数
        """
        start = index * frame_shift
        end = start + frame_length
        return start, end

    def to_patches(self, frames: int) -> np.ndarray:
        """
        フレームの系列を`frames`ごとにグループ化し, パッチにしたものを返します.
        フレーム数が足りない場合, パッチの大きさになるよう末尾にゼロ埋めされます.

        ```python
        data = np.arange(1000).reshape(20, 50)
        print(data.shape) # (20, 50)
        series = FrameSeries(data, 1, 1)
        patches = series.to_patches(2)
        print(patches.shape) # (10, 2, 50)
        ```

        Args:
            frames (int): グループ化するフレーム数

        Returns:
            np.ndarray: パッチ化された系列 (パッチ数, `frames`, 元のデータの列数)
        """
        padded: np.ndarray = np.pad(
            self.data,
            ((0, (frames - (self.data.shape[0] % frames))), (0, 0)),
        )

        return padded.reshape(padded.shape[0] // frames, frames, padded.shape[1])

    def linear_to_dB(self) -> "FrameSeries":
        """
        このフレームの系列をdB値に変換します.
        すでにdB値である場合または変換できない系列の場合は変換せずそのまま返します.

        Returns:
            FrameSeries: dB値に変換されたフレームの系列
        """
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
        """
        このフレームの系列をdB値からリニアに変換します.
        すでにリニアである場合または変換できない系列の場合は変換せずそのまま返します.
        元の系列がパワー値だった場合, パワー値として返されます.

        Returns:
            FrameSeries: リニアに変換されたフレームの系列. 元の系列がパワー値だった場合パワーの系列
        """

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
        """
        このフレームの系列をパワーに変換します.
        すでにパワーである場合または変換できない系列の場合は変換せずそのまま返します.

        Returns:
            FrameSeries: パワーに変換されたフレームの系列
        """
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
        """
        このフレームの系列をパワーからリニアに変換します.
        すでにリニアである場合または変換できない系列の場合は変換せずそのまま返します.

        Returns:
            FrameSeries: リニアに変換されたフレームの系列
        """

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
        """
        フレームの系列を2次元プロットします.

        Args:
            up_to_nyquist (bool, optional): ナイキスト周波数(フレーム長 / 2 + 1点)までのプロットかどうか
            show (bool, optional): プロットを表示するかどうか
            save_fig_path (Optional[str], optional): プロットの保存先のパス. `None`の場合保存は行われません
            color_map (str, optional): プロットに使用するカラーマップ
        """
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
    """
    時間波形のフレームの系列を扱うクラスです.
    """

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
        """
        時間波形から各種パラメータを使ってフレームの系列に変換し, `Waveform`インスタンスを生成します.

        Args:
            time_series (np.ndarray): 時間波形(1次元)
            frame_length (int): フレーム長
            frame_shift (int): シフト長
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

        return cls(frames_np, frame_length, frame_shift, dB=None, power=None)

    def to_spectrum(
        self,
        fft_point: Optional[int] = None,
        window: Union[str, np.ndarray] = "hann",
    ) -> "Spectrum":
        """
        時間波形のフレームの系列をスペクトル(位相情報含む)に変換します.

        Args:
            fft_point (Optional[int], optional): FFTポイント数. `None`の場合, フレーム長が使用されます
            window (Union[str, np.ndarray], optional): 使用する窓関数. scipyの関数名での指定か自身で作成した窓関数を指定します

        Raises:
            TypeError: 窓関数の指定が`str`または`np.ndarray`でない場合

        Returns:
            Spectrum: スペクトル
        """
        if type(window) is str:
            window_func: np.ndarray = get_window(window, self.frame_length)
        elif type(window) is np.ndarray:
            window_func = window
        else:
            raise TypeError("Type of window must str or np.ndarray")

        if fft_point is None:
            fft_point = self.shape[1]

        to_spectrum = lambda frame: np.fft.fft(frame, n=fft_point)
        spectrum = to_spectrum(self.data * window_func)

        return Spectrum(
            spectrum, self.frame_length, self.frame_shift, dB=None, power=None
        )


class Spectrum(FrameSeries):
    """
    スペクトル(位相情報含む)のフレームの系列を扱うクラスです.
    """

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
            dB=None,
            power=None,
        )


class AmplitudeSpectrum(FrameSeries):
    """
    振幅スペクトルのフレームの系列を扱うクラスです.
    """

    def to_cepstrum(self) -> "Cepstrum":
        """
        振幅スペクトルをケプストラムに変換します.

        Raises:
            RuntimeError: この系列がパワーもしくはdB値になっている場合

        Returns:
            Cepstrum: ケプストラム
        """
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
        """
        振幅スペクトルをメルスペクトルに変換します.

        Args:
            fs (int): サンプリング周波数
            bins (int): メルのビン数

        Returns:
            MelSpectrum: メルスペクトル
        """
        filter = librosa.filters.mel(sr=fs, n_fft=self.shape[1], n_mels=bins)

        return MelSpectrum(
            np.dot(filter, self.data[:, : self.shape[1] // 2 + 1].T).T,
            self.frame_length,
            self.frame_shift,
            dB=self.dB,
            power=self.power,
        )


class PhaseSpectrum(FrameSeries):
    """
    位相スペクトルのフレームの系列を扱うクラスです.
    """

    pass


class MelSpectrum(FrameSeries):
    """
    メルスペクトルのフレームの系列を扱うクラスです.
    """

    def to_cepstrum(self) -> "MelCepstrum":
        """
        メルスペクトルをメルケプストラムに変換します.

        Raises:
            RuntimeError: この値がパワーまたはdB値の場合

        Returns:
            MelCepstrum: メルケプストラム
        """
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
    """
    ケプストラムのフレームの系列を扱うクラスです.
    """

    def to_spectrum(self) -> "AmplitudeSpectrum":
        """
        ケプストラムから振幅スペクトルに変換します.

        Returns:
            AmplitudeSpectrum: 振幅スペクトル
        """
        return AmplitudeSpectrum(
            np.exp(np.real(np.fft.fft(self.data))),
            self.frame_length,
            self.frame_shift,
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
        """
        ケプストラムからメルケプストラムに変換します.
        この処理は処理時間がかかるので注意してください.

        Args:
            bins (int): メルのビン数
            alpha (float): 伸縮率

        Returns:
            MelCepstrum: メルケプストラム
        """
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
    def __freqt(cepstrum: np.ndarray, n: int, bins: int, alpha: float) -> np.ndarray:
        """
        ケプストラムからメルケプストラムにするアルゴリズム..

        Args:
            cepstrum (np.ndarray): 変換元のケプストラム
            n (int): 使用するケプストラムの次数
            bins (int): メルのビン数
            alpha (float): 伸縮率

        Returns:
            np.ndarray: 伸縮されたケプストラム
        """
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
            dB=False,
            power=False,
        )
