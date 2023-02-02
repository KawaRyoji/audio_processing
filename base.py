import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, overload

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override


class FrameSeries:
    """
    フレーム単位の情報を扱うクラスです.
    """

    def __init__(
        self,
        frame_series: np.ndarray,
        frame_length: int,
        frame_shift: int,
    ) -> None:
        """
        Args:
            frame_series (np.ndarray): フレーム単位の系列(2次元を想定)
            frame_length (int): フレーム長
            frame_shift (int): シフト長

        Raises:
            ValueError:
        """
        if len(frame_series.shape) != 2:
            raise ValueError("フレームの系列は2次元でなければなりません.")

        self.__frame_series = frame_series
        self.__frame_length = frame_length
        self.__frame_shift = frame_shift

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
    def frame_series(self) -> np.ndarray:
        """
        フレームの系列を返します.

        Returns:
            np.ndarray: フレームの系列
        """
        return self.__frame_series

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
            self.frame_series,
            ((0, (frames - (self.frame_series.shape[0] % frames))), (0, 0)),
        )

        return padded.reshape(padded.shape[0] // frames, frames, padded.shape[1])

    def apply(
        self, func: Callable[[np.ndarray], np.ndarray], axis: int = 1
    ) -> "FrameSeries":
        """
        `axis`に従って関数`func`を適用します.

        Args:
            func (Callable[[np.ndarray], np.ndarray]): 適用する関数
            axis (int, optional): 適用する軸

        Returns:
            FrameSeries: 関数を適用した後のフレームの系列
        """
        return self.copy_with(
            frame_series=np.apply_along_axis(func, axis=axis, arr=self.frame_series)
        )

    @overload
    def trim(self, end: int) -> "FrameSeries":
        """
        時間軸でフレームの系列を切り取ります.
        指定した終了インデックスの部分は切り取った後のフレームの系列に含まれません.

        Args:
            end (int): 終了インデックス

        Returns:
            FrameSeries: 切り取った後のフレームの系列
        """
        return self.trim(0, end)

    @overload
    def trim(self, start: int, end: int) -> "FrameSeries":
        """
        時間軸でフレームの系列を切り取ります.
        指定した終了インデックスの部分は切り取った後のフレームの系列に含まれません.

        Args:
            start (int): 開始インデックス
            end (int): 終了インデックス

        Returns:
            FrameSeries: 切り取った後のフレームの系列
        """
        return self.copy_with(frame_series=self.frame_series[start:end, :])

    def properties(self) -> dict[str, Any]:
        """
        このフレームの系列を生成したプロパティを辞書形式で返します

        Returns:
            dict[str, Any]: プロパティの辞書
        """
        return {
            "frame_length": self.frame_length,
            "frame_shift": self.frame_shift,
        }

    def save(
        self, path: str, compress: bool = False, overwrite: bool = False
    ) -> "FrameSeries":
        """
        このフレームの系列とそれを生成したプロパティをnpzファイルに保存します.
        `overwrite`が`False`の場合かつすでにファイルが存在する場合上書きされません.

        Args:
            path (str): 保存先のパス
            compress (bool, optional): 圧縮するかどうか
            overwrite (bool, optional): 上書きを許可するかどうか

        Returns:
            FrameSeries: 自身のインスタンス
        """
        if not overwrite and os.path.exists(path):
            print(path, "は既に存在します")
            return self

        Path(path).parent.mkdir(exist_ok=True)

        if compress:
            np.savez_compressed(
                path, frame_series=self.frame_series, **self.properties()
            )
        else:
            np.savez(path, frame_series=self.frame_series, **self.properties())

        return self

    @classmethod
    def from_npz(cls, path: str) -> "FrameSeries":
        """
        npzファイルからインスタンスを読み込みます.

        Args:
            path (str): npzファイルのパス

        Returns:
            FrameSeries: 読み込んだインスタンス
        """
        file = np.load(path, allow_pickle=True)

        return cls(file["frame_series"], file["frame_length"], file["frame_shift"])

    def plot(
        self,
        show: bool = True,
        save_fig_path: Optional[str] = None,
        color_map: str = "magma",
    ) -> "FrameSeries":
        """
        フレームの系列を2次元プロットします.

        Args:
            show (bool, optional): プロットを表示するかどうか
            save_fig_path (Optional[str], optional): プロットの保存先のパス. `None`の場合保存は行われません
            color_map (str, optional): プロットに使用するカラーマップ

        Returns:
            FrameSeries: 自身のインスタンス
        """
        fig, ax = plt.subplots(
            dpi=100,
            figsize=(self.shape[0] / 100, self.shape[1] / 100),
        )

        ax.pcolor(self.frame_series.T, cmap=color_map)
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

        return self

    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
        frame_length: Optional[int] = None,
        frame_shift: Optional[int] = None,
    ) -> "FrameSeries":
        """
        引数の値を使って自身のインスタンスをコピーします.

        Args:
            frame_series (Optional[np.ndarray], optional): フレーム単位の系列
            frame_length (Optional[int], optional): フレーム長
            frame_shift (Optional[int], optional): フレームシフト

        Returns:
            FrameSeries: コピーしたインスタンス
        """
        frame_series = self.frame_series if frame_series is None else frame_series
        frame_length = self.frame_length if frame_length is None else frame_length
        frame_shift = self.frame_shift if frame_shift is None else frame_shift

        return FrameSeries(frame_series, frame_length, frame_shift)

    def equals_property(self, other: "FrameSeries") -> None:
        """
        自身のインスタンスのプロパティともう一方が一致するかを確認します.

        Args:
            other (FrameSeries): 一方のインスタンス

        Raises:
            ValueError: プロパティが一致しない場合
        """
        if self.shape != other.shape:
            raise ValueError(
                "データの次元を一致させてください. self:{} other:{}".format(self.shape, other.shape)
            )

        if self.frame_length != other.frame_length:
            raise ValueError(
                "フレーム長を一致させてください. self:{} other:{}".format(
                    self.frame_length, other.frame_length
                )
            )

        if self.frame_shift != other.frame_shift:
            raise ValueError(
                "フレームシフトを一致させてください. self:{} other:{}".format(
                    self.frame_shift, other.frame_shift
                )
            )

    def dump(self) -> "FrameSeries":
        """
        自身のインスタンスの内容を出力します.

        Returns:
            FrameSeries: 自身のインスタンス
        """
        print(self.__str__())
        return self

    def __len__(self) -> int:
        return self.frame_series.__len__()

    def __add__(self, other: Any) -> "FrameSeries":
        if not isinstance(other, self.__class__):
            raise TypeError(
                "加算は同じクラスである必要があります. other:{}".format(other.__class__.__name__)
            )

        self.equals_property(other)

        return self.copy_with(frame_series=self.frame_series + other.frame_series)

    def __sub__(self, other: Any) -> "FrameSeries":
        if not isinstance(other, self.__class__):
            raise TypeError("減算は同じクラスである必要があります. other:{}".format(other.__class__))

        self.equals_property(other)

        return self.copy_with(frame_series=self.frame_series - other.frame_series)

    def __mul__(self, other: Any) -> "FrameSeries":
        if not isinstance(other, self.__class__):
            raise TypeError("乗算は同じクラスである必要があります. other:{}".format(other.__class__))

        self.equals_property(other)

        return self.copy_with(frame_series=self.frame_series * other.frame_series)

    def __truediv__(self, other: Any) -> "FrameSeries":
        if not isinstance(other, self.__class__):
            raise TypeError("乗算は同じクラスである必要があります. other:{}".format(other.__class__))

        self.equals_property(other)

        return self.copy_with(frame_series=self.frame_series / other.frame_series)

    def __str__(self) -> str:
        string = "Feature Summary\n"
        string += "------------------------------------\n"
        string += "type: " + self.__class__.__name__ + "\n"
        string += "data shape: {}\n".format(self.shape)
        for feature, value in self.properties().items():
            string += "{}: {}\n".format(feature, value)
        string += "------------------------------------\n"

        return string + "\n"


class TimeDomainFrameSeries(FrameSeries):
    def __init__(
        self,
        frame_series: np.ndarray,
        frame_length: int,
        frame_shift: int,
        fs: int,
    ) -> None:
        """
        Args:
            frame_series (np.ndarray): フレーム単位の系列(Frames, Time domain feature)
            frame_length (int): フレーム長
            frame_shift (int): シフト長
            fs (int): サンプリング周波数
        """
        super().__init__(frame_series, frame_length, frame_shift)
        self.__fs = fs

    @property
    def fs(self) -> int:
        """
        この系列を生成したサンプリング周波数を返します.

        Returns:
            int: サンプリング周波数
        """
        return self.__fs

    # 以下継承したメソッド

    @overload
    def trim(self, end: int) -> "TimeDomainFrameSeries":
        return super().trim(end)

    @overload
    def trim(self, start: int, end: int) -> "TimeDomainFrameSeries":
        return super().trim(start, end)

    def plot(
        self,
        show: bool = True,
        save_fig_path: Optional[str] = None,
        color_map: str = "magma",
    ) -> "TimeDomainFrameSeries":
        return super().plot(show=show, save_fig_path=save_fig_path, color_map=color_map)

    @override
    def equals_property(self, other: "TimeDomainFrameSeries") -> None:
        super().equals_property(other)

        if self.fs != other.fs:
            ValueError(
                "サンプリング周波数を一致させてください. self:{} other:{}".format(self.fs, other.fs)
            )

    @override
    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
        frame_length: Optional[int] = None,
        frame_shift: Optional[int] = None,
        fs: Optional[int] = None,
    ) -> "TimeDomainFrameSeries":
        """
        引数の値を使って自身のインスタンスをコピーします.

        Args:
            frame_series (Optional[np.ndarray], optional): フレーム単位の系列
            frame_length (Optional[int], optional): フレーム長
            frame_shift (Optional[int], optional): フレームシフト
            fs (Optional[int], optional): サンプリング周波数

        Returns:
            TimeDomainFrameSeries: コピーしたインスタンス
        """
        frame_series = self.frame_series if frame_series is None else frame_series
        frame_length = self.frame_length if frame_length is None else frame_length
        frame_shift = self.frame_shift if frame_shift is None else frame_shift
        fs = self.fs if fs is None else fs

        return TimeDomainFrameSeries(frame_series, frame_length, frame_shift, fs)

    @override
    def properties(self) -> dict[str, Any]:
        properties = super().properties()
        properties.update({"fs": self.__fs})
        return properties

    def save(
        self, path: str, compress: bool = False, overwrite: bool = False
    ) -> "TimeDomainFrameSeries":
        return super().save(path, compress=compress, overwrite=overwrite)

    @override
    @classmethod
    def from_npz(cls, path: str) -> "TimeDomainFrameSeries":
        file = np.load(path, allow_pickle=True)

        return cls(
            file["frame_series"], file["frame_length"], file["frame_shift"], file["fs"]
        )

    def dump(self) -> "TimeDomainFrameSeries":
        return super().dump()

    def __add__(self, other: Any) -> "TimeDomainFrameSeries":
        return super().__add__(other)

    def __mul__(self, other: Any) -> "TimeDomainFrameSeries":
        return super().__mul__(other)

    def __sub__(self, other: Any) -> "TimeDomainFrameSeries":
        return super().__sub__(other)

    def __truediv__(self, other: Any) -> "TimeDomainFrameSeries":
        return super().__truediv__(other)


class FreqDomainFrameSeries(FrameSeries):
    def __init__(
        self,
        frame_series: np.ndarray,
        frame_length: int,
        frame_shift: int,
        fft_point: int,
        fs: int,
        dB: bool = False,
        power: bool = False,
    ) -> None:
        """
        Args:
            frame_series (np.ndarray): フレーム単位の系列(Frames, Frequency domain feature)
            frame_length (int): フレーム長
            frame_shift (int): シフト長
            fft_point (int): FFTポイント数
            fs (int): サンプリング周波数
            dB (bool, optional): この系列がdB値であるか. Trueの場合, この系列はdB値であることを示します.
            power (bool, optional): この系列がパワーであるか. Trueの場合, この系列はパワー値であることを示します.
        """
        super().__init__(frame_series, frame_length, frame_shift)
        self.__fs = fs
        self.__fft_point = fft_point
        self.__dB = dB
        self.__power = power

    @property
    def dB(self) -> bool:
        """
        この系列がdB値であるかを返します.
        Trueの場合このフレームの系列はdB値です.

        Returns:
            bool: この系列がdB値であるかどうか.
        """
        return self.__dB

    @property
    def power(self) -> bool:
        """
        この系列がパワーであるかを返します.
        Trueの場合このフレームの系列はパワー値です.

        Returns:
            bool: この系列がパワーであるかどうか.
        """
        return self.__power

    @property
    def fs(self) -> int:
        """
        この系列を生成したサンプリング周波数を返します.

        Returns:
            int: サンプリング周波数
        """
        return self.__fs

    @property
    def fft_point(self) -> int:
        """
        この系列を生成したFFTポイント数を返します.

        Returns:
            int: FFTポイント数
        """
        return self.__fft_point

    def linear_to_dB(self) -> "FreqDomainFrameSeries":
        """
        このフレームの系列をdB値に変換します.
        すでにdB値である場合は変換せずそのまま返します.

        Returns:
            FreqDomainFrameSeries: dB値に変換されたフレームの系列
        """
        if self.dB:
            print("この特徴量はすでにdB値です.")
            return self

        if self.power:
            dB_func = lambda frame: 10 * np.log10(
                frame, out=np.zeros_like(frame), where=frame != 0
            )
        else:
            dB_func = lambda frame: 20 * np.log10(
                frame, out=np.zeros_like(frame), where=frame != 0
            )

        return self.copy_with(frame_series=dB_func(self.frame_series), dB=True)

    def dB_to_linear(self) -> "FreqDomainFrameSeries":
        """
        このフレームの系列をdB値からリニアに変換します.
        すでにリニアである場合は変換せずそのまま返します.
        元の系列がパワー値だった場合, パワー値として返されます.

        Returns:
            FreqDomainFrameSeries: リニアに変換されたフレームの系列. 元の系列がパワー値だった場合パワーの系列
        """

        if not self.dB:
            print("この特徴量はすでに線形値です.")
            return self

        if self.power:
            linear_func = lambda x: np.power(x / 10, 10)
        else:
            linear_func = lambda x: np.power(x / 20, 10)

        return self.copy_with(frame_series=linear_func(self.frame_series), dB=False)

    def linear_to_power(self) -> "FreqDomainFrameSeries":
        """
        このフレームの系列をパワーに変換します.
        すでにパワーである場合またはdB値である場合は変換せずそのまま返します.

        Returns:
            FrameSeries: パワーに変換されたフレームの系列
        """
        if self.power:
            print("この特徴量はすでにパワー値です.")
            return self
        elif self.dB:
            print("この特徴量はdB値です.")
            return self

        return self.copy_with(frame_series=np.power(self.frame_series, 2), power=True)

    def power_to_linear(self) -> "FreqDomainFrameSeries":
        """
        このフレームの系列をパワーからリニアに変換します.
        すでにリニアである場合またはdB値の場合は変換せずそのまま返します.

        Returns:
            FrameSeries: リニアに変換されたフレームの系列
        """

        if not self.power:
            print("この特徴量はすでに線形値です.")
            return self
        elif self.dB:
            print("この特徴量はdB値です.")
            return self

        return self.copy_with(frame_series=np.sqrt(self.frame_series), power=False)

    @staticmethod
    def to_symmetry(series: np.ndarray) -> np.ndarray:
        """
        (time, fft_point // 2 + 1)の非対称スペクトログラムを対称にします.

        Args:
            series (np.ndarray): 非対称スペクトログラム

        Returns:
            np.ndarray: 対称スペクトログラム (time, fft_point)
        """
        return np.concatenate([series, np.fliplr(series)[:, 1:-1]], axis=1)

    # 以下継承したメソッド

    @overload
    def trim(self, end: int) -> "FreqDomainFrameSeries":
        return super().trim(end)

    @overload
    def trim(self, start: int, end: int) -> "FreqDomainFrameSeries":
        return super().trim(start, end)

    @override
    def plot(
        self,
        up_to_nyquist: bool = True,
        show: bool = True,
        save_fig_path: Optional[str] = None,
        color_map: str = "magma",
    ) -> "FreqDomainFrameSeries":
        """
        フレームの系列を2次元プロットします.

        Args:
            show (bool, optional): プロットを表示するかどうか
            save_fig_path (Optional[str], optional): プロットの保存先のパス. `None`の場合保存は行われません
            color_map (str, optional): プロットに使用するカラーマップ

        Returns:
            FrameSeries: 自身のインスタンス
        """
        fig, ax = plt.subplots(
            dpi=100,
            figsize=(self.shape[0] / 100, self.shape[1] / 100),
        )

        if up_to_nyquist:
            show_data = self.frame_series[:, : self.shape[1] // 2 + 1]
        else:
            show_data = self.frame_series

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

        return self

    def save(
        self, path: str, compress: bool = False, overwrite: bool = False
    ) -> "FreqDomainFrameSeries":
        return super().save(path, compress=compress, overwrite=overwrite)

    @override
    def properties(self) -> dict[str, Any]:
        properties = super().properties()
        properties.update(
            {
                "fs": self.fs,
                "fft_point": self.fft_point,
                "dB": self.dB,
                "power": self.power,
            }
        )
        return properties

    @override
    @classmethod
    def from_npz(cls, path: str) -> "FreqDomainFrameSeries":
        file = np.load(path, allow_pickle=True)
        return cls(
            file["frame_series"],
            file["frame_length"],
            file["frame_shift"],
            file["fft_point"],
            file["fs"],
            dB=file["dB"],
            power=file["power"],
        )

    @override
    def equals_property(self, other: "FreqDomainFrameSeries") -> None:
        super().equals_property(other)

        if self.fs != other.fs:
            ValueError(
                "サンプリング周波数を一致させてください. self:{} other:{}".format(self.fs, other.fs)
            )

        if self.fft_point != other.fft_point:
            ValueError(
                "FFTポイント数を一致させてください. self:{} other:{}".format(
                    self.fft_point, other.fft_point
                )
            )

        if self.dB != other.dB:
            ValueError("dB値であるか一致させてください. self:{} other:{}".format(self.dB, other.dB))

        if self.power != other.power:
            ValueError(
                "パワー値であるか一致させてください. self:{} other:{}".format(self.power, other.power)
            )

    @override
    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
        frame_length: Optional[int] = None,
        frame_shift: Optional[int] = None,
        fft_point: Optional[int] = None,
        fs: Optional[int] = None,
        dB: Optional[bool] = None,
        power: Optional[bool] = None,
    ) -> "FreqDomainFrameSeries":
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
            FreqDomainFrameSeries: コピーしたインスタンス
        """
        frame_series = self.frame_series if frame_series is None else frame_series
        frame_length = self.frame_length if frame_length is None else frame_length
        frame_shift = self.frame_shift if frame_shift is None else frame_shift
        fft_point = self.fft_point if fft_point is None else fft_point
        fs = self.fs if fs is None else fs
        dB = self.dB if dB is None else dB
        power = self.power if power is None else power

        return FreqDomainFrameSeries(
            frame_series, frame_length, frame_shift, fft_point, fs, dB=dB, power=power
        )

    def dump(self) -> "FreqDomainFrameSeries":
        return super().dump()

    def __add__(self, other: Any) -> "FreqDomainFrameSeries":
        return super().__add__(other)

    def __mul__(self, other: Any) -> "FreqDomainFrameSeries":
        return super().__mul__(other)

    def __sub__(self, other: Any) -> "FreqDomainFrameSeries":
        return super().__sub__(other)

    def __truediv__(self, other: Any) -> "FreqDomainFrameSeries":
        return super().__truediv__(other)
