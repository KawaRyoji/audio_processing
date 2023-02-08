from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, overload

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import Self, override


class FrameSeries:
    """
    フレーム単位の情報を扱うクラスです.
    継承して新しいプロパティを追加する場合, `copy_with()`と`properties()`を適切にオーバーライドしてください.
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
            ValueError: フレームの系列が2次元でない場合
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

    def apply(self, func: Callable[[np.ndarray], np.ndarray], axis: int = 1) -> Self:
        """
        `axis`に従って関数`func`を適用します.

        Args:
            func (Callable[[np.ndarray], np.ndarray]): 適用する関数
            axis (int, optional): 適用する軸

        Returns:
            Self: 関数を適用した後のフレームの系列
        """
        return self.copy_with(
            frame_series=np.apply_along_axis(func, axis=axis, arr=self.frame_series)
        )

    @overload
    def trim(self, end: int) -> Self:
        """
        時間軸でフレームの系列を切り取ります.
        指定した終了インデックスの部分は切り取った後のフレームの系列に含まれません.

        Args:
            end (int): 終了インデックス

        Returns:
            Self: 切り取った後のフレームの系列
        """
        return self.trim(0, end)

    @overload
    def trim(self, start: int, end: int) -> Self:
        """
        時間軸でフレームの系列を切り取ります.
        指定した終了インデックスの部分は切り取った後のフレームの系列に含まれません.

        Args:
            start (int): 開始インデックス
            end (int): 終了インデックス

        Returns:
            Self: 切り取った後のフレームの系列
        """
        return self.copy_with(frame_series=self.frame_series[start:end, :])

    def concat(self, *others: Self) -> Self:
        """
        時間方向にフレームの系列を結合します.
        連結できるインスタンスは同じプロパティで生成された者同士でなければなりません.

        Raises:
            ValueError: 引数に異なるプロパティで生成されたインスタンスがある場合

        Returns:
            Self: 結合されたインスタンス
        """
        if not all(
            [
                self.__class__ == other.__class__ and self.same_property(other)
                for other in others
            ]
        ):
            raise ValueError("結合する全てのインスタンスのプロパティが一致する必要があります")

        return self.copy_with(
            frame_series=np.concatenate(
                [self.frame_series] + [f.frame_series for f in others], axis=1
            )
        )

    def join(self, *others: Self) -> Self:
        """
        自身のインスタンスを他のインスタンスの間にはさんで結合します.

        Example:

        ```python
        >>> a = FrameSeries(np.zeros(2).reshape((2, 1)), 1, 1)
        >>> b = FrameSeries(np.zeros(4).reshape((2, 2)) + 1, 1, 1)
        >>> c = FrameSeries(np.zeros(4).reshape((2, 2)) + 2, 1, 1)
        >>> d = FrameSeries(np.zeros(4).reshape((2, 2)) + 3, 1, 1)
        >>> print(a.join(b, c, d).frame_series)
        [[1. 1. 0. 2. 2. 0. 3. 3.]
         [1. 1. 0. 2. 2. 0. 3. 3.]]
        ```

        Raises:
            ValueError: 引数が1つの場合
            ValueError: 引数に異なるプロパティで生成されたインスタンスがある場合

        Returns:
            Self: 結合されたインスタンス
        """
        if len(others) < 2:
            raise ValueError("引数のインスタンスは2つ以上で設定してください")

        if not all(
            [
                self.__class__ == other.__class__ and self.same_property(other)
                for other in others
            ]
        ):
            raise ValueError("結合する全てのインスタンスのプロパティが一致する必要があります")

        return self.copy_with(
            frame_series=np.concatenate(
                [others[0].frame_series]
                + [
                    np.concatenate(
                        [self.frame_series, other.frame_series],
                        axis=1,
                    )
                    for other in others[1:]
                ],
                axis=1,
            )
        )

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

    def save(self, path: str, compress: bool = False, overwrite: bool = False) -> Self:
        """
        このフレームの系列とそれを生成したプロパティをnpzファイルに保存します.
        `overwrite`が`False`の場合かつすでにファイルが存在する場合上書きされません.

        Args:
            path (str): 保存先のパス
            compress (bool, optional): 圧縮するかどうか
            overwrite (bool, optional): 上書きを許可するかどうか

        Returns:
            Self: 自身のインスタンス
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
    def from_npz(cls, path: str) -> Self:
        """
        npzファイルからインスタンスを読み込みます.

        Args:
            path (str): npzファイルのパス

        Returns:
            Self: 読み込んだインスタンス
        """
        npz = np.load(path, allow_pickle=True)
        params = {k: npz[k] for k in npz.files}
        return cls(**params)

    def plot(self, color_map: str = "magma") -> None:
        """
        フレーム系列をプロットします.
        このメソッドをオーバーライドすることで`show()`と`save_as_fig()`の出力を任意のプロットに変更できます.
        オーバーライドしない場合, フレームの系列の1要素を1ドットとするカラーマップが表示されます.

        Args:
            color_map (str, optional): プロットに使用するカラーマップのタイプ
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

    def show_fig(
        self,
        color_map: str = "magma",
    ) -> Self:
        """
        `plot()`で作成されたプロットを表示します.

        Args:
            color_map (str, optional): プロットに使用するカラーマップのタイプ

        Returns:
            Self: 自身のインスタンス
        """
        self.plot(color_map=color_map)
        plt.show()
        return self

    def save_as_fig(self, path: str, color_map: str = "magma") -> Self:
        """
        `plot()`で作成されたプロットを保存します.

        Args:
            path (str): 保存先のパス
            color_map (str, optional): プロットに使用するカラーマップのタイプ

        Returns:
            Self: 自身のインスタンス
        """
        self.plot(color_map=color_map)
        plt.savefig(path)
        plt.close()
        return self

    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
        frame_length: Optional[int] = None,
        frame_shift: Optional[int] = None,
    ) -> Self:
        """
        引数の値を使って自身のインスタンスをコピーします.

        Args:
            frame_series (Optional[np.ndarray], optional): フレーム単位の系列
            frame_length (Optional[int], optional): フレーム長
            frame_shift (Optional[int], optional): フレームシフト

        Returns:
            Self: コピーしたインスタンス
        """
        frame_series = self.frame_series if frame_series is None else frame_series
        frame_length = self.frame_length if frame_length is None else frame_length
        frame_shift = self.frame_shift if frame_shift is None else frame_shift

        return FrameSeries(frame_series, frame_length, frame_shift)

    def same_property(self, other: Self) -> bool:
        """
        自身のインスタンスのプロパティともう一方が一致するかを確認します.

        Args:
            other (FrameSeries): 一方のインスタンス

        Returns:
            bool: 自身のインスタンスのプロパティともう一方のプロパティが一致するか
        """
        return self.properties() == other.properties()

    def dump(self) -> Self:
        """
        自身のインスタンスの内容を出力します.

        Returns:
            Self: 自身のインスタンス
        """
        print(self.__str__())
        return self

    def check_can_calc(self, other: Any) -> None:
        """
        2項演算が行えるかを確認し, できない場合はエラーを出します.

        Args:
            other (Any): 確認するインスタンス

        Raises:
            TypeError: 確認するインスタンスのクラスが自身のインスタンスと一致しない場合
            ValueError: 確認するインスタンスのプロパティが自身のインスタンスと一致しない場合
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                "2項演算の引数は同じクラスである必要があります. self:{} other:{}".format(
                    self.__class__.__name__, other.__class__.__name__
                )
            )

        if not self.same_property(other):
            raise ValueError(
                "同じプロパティで生成したインスタンス同士でしか演算できません. self:{} other:{}".format(
                    self.properties(), other.properties()
                )
            )

    def __len__(self) -> int:
        return self.frame_series.__len__()

    def __add__(self, other: Any) -> Self:
        self.check_can_calc(other)
        return self.copy_with(frame_series=self.frame_series + other.frame_series)

    def __sub__(self, other: Any) -> Self:
        self.check_can_calc(other)
        return self.copy_with(frame_series=self.frame_series - other.frame_series)

    def __mul__(self, other: Any) -> Self:
        self.check_can_calc(other)
        return self.copy_with(frame_series=self.frame_series * other.frame_series)

    def __truediv__(self, other: Any) -> Self:
        self.check_can_calc(other)
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
    """
    時間領域のフレームの系列を扱うクラスです.
    継承して新しいプロパティを追加する場合, `copy_with()`と`properties()`を適切にオーバーライドしてください.
    """

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
    @override
    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
        frame_length: Optional[int] = None,
        frame_shift: Optional[int] = None,
        fs: Optional[int] = None,
    ) -> Self:
        """
        引数の値を使って自身のインスタンスをコピーします.

        Args:
            frame_series (Optional[np.ndarray], optional): フレーム単位の系列
            frame_length (Optional[int], optional): フレーム長
            frame_shift (Optional[int], optional): フレームシフト
            fs (Optional[int], optional): サンプリング周波数

        Returns:
            Self: コピーしたインスタンス
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


class FreqDomainFrameSeries(FrameSeries):
    """
    周波数領域のフレームの系列を扱うクラスです.
    継承して新しいプロパティを追加する場合, `copy_with()`と`properties()`を適切にオーバーライドしてください.
    """

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

    def linear_to_dB(self) -> Self:
        """
        このフレームの系列をdB値に変換します.
        すでにdB値である場合は変換せずそのまま返します.

        Returns:
            Self: dB値に変換されたフレームの系列
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

    def dB_to_linear(self) -> Self:
        """
        このフレームの系列をdB値からリニアに変換します.
        すでにリニアである場合は変換せずそのまま返します.
        元の系列がパワー値だった場合, パワー値として返されます.

        Returns:
            Self: リニアに変換されたフレームの系列. 元の系列がパワー値だった場合パワーの系列
        """

        if not self.dB:
            print("この特徴量はすでに線形値です.")
            return self

        if self.power:
            linear_func = lambda x: np.power(x / 10, 10)
        else:
            linear_func = lambda x: np.power(x / 20, 10)

        return self.copy_with(frame_series=linear_func(self.frame_series), dB=False)

    def linear_to_power(self) -> Self:
        """
        このフレームの系列をパワーに変換します.
        すでにパワーである場合またはdB値である場合は変換せずそのまま返します.

        Returns:
            Self: パワーに変換されたフレームの系列
        """
        if self.power:
            print("この特徴量はすでにパワー値です.")
            return self
        elif self.dB:
            print("この特徴量はdB値です.")
            return self

        return self.copy_with(frame_series=np.power(self.frame_series, 2), power=True)

    def power_to_linear(self) -> Self:
        """
        このフレームの系列をパワーからリニアに変換します.
        すでにリニアである場合またはdB値の場合は変換せずそのまま返します.

        Returns:
            Self: リニアに変換されたフレームの系列
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

    @override
    def plot(
        self,
        up_to_nyquist: bool = True,
        color_map: str = "magma",
    ) -> Self:
        """
        周波数領域のフレームの系列を2次元プロットします.

        Args:
            up_to_nyquist (bool, optional): ナイキスト周波数までのプロットにするか
            color_map (str, optional): プロットに使用するカラーマップ
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

    @override
    def show_fig(self, up_to_nyquist: bool = True, color_map: str = "magma") -> Self:
        """
        `plot()`で作成されたプロットを表示します.

        Args:
            up_to_nyquist (bool, optional): ナイキスト周波数までのプロットにするか
            color_map (str, optional): プロットに使用するカラーマップのタイプ

        Returns:
            Self: 自身のインスタンス
        """
        self.plot(up_to_nyquist=up_to_nyquist, color_map=color_map)
        plt.show()
        return self

    @override
    def save_as_fig(
        self, path: str, up_to_nyquist: bool = True, color_map: str = "magma"
    ) -> Self:
        """
        `plot()`で作成されたプロットを保存します.

        Args:
            up_to_nyquist (bool, optional): ナイキスト周波数までのプロットにするか
            color_map (str, optional): プロットに使用するカラーマップのタイプ

        Returns:
            Self: 自身のインスタンス
        """
        self.plot(up_to_nyquist=up_to_nyquist, color_map=color_map)
        plt.savefig(path)
        plt.close()
        return self

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
    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
        frame_length: Optional[int] = None,
        frame_shift: Optional[int] = None,
        fft_point: Optional[int] = None,
        fs: Optional[int] = None,
        dB: Optional[bool] = None,
        power: Optional[bool] = None,
    ) -> Self:
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
            Self: コピーしたインスタンス
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
