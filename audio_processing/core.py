from __future__ import annotations

import inspect
import os
import re
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Union

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from typing_extensions import Self, override

if TYPE_CHECKING:
    from .features import ComplexCQT, WaveformFrameSeries


class Audio:
    def __init__(self, data: np.ndarray, fs: int, dtype: Optional[type] = None) -> None:
        """
        Args:
            data (np.ndarray): 時間波形 (1次元を想定)
            fs (int): サンプリング周波数
            dtype (Optional[type], optional): データタイプ. `None`の場合`data`のデータタイプになります

        Raises:
            ValueError: 入力データが1次元出ない場合
        """
        if data.ndim != 1:
            raise ValueError("時間波形は1次元配列でなければなりません")

        self.__fs = fs
        self.__data = data.astype(dtype=dtype) if dtype is not None else data

    @property
    def fs(self) -> int:
        """
        サンプリング周波数
        """
        return self.__fs

    @property
    def data(self) -> np.ndarray:
        """
        オーディオファイルのデータ
        """
        return self.__data

    @property
    def dtype(self) -> type:
        """
        音データのデータタイプ
        """
        return self.__data.dtype

    @property
    def sec(self) -> float:
        """
        音データの秒数
        """
        return self.data.shape[0] / self.fs

    @classmethod
    def read(
        cls, path: str, fs: Optional[int] = None, dtype: type = np.float32
    ) -> Self:
        """
        オーディオファイルからインスタンスを生成します. 対応しているフォーマットは`librosa.core.load`を確認してください.

        Args:
            path (str): オーディオファイルパス
            fs (Optional[int], optional): サンプリング周波数. Noneの場合読み込んだオーディオファイルのサンプリング周波数になります
            dtype (type, optional): データタイプ

        Returns:
            Audio: 生成したインスタンス
        """
        data, fs = librosa.core.load(path, sr=fs, dtype=dtype)
        return cls(data, fs, dtype=data.dtype)

    @classmethod
    def from_noise(
        cls,
        fs: int,
        sec: float,
        sigma: float,
        mu: float = 0,
        dtype: type = np.float32,
    ) -> Self:
        """
        ガウシアンノイズから音を生成します.

        Args:
            fs (int): サンプリング周波数
            sec (float): 継続時間
            sigma (float): 標準偏差
            mu (float, optional): 平均
            dtype (type, optional): データタイプ

        Returns:
            Audio: 生成した音
        """
        length = int(fs * sec)
        signal = np.random.normal(mu, sigma, length)

        return cls(signal, fs, dtype=dtype)

    def resample(self, target_fs: int) -> Self:
        """
        サンプリング周波数を変換します.
        目標のサンプリング周波数が現在のサンプリング周波数と同じ場合は自身のインスタンスを返します.

        Args:
            target_fs (int): 目標のサンプリング周波数

        Returns:
            Audio: サンプリング周波数を変換したインスタンス
        """
        if self.fs == target_fs:
            return self

        resampled: np.ndarray = librosa.core.resample(
            y=self.data, orig_sr=self.fs, target_sr=target_fs
        )

        return Audio(resampled, target_fs, dtype=resampled.dtype)

    def normalize(self) -> Self:
        """
        音データを正規化し, 振幅を[-1, 1]にします.

        Returns:
            Audio: 正規化した音データのオーディオファイル
        """
        d_max = np.max(np.abs(self.data))

        return Audio(self.data / d_max, self.fs, dtype=self.dtype)

    def pad(
        self, pad_width: Union[int, tuple[int, int]], value: Union[int, float] = 0
    ) -> Self:
        return Audio(
            np.pad(self.data, pad_width, mode="constant", constant_values=value),
            self.fs,
        )

    def save(
        self,
        path: str,
        target_bits: int = 16,
        normalize: bool = True,
        overwrite: bool = False,
    ) -> Self:
        """
        オーディオファイルに書き込みます.
        `overwrite`が`False`の場合かつすでにファイルが存在する場合上書きされません.
        対応しているフォーマットは`soundfile.write`を確認してください.

        Args:
            path (str): 保存先のパス
            target_bits (int, optional): 書き込むbit数
            normalize (bool, optional): 書き込む前に上書きするかどうか
            overwrite (bool, optional): 上書きを許可するかどうか

        Returns:
            Audio: 自身のインスタンス
        """
        if not overwrite and os.path.exists(path):
            print(path, "は既に存在します")
            return self

        data = self.normalize().data if normalize else self.data

        Path(path).parent.mkdir(exist_ok=True)

        soundfile.write(path, data, self.fs, "PCM_{}".format(target_bits))

        return self

    def plot(self) -> None:
        """
        音データをプロットします.
        """
        x = np.arange(self.data.shape[0]) / self.fs
        plt.plot(x, self.data)
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Amplitude", fontsize=16)
        plt.grid(True)
        plt.tight_layout()

    def show_fig(self) -> Self:
        """
        音データをプロットしたものを表示します

        Returns:
            Self: 自身のインスタンス
        """
        self.plot()
        plt.show()
        return self

    def save_as_fig(self, path: str) -> Self:
        """
        音データをプロットしたものを保存します.

        Args:
            path (str): 保存するパス

        Returns:
            Self: 自身のインスタンス
        """
        self.plot()
        plt.savefig(path)
        plt.close()
        return self

    def to_frames(
        self,
        frame_length: int,
        frame_shift: int,
        padding: bool = True,
        padding_mode: str = "reflect",
        dtype: type = np.float32,
    ) -> WaveformFrameSeries:
        """
        フレーム系列の時間波形に変換します.

        Args:
            frame_length (int): フレーム長
            frame_shift (int): フレームシフト
            padding (bool, optional): パディングするかどうか
            padding_mode (str, optional): パディングの方法. numpyのpad関数に使用します
            dtype (type, optional): データタイプ

        Returns:
            Waveform: フレーム系列の時間波形
        """
        from audio_processing.features import WaveformFrameSeries

        return WaveformFrameSeries.from_waveform(
            self.data,
            frame_length,
            frame_shift,
            self.fs,
            padding=padding,
            padding_mode=padding_mode,
            dtype=dtype,
        )

    def to_cqt(
        self,
        frame_shift: int,
        fmin: float,
        n_bins: int = 180,
        bins_per_octave: int = 36,
        window: str = "hann",
    ) -> ComplexCQT:
        from audio_processing.features import ComplexCQT

        cqt = librosa.cqt(
            self.data,
            sr=self.fs,
            hop_length=frame_shift,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            window=window,
        )

        return ComplexCQT(
            cqt.T,
            frame_shift,
            self.fs,
            fmin,
            bins_per_octave=bins_per_octave,
        )

    def to_npz(
        self, path: str, compress: bool = False, overwrite: bool = False
    ) -> Self:
        """
        自身のインスタンスをnpzファイルに保存します.
        `overwrite`が`False`の場合かつすでにファイルが存在する場合上書きされません.

        Args:
            path (str): 保存先のパス
            compress (bool, optional): 圧縮するかどうか
            overwrite (bool, optional): 上書きを許可するかどうか

        Returns:
            Audio: 自身のインスタンス
        """
        if not overwrite and os.path.exists(path):
            print(path, "は既に存在します")
            return self

        Path(path).parent.mkdir(exist_ok=True)

        if compress:
            np.savez_compressed(path, data=self.data, fs=self.fs)
        else:
            np.savez(path, data=self.data, fs=self.fs)

        return self

    @classmethod
    def from_npz(cls, path: str) -> Audio:
        """
        npzファイルからインスタンスを生成します.

        Args:
            path (str): npzファイルのパス

        Returns:
            Audio: 生成したインスタンス
        """
        file = np.load(path, allow_pickle=True)
        data: np.ndarray = file["data"]
        fs: int = file["fs"]

        return cls(data, fs, dtype=data.dtype)

    def __str__(self) -> str:
        return "length: {}\nsampling frequency: {}".format(self.data.shape, self.fs)

    def __len__(self) -> int:
        return self.data.__len__()


class StereoAudio:
    def __init__(self, left_channel: Audio, right_channel: Audio) -> None:
        """
        Args:
            left_channel (Audio): 左チャネルのオーディオファイル
            right_channel (Audio): 右チャネルのオーディオファイル

        Raises:
            ValueError: 各チャネルの長さが異なる場合
            ValueError: 各チャネルのサンプリング周波数が異なる場合
        """
        if len(left_channel) != len(right_channel):
            raise ValueError(
                "各チャネルの長さが異なります left: {}, right:{}".format(
                    len(left_channel), len(right_channel)
                )
            )

        if left_channel.fs != right_channel.fs:
            raise ValueError(
                "各チャネルのサンプリング周波数が異なります left: {}, right:{}".format(
                    left_channel.fs, right_channel.fs
                )
            )

        self.__left_channel = left_channel
        self.__right_channel = right_channel

    @property
    def fs(self) -> int:
        """
        サンプリング周波数
        """
        return self.left_channel.fs

    @property
    def left_channel(self) -> Audio:
        """
        左チャネルのオーディオファイル
        """
        return self.__left_channel

    @property
    def right_channel(self) -> Audio:
        """
        右チャネルのオーディオファイル
        """
        return self.__right_channel

    @property
    def dtype(self) -> type:
        """
        音データのデータタイプ
        """
        return self.left_channel.dtype

    @classmethod
    def read(
        cls, path: str, fs: Optional[int] = None, dtype: type = np.float32
    ) -> Self:
        """
        オーディオファイルからインスタンスを生成します. 対応しているフォーマットは`librosa.core.load`を確認してください.

        Raises:
            RuntimeError: 読み込んだファイルがステレオ音源でない場合
        Args:
            path (str): オーディオファイルパス
            fs (Optional[int], optional): サンプリング周波数. Noneの場合読み込んだオーディオファイルのサンプリング周波数になります
            dtype (type, optional): データタイプ

        Returns:
            StereoAudio: 生成したインスタンス
        """
        data, fs = librosa.core.load(path, sr=fs, dtype=dtype)
        if data.ndim != 2:
            raise RuntimeError("読み込んだファイルはステレオ音源ではありません")

        return cls.from_numpy(data[0], data[1], fs, dtype=dtype)

    @classmethod
    def from_numpy(
        cls,
        left_channel: np.ndarray,
        right_channel: np.ndarray,
        fs: int,
        dtype: Optional[type] = None,
    ) -> Self:
        """
        numpy配列からステレオオーディオファイルを生成します

        Args:
            left_channel (np.ndarray): 左チャネルのデータ
            right_channel (np.ndarray): 右チャネルのデータ
            fs (int): サンプリング周波数
            dtype (Optional[type], optional): データタイプ

        Returns:
            StereoAudio: 生成したステレオオーディオファイル
        """
        return cls(
            Audio(left_channel, fs, dtype=dtype), Audio(right_channel, fs, dtype=dtype)
        )

    def to_audio_instances(self) -> tuple[Audio, Audio]:
        """
        各チャネルの音をモノラルとしてオーディオファイルを生成します

        Returns:
            tuple[Audio, Audio]: 左チャネルのオーディオファイル, 右チャネルのオーディオファイル
        """
        return Audio(self.left_channel, self.fs), Audio(self.right_channel, self.fs)

    def normalize(self) -> Self:
        """
        各チャネルの音データをmin-max正規化し, 振幅を[-1, 1]にします.

        Returns:
            StereoAudio: 正規化した音データのオーディオファイル
        """
        return StereoAudio(
            self.left_channel.normalize(), self.right_channel.normalize()
        )

    def save(
        self,
        path: str,
        target_bits: int = 16,
        normalize: bool = True,
        overwrite: bool = False,
    ) -> Self:
        """
        オーディオファイルに書き込みます.
        `overwrite`が`False`の場合かつすでにファイルが存在する場合上書きされません.
        対応しているフォーマットは`soundfile.write`を確認してください.

        Args:
            path (str): 保存先のパス
            target_bits (int, optional): 書き込むbit数
            normalize (bool, optional): 書き込む前に上書きするかどうか
            overwrite (bool, optional): 上書きを許可するかどうか

        Returns:
            Audio: 自身のインスタンス
        """
        if not overwrite and os.path.exists(path):
            print(path, "は既に存在します")
            return self

        left_channel = self.normalize().left_channel if normalize else self.left_channel
        right_channel = (
            self.normalize().right_channel if normalize else self.right_channel
        )

        Path(path).parent.mkdir(exist_ok=True)

        soundfile.write(
            path,
            np.vstack((left_channel.data, right_channel.data)).T,
            self.fs,
            "PCM_{}".format(target_bits),
        )

        return self

    def __str__(self) -> str:
        return "length: {}\nsampling frequency: {}".format(self.__len__(), self.fs)

    def __len__(self) -> int:
        return self.left_channel.__len__()


class CannotCalcError(Exception):
    pass


class FrameSeries:
    """
    フレーム単位の情報を扱うクラスです.
    継承して新しいプロパティを追加する場合, `copy_with()`を適切にオーバーライドしてください.
    """

    def __init__(
        self,
        frame_series: np.ndarray,
        dtype: Optional[type] = None,
    ) -> None:
        """
        Args:
            frame_series (np.ndarray): フレーム単位の系列(2次元を想定)
            dtype (type): フレームのデータタイプ. `None` の場合 `frame_series` のデータタイプになります

        Raises:
            ValueError: フレームの系列が2次元でない場合
        """
        if len(frame_series.shape) != 2:
            raise ValueError("フレームの系列は2次元でなければなりません.")

        self.__frame_series = (
            frame_series.astype(dtype=dtype) if dtype is not None else frame_series
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """
        フレームの系列の形状を返します.

        Returns:
            tuple: フレームの系列の形状
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

    @property
    def dtype(self) -> type:
        """
        フレームの系列のデータタイプを返します.

        Returns:
            type: データタイプ
        """
        return self.__frame_series.dtype

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
            (
                (0, (frames - (self.frame_series.shape[0] % frames)) % frames),
                (0, 0),
            ),
        )

        return padded.reshape(padded.shape[0] // frames, frames, padded.shape[1])

    def map(self, func: Callable[[np.ndarray], np.ndarray], axis: int = 0) -> Self:
        """
        軸`axis`に従って関数`func`を適用します.

        Args:
            func (Callable[[np.ndarray], np.ndarray]): 適用する関数
            axis (int, optional): 適用する軸

        Returns:
            Self: 関数を適用した後のフレームの系列
        """
        return self.copy_with(
            frame_series=np.apply_along_axis(func, axis=axis, arr=self.frame_series)
        )

    def reduce(
        self,
        func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        initial: Optional[np.ndarray] = None,
        axis: int = 0,
    ) -> np.ndarray:
        """
        軸`axis`に従って集約関数`func`を適用します.

        Args:
            func (Callable[[np.ndarray, np.ndarray], np.ndarray]): 集約関数
            initial (Optional[np.ndarray], optional): 初期状態
            axis (int, optional): 適用する軸

        Returns:
            np.ndarray: 集計結果
        """
        return (
            reduce(func, self.frame_series if axis == 0 else self.frame_series.T)
            if initial is None
            else reduce(
                func, self.frame_series if axis == 0 else self.frame_series.T, initial
            )
        )

    def average_by_axis(self, axis: int = 0) -> np.ndarray:
        """
        軸`axis`に従って平均した系列を返します.

        Args:
            axis (int, optional): 平均する軸. デフォルトは時間軸です.

        Returns:
            np.ndarray: 平均した系列
        """
        return self.reduce(lambda x, y: x + y, axis=axis) / self.shape[axis]

    def global_average(self) -> float:
        """
        フレームの系列全体の平均を返します.

        Returns:
            float: 平均
        """
        return np.mean(self.frame_series)

    def trim_by_value(self, start: int, end: int) -> FrameSeries:
        """
        値軸でフレームの系列を切り取ります.
        指定した終了インデックスの部分は切り取った後のフレームの系列に含まれません.
        この操作は特徴量によっては不整合を起こす可能性があるため, 元のクラスには戻らず全てFrameSeriesになります.

        Args:
            start (int): 開始インデックス
            end (int): 終了インデックス

        Returns:
            FrameSeries: 切り取った後のフレームの系列
        """
        return FrameSeries(frame_series=self.frame_series[:, start:end])

    def trim_by_time(self, start: int, end: int) -> Self:
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

    def pad_by_value(
        self, pad_width: tuple[int, int], mode: str = "constant"
    ) -> FrameSeries:
        """
        値軸でフレームの系列をパディングします.
        この操作は特徴量によっては不整合を起こす可能性があるため, 元の暮らすには戻らず全てFrameSeriesになります.

        Args:
            pad_width (tuple[int, int]): パディングの長さ
            mode (str): パディングの方法 デフォルトでは0埋めされます

        Returns:
            FrameSeries: パディングされたフレームの系列
        """
        return FrameSeries(
            frame_series=np.pad(
                self.frame_series, pad_width=((0, 0), pad_width), mode=mode
            ),
            dtype=self.dtype,
        )

    def pad_by_time(self, pad_width: tuple[int, int], mode: str = "constant") -> Self:
        """
        時間軸でフレームの系列をパディングします.

        Args:
            pad_width (tuple[int, int]): パディングの長さ
            mode (str): パディングの方法 デフォルトでは0埋めされます

        Returns:
            Self: パディングされたフレームの系列
        """
        return self.copy_with(
            frame_series=np.pad(
                self.frame_series, pad_width=(pad_width, (0, 0)), mode=mode
            )
        )

    def concat(self, *others: Self, axis: int = 0) -> Self:
        """
        軸`axis`に従ってフレームの系列を結合します.
        連結できるインスタンスは同じプロパティで生成されたインスタンスです.

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
            frame_series=reduce(
                lambda x, y: np.concatenate([x, y.frame_series], axis=axis),
                others,
                self.frame_series,
            )
        )

    def join(self, *others: Self, axis: int = 0) -> Self:
        """
        軸`axis`に従って自身のインスタンスを他のインスタンスの間にはさんで結合します.

        Example:

        ```python
        >>> a = FrameSeries(np.zeros(2).reshape((2, 1)))
        >>> b = FrameSeries(np.zeros(4).reshape((2, 2)) + 1)
        >>> c = FrameSeries(np.zeros(4).reshape((2, 2)) + 2)
        >>> d = FrameSeries(np.zeros(4).reshape((2, 2)) + 3)
        >>> print(a.join(b, c, d, axis=1).frame_series)
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
            frame_series=reduce(
                lambda x, y: np.concatenate(
                    [x, self.frame_series, y.frame_series], axis=axis
                ),
                others[1:],
                others[0].frame_series,
            )
        )

    def properties(self) -> dict[str, Any]:
        """
        このフレームの系列を生成したプロパティを辞書形式で返します

        Returns:
            dict[str, Any]: プロパティの辞書
        """

        bases = list(
            map(
                lambda cls: "_" + cls.__name__,
                filter(
                    lambda cls: cls is not object,
                    inspect.getmro(self.__class__),
                ),
            )
        )

        return dict(
            map(
                lambda e: (
                    re.sub("(" + "|".join(bases + ["__"]) + ")", "", e[0]),
                    e[1],
                ),
                filter(
                    lambda e: not "__frame_series" in e[0],
                    self.__dict__.items(),
                ),
            ),
        )

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
                path,
                type=self.__class__.__name__,
                frame_series=self.frame_series,
                **self.properties(),
            )
        else:
            np.savez(
                path,
                type=self.__class__.__name__,
                frame_series=self.frame_series,
                **self.properties(),
            )

        return self

    @classmethod
    def from_npz(cls, path: str) -> Self:
        """
        npzファイルからインスタンスを読み込みます.

        Args:
            path (str): npzファイルのパス

        Raises:
            TypeError: 読み込むインスタンスのタイプが一致しない場合

        Returns:
            Self: 読み込んだインスタンス
        """
        npz = np.load(path, allow_pickle=True)
        params = {k: npz[k] for k in npz.files}
        type_name = params.pop("type")

        if type_name != cls.__name__:
            raise TypeError("{} は type:{} で読み込む必要があります.".format(path, type_name))

        return cls(**params)

    def plot(self, color_map: str = "magma") -> None:
        """
        フレーム系列をプロットします.
        このメソッドをオーバーライドすることで`show()`と`save_as_fig()`の出力を任意のプロットに変更できます.
        オーバーライドしない場合, フレームの系列の1要素を1ドットとするカラーマップが表示されます.

        Args:
            color_map (str, optional): プロットに使用するカラーマップのタイプ
        """
        dpi = 100
        plt.figure(dpi=dpi, figsize=(self.shape[0] / dpi, self.shape[1] / dpi))
        plt.axis("off")
        plt.imshow(
            self.frame_series.T, cmap=color_map, interpolation="none", origin="lower"
        )
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

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

    def plot_with(
        self,
        plot_func: Callable[[np.ndarray, tuple[Any, ...], dict[str, Any]], None],
        path: str,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """
        任意のプロット関数を使ってフレームの系列をプロットし, 保存します.
        プロット関数に渡されるのは, 横軸が時間, 縦軸が系列の値の配列です.

        Args:
            plot_func (Callable[[np.ndarray, dict[str, Any]], None]): プロットする関数. 第一引数に系列の配列が渡されます. args, kwargsで任意の引数を渡せます.
            path (str): 保存するパス.

        Returns:
            Self: 自身のインスタンス
        """
        plot_func(self.frame_series.T, *args, **kwargs)
        plt.savefig(path)
        plt.close()
        return self

    def copy_with(
        self,
        frame_series: Optional[np.ndarray] = None,
    ) -> Self:
        """
        引数の値を使って自身のインスタンスをコピーします.

        Args:
            frame_series (Optional[np.ndarray], optional): フレーム単位の系列

        Returns:
            Self: コピーしたインスタンス
        """
        frame_series = self.frame_series if frame_series is None else frame_series

        return self.__class__(frame_series)

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

    def __can_calc(self, other: FrameSeries) -> bool:
        """
        2項演算が行えるかをbool値で返します

        Args:
            other (FrameSeries): 確認するインスタンス

        Returns:
            bool: 2項演算が行えるかどうか
        """

        return isinstance(other, self.__class__) and self.same_property(other)

    def __len__(self) -> int:
        return self.frame_series.__len__()

    def __add__(self, other: Any) -> Self:
        if isinstance(other, FrameSeries):
            if self.__can_calc(other):
                return self.copy_with(
                    frame_series=self.frame_series + other.frame_series
                )
            else:
                raise CannotCalcError("この二つのフレーム系列の演算は行えません.\n", str(self), str(other))

        return self.copy_with(frame_series=self.frame_series + other)

    def __sub__(self, other: Any) -> Self:
        if isinstance(other, FrameSeries):
            if self.__can_calc(other):
                return self.copy_with(
                    frame_series=self.frame_series - other.frame_series
                )
            else:
                raise CannotCalcError("この二つのフレーム系列の演算は行えません.\n", str(self), str(other))

        return self.copy_with(frame_series=self.frame_series - other)

    def __mul__(self, other: Any) -> Self:
        if isinstance(other, FrameSeries):
            if self.__can_calc(other):
                return self.copy_with(
                    frame_series=self.frame_series * other.frame_series
                )
            else:
                raise CannotCalcError("この二つのフレーム系列の演算は行えません.\n", str(self), str(other))

        return self.copy_with(frame_series=self.frame_series * other)

    def __truediv__(self, other: Any) -> Self:
        if isinstance(other, FrameSeries):
            if self.__can_calc(other):
                return self.copy_with(
                    frame_series=self.frame_series / other.frame_series
                )
            else:
                raise CannotCalcError("この二つのフレーム系列の演算は行えません.\n", str(self), str(other))

        return self.copy_with(frame_series=self.frame_series / other)

    def __getitem__(self, key: Any) -> Union[np.ndarray, np.number]:
        return self.frame_series.__getitem__(key)

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.frame_series.__iter__()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            raise TypeError(
                "{} と {} は比較できません.".format(
                    self.__class__.__name__, other.__class__.__name__
                )
            )

        return self.same_property(other) and np.array_equal(
            other.frame_series, self.frame_series
        )

    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            raise TypeError(
                "{} と {} は比較できません.".format(
                    self.__class__.__name__, other.__class__.__name__
                )
            )

        return not self.same_property(other) or not np.array_equal(
            other.frame_series, self.frame_series
        )

    def __lt__(self, other: Any) -> np.ndarray:
        return self.frame_series.__lt__(other)

    def __gt__(self, other: Any) -> np.ndarray:
        return self.frame_series.__gt__(other)

    def __repr__(self) -> str:
        return self.frame_series.__repr__()

    def __str__(self) -> str:
        string = "Feature Summary\n"
        string += "------------------------------------\n"
        string += "type: " + self.__class__.__name__ + "\n"
        string += "data shape: {}\n".format(self.shape)
        string += "data type: {}\n".format(self.dtype)
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
        dtype: Optional[type] = None,
    ) -> None:
        """
        Args:
            frame_series (np.ndarray): フレーム単位の系列(Frames, Time domain feature)
            frame_length (int): フレーム長
            frame_shift (int): シフト長
            fs (int): サンプリング周波数
            dtype (Optional[type], optional): フレームのデータタイプ. `None` の場合 `frame_series` のデータタイプになります
        """
        super().__init__(frame_series, dtype=dtype)
        self.__frame_length = frame_length
        self.__frame_shift = frame_shift
        self.__fs = fs

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

        return self.__class__(frame_series, frame_length, frame_shift, fs)


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
        dtype: Optional[type] = None,
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
            dtype (Optional[type], optional): フレームのデータタイプ. `None` の場合 `frame_series` のデータタイプになります
        """
        super().__init__(frame_series, dtype=dtype)
        self.__frame_length = frame_length
        self.__frame_shift = frame_shift
        self.__fs = fs
        self.__fft_point = fft_point
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
            linear_func = lambda x: np.power(10, x / 10)
        else:
            linear_func = lambda x: np.power(10, x / 20)

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

    # 以下継承したメソッド

    @override
    def plot(
        self,
        up_to_nyquist: bool = True,
        color_map: str = "magma",
    ) -> None:
        """
        周波数領域のフレームの系列を2次元プロットします.

        Args:
            up_to_nyquist (bool, optional): ナイキスト周波数までのプロットにするか
            color_map (str, optional): プロットに使用するカラーマップ
        """
        if up_to_nyquist:
            show_data = self.frame_series[:, : self.shape[1] // 2 + 1]
        else:
            show_data = self.frame_series

        dpi = 100
        plt.figure(
            dpi=dpi, figsize=(show_data.shape[0] / dpi, show_data.shape[1] / dpi)
        )
        plt.axis("off")
        plt.imshow(show_data.T, cmap=color_map, interpolation="none", origin="lower")
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

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

        return self.__class__(
            frame_series, frame_length, frame_shift, fft_point, fs, dB=dB, power=power
        )
