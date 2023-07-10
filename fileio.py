from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from typing_extensions import Self

if TYPE_CHECKING:
    from .features import ComplexCQT, WaveformFrameSeries


class AudioFile:
    def __init__(self, data: np.ndarray, fs: int, dtype: np.dtype = np.float32) -> None:
        """
        Args:
            data (np.ndarray): 時間波形 (1次元を想定)
            fs (int): サンプリング周波数
            dtype (np.dtype, optional): データタイプ

        Raises:
            ValueError: 入力データが1次元出ない場合
        """
        if len(data.shape) != 1:
            raise ValueError("時間波形は1次元配列でなければなりません")

        self.__fs = fs
        self.__data = np.array(data, dtype=dtype)

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
    def dtype(self) -> np.dtype:
        """
        音データのデータタイプ

        Returns:
            np.dtype: データタイプ
        """
        return self.__data.dtype

    @classmethod
    def read(
        cls, path: str, fs: Optional[int] = None, dtype: np.dtype = np.float32
    ) -> Self:
        """
        オーディオファイルからインスタンスを生成します. 対応しているフォーマットは`librosa.core.load`を確認してください.

        Args:
            path (str): オーディオファイルパス
            fs (Optional[int], optional): サンプリング周波数. Noneの場合読み込んだオーディオファイルのサンプリング周波数になります
            dtype (np.dtype, optional): データタイプ

        Returns:
            AudioFile: 生成したインスタンス
        """
        data, fs = librosa.core.load(path, sr=fs)
        return cls(data, fs, dtype=dtype)

    def resample(self, target_fs: int) -> Self:
        """
        サンプリング周波数を変換します.
        目標のサンプリング周波数が現在のサンプリング周波数と同じ場合は自身のインスタンスを返します.

        Args:
            target_fs (int): 目標のサンプリング周波数

        Returns:
            AudioFile: サンプリング周波数を変換したインスタンス
        """
        if self.fs == target_fs:
            return self

        resampled: np.ndarray = librosa.core.resample(
            y=self.data, orig_sr=self.fs, target_sr=target_fs
        )

        return AudioFile(resampled, target_fs, dtype=resampled.dtype)

    def normalize(self) -> Self:
        """
        音データをmin-max正規化し, 振幅を[0, 1]にします.

        Returns:
            Self: 正規化した音データのオーディオファイル
        """
        d_max = np.max(np.abs(self.data))

        return AudioFile(self.data / d_max, self.fs, dtype=self.dtype)

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
            AudioFile: 自身のインスタンス
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
        dtype: np.dtype = np.float32,
    ) -> WaveformFrameSeries:
        """
        フレーム系列の時間波形に変換します.

        Args:
            frame_length (int): フレーム長
            frame_shift (int): フレームシフト
            padding (bool, optional): パディングするかどうか
            padding_mode (str, optional): パディングの方法. numpyのpad関数に使用します
            dtype (np.dtype, optional): データタイプ

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
            dtype=None,
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
            AudioFile: 自身のインスタンス
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
    def from_npz(cls, path: str) -> AudioFile:
        """
        npzファイルからインスタンスを生成します.

        Args:
            path (str): npzファイルのパス

        Returns:
            AudioFile: 生成したインスタンス
        """
        file = np.load(path, allow_pickle=True)
        data: np.ndarray = file["data"]
        fs: int = file["fs"]

        return cls(data, fs, dtype=data.dtype)

    def __str__(self) -> str:
        return "data shape: {}\nsampling frequency: {}".format(self.data.shape, self.fs)
