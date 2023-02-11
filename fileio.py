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
    from audio_processing.features import Waveform


class WavFile:
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
        wavファイルのデータ
        """
        return self.__data

    @classmethod
    def read(
        cls, path: str, fs: Optional[int] = None, dtype: np.dtype = np.float32
    ) -> "WavFile":
        """
        wavファイルからインスタンスを生成します.

        Args:
            path (str): wavファイルパス
            fs (Optional[int], optional): サンプリング周波数. Noneの場合読み込んだwavファイルのサンプリング周波数になります
            dtype (np.dtype, optional): データタイプ

        Returns:
            WavFile: 生成したインスタンス
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
            WavFile: サンプリング周波数を変換したインスタンス
        """
        if self.fs == target_fs:
            return self

        resampled: np.ndarray = librosa.core.resample(
            y=self.data, orig_sr=self.fs, target_sr=target_fs
        )

        return WavFile(resampled, target_fs, dtype=resampled.dtype)

    def save(self, path: str, target_bits: int = 16, overwrite: bool = False) -> Self:
        """
        wavファイルに書き込みます.
        `overwrite`が`False`の場合かつすでにファイルが存在する場合上書きされません.

        Args:
            path (str): 保存先のパス
            target_bits (int, optional): 書き込むbit数
            overwrite (bool, optional): 上書きを許可するかどうか

        Returns:
            WavFile: 自身のインスタンス
        """
        if not overwrite and os.path.exists(path):
            print(path, "は既に存在します")
            return self

        Path(path).parent.mkdir(exist_ok=True)

        soundfile.write(path, self.data, self.fs, "PCM_{}".format(target_bits))

        return self

    def plot(self) -> None:
        x = np.arange(self.data.shape[0]) / self.fs
        plt.plot(x, self.data)
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Amplitude", fontsize=16)
        plt.grid(True)
        plt.tight_layout()

    def show_fig(self) -> Self:
        self.plot()
        plt.show()
        return self

    def save_as_fig(self, path: str) -> Self:
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
    ) -> Waveform:
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
        from audio_processing.features import Waveform

        return Waveform.create(
            self.data,
            frame_length,
            frame_shift,
            self.fs,
            padding=padding,
            padding_mode=padding_mode,
            dtype=dtype,
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
            WavFile: 自身のインスタンス
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
    def from_npz(cls, path: str) -> WavFile:
        """
        npzファイルからインスタンスを生成します.

        Args:
            path (str): npzファイルのパス

        Returns:
            WavFile: 生成したインスタンス
        """
        file = np.load(path, allow_pickle=True)
        data: np.ndarray = file["data"]
        fs: int = file["fs"]

        return cls(data, fs, dtype=data.dtype)

    def __str__(self) -> str:
        return "data shape: {}\nsampling frequency: {}".format(self.data.shape, self.fs)
