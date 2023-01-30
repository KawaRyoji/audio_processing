from typing import Optional

import librosa
import numpy as np
import soundfile
from features import Waveform


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

        self.fs = fs
        self.data = np.array(data, dtype=dtype)

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

    def resample(self, target_fs: int) -> "WavFile":
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

    def save(self, path: str, target_bits: int = 16) -> None:
        soundfile.write(path, self.data, self.fs, "PCM_{}".format(target_bits))

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
        return Waveform.create(
            self.data,
            frame_length,
            frame_shift,
            self.fs,
            padding=padding,
            padding_mode=padding_mode,
            dtype=dtype,
        )

    def to_npz(self, path: str, compress: bool = False) -> None:
        """
        自身のインスタンスをnpzファイルに保存します.

        Args:
            path (str): 保存先のパス
            compress (bool, optional): 圧縮するかどうか
        """
        if compress:
            np.savez_compressed(path, x=self.data, fs=self.fs)
        else:
            np.savez(path, x=self.data, fs=self.fs)

    @classmethod
    def from_npz(cls, path: str) -> "WavFile":
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
