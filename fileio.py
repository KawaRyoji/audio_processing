import librosa
import numpy as np
import soundfile


class WavFile:
    def __init__(self, data: np.ndarray, fs: int, dtype=np.float32) -> None:
        self.fs = fs
        self.data = np.array(data, dtype=dtype)

    @classmethod
    def read(cls, path: str, fs=None, dtype=np.float32) -> "WavFile":
        data, fs = librosa.core.load(path, sr=fs)
        return cls(data, fs, dtype=dtype)

    def resample(self, target_fs: int) -> "WavFile":
        if self.fs == target_fs:
            return self

        resampled: np.ndarray = librosa.core.resample(
            y=self.data, orig_sr=self.fs, target_sr=target_fs
        )

        return WavFile(resampled, target_fs, dtype=resampled.dtype)

    def save(self, path: str, target_bits: int = 16) -> None:
        soundfile.write(path, self.data, self.fs, "PCM_{}".format(target_bits))

    def to_npz(self, path: str, compress: bool) -> None:
        if compress:
            np.savez_compressed(path, x=self.data, fs=self.fs)
        else:
            np.savez(path, x=self.data, fs=self.fs)

    @classmethod
    def from_npz(cls, path: str) -> "WavFile":
        file = np.load(path, allow_pickle=True)
        data: np.ndarray = file["data"]
        fs: int = file["fs"]

        return cls(data, fs, dtype=data.dtype)
