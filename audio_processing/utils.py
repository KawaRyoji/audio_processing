from typing import Tuple
import numpy as np


def center_frame_samples(
    num_frames: int, frame_length: int, frame_shift: int, offset: int = 0
) -> np.ndarray:
    return np.fromiter(
        map(
            lambda i: (2 * i * frame_shift + frame_length) // 2 + offset,
            range(num_frames),
        ),
        int,
    )


def edge_point(index: int, frame_length: int, frame_shift: int) -> Tuple[int, int]:
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


def to_symmetry(series: np.ndarray) -> np.ndarray:
    """
    (time, fft_point // 2 + 1) または (time, fft_point // 2)の非対称スペクトログラムを対称にします.

    Args:
        series (np.ndarray): 非対称スペクトログラム

    Returns:
        np.ndarray: 対称スペクトログラム (time, fft_point)
    """
    return (
        np.concatenate([series, np.fliplr(series)], axis=1)
        if series.shape[1] % 2 == 0
        else np.concatenate([series, np.fliplr(series)[:, 1:-1]], axis=1)
    )
