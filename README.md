# 音信号処理ライブラリ

## はじめに

このプログラムは簡単に音信号を処理できるライブラリです。
複雑な処理は目的にしておらず、基本的な音に関する操作を対象としています。

主な機能としては以下のようになっています。

1. **チェインメソッド**により簡潔で直感的なオブジェクト操作
2. 処理した結果を保存/読み込みできる機能
3. 誤った演算を防ぐエラー機能

何かしらバグがある場合は、報告してもらえると幸いです。

## 使用例

### チェインメソッド

#### wavファイルのデータの読み込みからdB値の振幅スペクトルを得る

```python
from audio_processing import AudioFile

amp_spectrum_dB = (
    AudioFile.read("path/to/wav/file.wav") # wavファイルを読み込む
    .to_frames(frame_length=1024, frame_shift=256) # 時間波形をフレーム系列に変換する
    .to_spectrum() # スペクトルに変換
    .to_amplitude() # 振幅スペクトルに変換
    .linear_to_dB() # 振幅スペクトルをdB値に変換
)
```

途中にdump()またはplot()を呼ぶことで、その時点におけるオブジェクトの状態を確認できます。

```python
amp_spectrum_dB = (
    AudioFile.read("path/to/audio/file.wav")
    .to_frames(frame_length=1024, frame_shift=256)
    .to_spectrum()
    .dump()
    .to_amplitude()
    .linear_to_dB()
    .dump()
)
```

出力1

```
Feature Summary
------------------------------------
type: Spectrum
data shape: (2000, 1024)
data type: complex128
frame_length: 1024
frame_shift: 256
fs: 16000
fft_point: 1024
dB: False
power: False
------------------------------------
```

出力2

```
Feature Summary
------------------------------------
type: AmplitudeSpectrum
data shape: (2000, 1024)
data type: float64
frame_length: 1024
frame_shift: 256
fs: 16000
fft_point: 1024
dB: True
power: False
------------------------------------
```

### データの永続化

#### 処理した結果をnpzファイルに保存する

```python
amp_spectrum_dB.save("path/to/npz/file.npz")
```

saveメソッドもチェインメソッドで途中の状態を保存できます。
なお、すでに指定したパスが存在する場合、保存されません。
上書きを許可する場合は以下のようにしてください。

```python
amp_spectrum_dB.save("path/to/npz/file.npz", overwrite=True)
```

#### npzファイルからオブジェクトを復元する

```python
amp_spectrum_dB = AmplitudeSpectrum.from_npz("path/to/npz/file.npz")
```

### エラー機能

#### 振幅スペクトルの四則演算

```python
amp_spectrum_add = amp_spectrum1 + amp_spectrum2
amp_spectrum_mul = amp_spectrum1 * amp_spectrum2
amp_spectrum_sub = amp_spectrum1 - amp_spectrum2
amp_spectrum_div = amp_spectrum1 / amp_spectrum2
```

以下のように、異なるプロパティのクラス同士は演算できません。

```python
amp_spectrum_linear = (
    WavFile.read("path/to/wav/file.wav")
    .to_frames(frame_length=1024, frame_shift=256)
    .to_spectrum()
    .to_amplitude()
)
amp_spectrum_dB = amp_spectrum_linear.linear_to_dB()
amp_spectrum_error = amp_spectrum_dB * amp_spectrum_linear # Error!
```