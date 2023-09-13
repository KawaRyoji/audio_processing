import setuptools

setuptools.setup(
    name="audio_processing",
    version="1.0.0",
    requires=open("./requirements.txt").read().splitlines(),
    packages=["audio_processing"],
    package_dir={"audio_processing": "."}
)
