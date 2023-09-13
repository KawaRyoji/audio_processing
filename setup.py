import setuptools

setuptools.setup(
    name="audio_processing",
    version="1.0.0",
    install_requires=open("./requirements.txt").read().splitlines(),
)
