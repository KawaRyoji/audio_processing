import setuptools

setuptools.setup(
    name="audio_processing",
    version="1.0.0",
    requires=open("./requirements.txt").read().splitlines(),
    packages=setuptools.find_packages("./audio_processing"),
)
