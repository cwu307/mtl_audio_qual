#!/usr/bin/env python

from setuptools import setup

setup(name="smaq_cli",
      description="scripts accompanying the paper A Multitask Teacher-Student Framework for Perceptual Audio Quality Assessment",
      author="Chih-Wei Wu",
      author_email="cwu307@gmail.com",
      url="https://github.com/cwu307/mtl_audio_qual",
      packages=["smaq_cli"],
      classifiers=[
          'Programming Language :: Python :: 3.7',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: MacOS',
          'Operating System :: Unix'
      ],
      install_requires=[
          "click",
          "runez",
          "librosa==0.7.2",
          "numpy<1.19.0",
          "scipy==1.4.1",
          "numba==0.48.0",
          "pyACA",
          "PySoundFile",
          "scikit-learn==0.20.3",
          "tensorflow",
          "tensorflow-cpu",
      ],
      package_data={'': ['data/*.save', 'data/*.h5', 'data/*.wav']},
      include_package_data=True,
      entry_points={
        'console_scripts': [
            'smaq-cli = smaq_cli.cli:main',
        ],
      },
     )