import sys
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

NAME = 'megspikes'
MAINTAINER = 'Valerii Chirkov'
MAINTAINER_EMAIL = 'valerii.chirkov@hu-berlin.de'
DESCRIPTION = 'Pipelines for detection epileptic spikes in MEG recording.'
URL = 'https://github.com/MEG-SPIKES/megspikes'
AUTHOR = 'Valerii Chirkov'
AUTHOR_EMAIL = 'valerii.chirkov@hu-berlin.de'
PACKAGES = find_packages()
VERSION = '0.1.5'
CLASSIFIERS = ['Programming Language :: Python :: 3',
               'Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: MIT License',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: OS Independent']
LICENSE = 'MIT'
PYTHON_REQUIRES = '>=3.7'
PLATFORMS = 'any'

with open('requirements.txt') as f:
    REQUIRES = f.read().splitlines()
REQUIRES = [f"{i}" for i in REQUIRES]


# Give setuptools a hint to complain if it's too old a version
# 24.2.0 added the python_requires option
# Should match pyproject.toml
SETUP_REQUIRES = ['setuptools >= 24.2.0']
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ['wheel'] if 'bdist_wheel' in sys.argv else []
# for AlphaCSC installation
SETUP_REQUIRES += ['numpy', 'Cython']

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            long_description_content_type="text/markdown",
            url=URL,
            download_url=URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            include_package_data=True,
            install_requires=REQUIRES,
            python_requires=PYTHON_REQUIRES,
            setup_requires=SETUP_REQUIRES)


if __name__ == '__main__':
    setup(**opts)
