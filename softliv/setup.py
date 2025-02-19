from setuptools import setup, find_packages

setup(
    name='softliv',
    version='0.1.0',
    description='A Python library for the softliv projects',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Takumi Matasuzawa, Teagan Bate',
    author_email='tmatsuzawa@cornell.edu',
    url='https://github.coecis.cornell.edu/tm688/softliv',
    packages=find_packages(),
    install_requires=[
        # List of dependencies,
        'numpy', 'scipy',
        'matplotlib', 'pandas', 'seaborn', 'scikit-learn', 'scikit-image', 'opencv-python',
        'tqdm', 'h5py',
        'requests', 'Bio==1.6.2', 'biopython==1.83',

    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)