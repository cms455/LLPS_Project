from setuptools import setup, find_packages

setup(
    name='alps',
    version='0.1.0',
    description='A package to explore physical computing using LLPS',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Takumi Matsuzawa',
    author_email='tmatsuzawa@cornell.edu',
    url='https://github.com/tmatsuzawa/Learn-using-LLPS',
    packages=find_packages(),
    install_requires=[
        # List of dependencies,
        'numpy', 'scipy',
        'matplotlib', 'seaborn',
        'tqdm', 'mpltern', 'sympy', 'numba',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)