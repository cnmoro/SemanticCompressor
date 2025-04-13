from setuptools import setup, find_packages

setup(
    name='semantic_compressor',
    version='2.0',
    author='Carlo Moro',
    author_email='cnmoro@gmail.com',
    description="Semantic text compression",
    packages=find_packages(),
    package_data={
        "compressor": ["resources/**/*"]
    },
    include_package_data=True,
    install_requires=[
        "numpy<2",
        "nltk",
        "scikit-learn",
        "fasttext",
        "model2vec",
        "pyspellchecker"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)