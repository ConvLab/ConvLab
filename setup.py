from setuptools import setup, find_packages
from os import path


rootwd = path.abspath(path.dirname(__file__))

with open(path.join(rootwd, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(rootwd, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [l.strip() for l in f.readlines() if l.strip()]

setup(
    name='convlab',
    version='0.1.1',
    description='An open-source multi-domain end-to-end dialog system platform',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ConvLab/ConvLab',
    author='Microsoft Research - NLP & Dialog Systems Group',
    author_email='jinchao.li@microsoft.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='conversational-ai dialog dialogue systems',
    packages=find_packages(exclude=['docs', 'tutorial']),
    python_requires='>=3.6.5',
    install_requires=requirements,
    extras_require={
        'dev': [],
        'test': [],
    },
)
