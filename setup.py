import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pyopls',
    version='20.03-1',
    author='BiRG @ Wright State University',
    author_email='foose.3@wright.edu',
    description='Orthogonal Projection to Latent Structures',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/BiRG/pyopls',
    keywords='metabolomics chemometrics partial-least-squares',
    download_url='https://github.com/BiRG/pyopls/archive/20.02.tar.gz',
    packages=setuptools.find_packages(),
    python_requires='>=3.5',
    install_requires=[
        'numpy>=1.11.0',
        'scipy>=0.18.0',
        'scikit-learn>=0.18.0'
    ],
    classifiers=[
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
