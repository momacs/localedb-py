import setuptools

requires = ['folium', 'matplotlib', 'numpy', 'pandas', 'psycopg2-binary', 'pywt', 'scipy', 'scikit-learn', 'scikit-learn-extra', 'tslearn']


setuptools.setup(
    name='localedb',
    version='0.0.2',
    author='Tomek D. Loboda',
    author_email='tomek.loboda@gmail.com',
    description='Python interface to the LocaleDB',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/momacs/localedb_py',
    keywords=['binding', 'db', 'dbi', 'interface', 'postgres', 'postgresql'],
    packages=['localedb'],
    package_dir={'': 'src'},
    python_requires='>=3.6',
    install_requires=requires,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Database :: Front-Ends'
    ],
    license="BSD"
)
