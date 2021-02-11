import setuptools

requires = ['numpy', 'psycopg2-binary']

setuptools.setup(
    name='localedb',
    version='0.0.1',
    author='Tomek D. Loboda',
    author_email='tomek.loboda@gmail.com',
    description='Python interface to the LocaleDB',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/momacs/localedb_py',
    keywords=['db', 'dbi', 'interface', 'postgres', 'postgresql'],
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
