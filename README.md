# localedb-py

Python bindings for [LocaleDB](https://github.com/momacs/localedb).


## Dependencies

- [LocaleDB](https://github.com/momacs/localedb)
- [Python 3](https://www.python.org)
- [folium](https://github.com/python-visualization/folium)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org)
- [pandas](https://pandas.pydata.org/)
- [psycopg2](https://pypi.org/project/psycopg2)
- [pywt](https://github.com/PyWavelets/pywt)
- [scipy](https://www.scipy.org/)
- [scikit-learn](https://scikit-learn.org)
- [scikit-learn-extra](https://github.com/scikit-learn-contrib/scikit-learn-extra)
- [tslearn](https://github.com/tslearn-team/tslearn/)


## Setup

```
pip install git+https://github.com/momacs/localedb-py.git
```

Remember to activate your `venv` of choice unless you want to go system-level.


## Usage

```python
from localedb import LocaleDB
db = LocaleDB()

print(db.get_rand_us_state_fips(3))   # select three random U.S. state  FIPS codes
print(db.get_rand_us_county_fips(3))  # select three random U.S. county FIPS codes

db.set_disease('COVID-19')      # set COVID-19 as the current disease
db.set_locale_by_name('Italy')  # set Italy as the current locale

db.set_locale_by_name('US', 'Alaska')  # set the state of Alaska as the current locale
db.set_locale_by_us_fips('02')         # same

db.set_locale_by_name('US', 'Alaska', 'Anchorage')  # set the county of Anchorage, Alaska as the current locale
db.set_locale_by_us_fips('02020')                   # same

conf = db.get_dis_dyn_by_day_conf(20,77)  # get the time series of confirmed cases from day 20 to day 77
print(conf.flatten().tolist())            # print that time series
```

The last instruction should result in the following being printed:

```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 3, 5, 5, 5, 7, 16, 17, 21, 30, 35, 45, 59, 61, 63, 66, 68, 75, 82, 86, 89, 99, 105]
```


## Notebooks

The following computational notebooks demonstrate the usage and capabilities of the package:
- [Data Access](https://github.com/momacs/localedb-py/notebooks/01-data-access.ipynb)
- [Multivariate Time Series Clustering](https://github.com/momacs/localedb-py/notebooks/02-mvts-cluster.ipynb)
- [Multivariate Time Series Clustering (Continuous Wavelet Transform)](https://github.com/momacs/localedb-py/notebooks/03-mvts-cluster-cwt.ipynb)
- [Multivariate Time Series Clustering (Mapping)](https://github.com/momacs/localedb-py/notebooks/04-mvts-cluster-map.ipynb)
- [Multivariate Time Series Clustering (Distance Matrices](https://github.com/momacs/localedb-py/notebooks/05-mvts-cluster-dist-mat.ipynb)
- [Multivariate Time Series Clustering (Performance Evaluation](https://github.com/momacs/localedb-py/notebooks/06-mvts-cluster-perf-eval.ipynb)


## License

This project is licensed under the [BSD License](LICENSE.md).
