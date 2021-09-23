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

The following [computational notebooks](https://github.com/momacs/localedb-py/tree/master/notebooks) demonstrate the usage and capabilities of the package:
- [Data Access](https://github.com/momacs/localedb-py/blob/master/notebooks/01-data-access.ipynb)
- [Multivariate Time Series Clustering](https://github.com/momacs/localedb-py/blob/master/notebooks/02-mvts-cluster.ipynb)
- [Multivariate Time Series Clustering (Continuous Wavelet Transform)](https://github.com/momacs/localedb-py/blob/master/notebooks/03-mvts-cluster-cwt.ipynb)
- [Multivariate Time Series Clustering (Mapping)](https://github.com/momacs/localedb-py/blob/master/notebooks/04-mvts-cluster-map.ipynb)
- [Multivariate Time Series Clustering (Distance Matrices](https://github.com/momacs/localedb-py/blob/master/notebooks/05-mvts-cluster-dist-mat.ipynb)
- [Multivariate Time Series Clustering (Performance Evaluation](https://github.com/momacs/localedb-py/blob/master/notebooks/06-mvts-cluster-perf-eval.ipynb)


## Example: Multivariate Time Series Clustering of Covid-19 Disease Dynamics

Below is a brief narrative based on a few results we discuss in the [notebooks](https://github.com/momacs/localedb-py/tree/master/notebooks).

#### 1. We cluster the 67 counties of the state of Pennsylvania by Covid-19 disease dynamics (specifically, the number of confirmed cases and the number of deaths, smoothed and standardized)

![Alt text](media/example-mvts-cluster/01-c19-mvts-cluster-pa.png?raw=true)

#### 2. We keep the same cluster assignments, but plot differenced dynamics (also smoothed)

![Alt text](media/example-mvts-cluster/02-c19-mvts-cluster-diff-pa.png?raw=true)

#### 3. We map the clusters

![Alt text](media/example-mvts-cluster/03-c19-mvts-cluster-pa-map.png?raw=true)

#### 4. We compute the Continuous Wavelet Transform (CWT) of the standardized number of confirmed Covid-19 cases in the Allegheny County, PA

![Alt text](media/example-mvts-cluster/04-c19-cwt.png?raw=true)

#### 5. We visualize distance matrices computed using five different distance measures; the distances are among multivariate disease dynamics in counties of five states

![Alt text](media/example-mvts-cluster/05-c19-dist-mat-5-states.png?raw=true)

#### 6. We plot histograms of those distances

![Alt text](media/example-mvts-cluster/06-c19-dist-mat-hist-5-states.png?raw=true)

#### 7. We evaluate clustering performance across several distance measures, clustering algorithms, and datasets using the adjusted Rand index (ARI)

![Alt text](media/example-mvts-cluster/07-cluster-perf-eval.png?raw=true)


## License

This project is licensed under the [BSD License](LICENSE.md).
