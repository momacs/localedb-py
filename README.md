# localedb-py

Python bindings for [LocaleDB](https://github.com/momacs/localedb).


## Dependencies

- [Python 3](https://www.python.org)
- [numpy](https://numpy.org)
- [psycopg2](https://pypi.org/project/psycopg2)
- [LocaleDB](https://github.com/momacs/localedb)


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

conf = db.get_dis_dyn_by_day_conf(20,77)  # get the time series of confirmed COVID-19 cases from day 20 to day 77
print(conf.flatten().tolist())            # print the time series
```

The last instruction should result in the following being printed:

```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 3, 5, 5, 5, 7, 16, 17, 21, 30, 35, 45, 59, 61, 63, 66, 68, 75, 82, 86, 89, 99, 105]
```


## License

This project is licensed under the [BSD License](LICENSE.md).
