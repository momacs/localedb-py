# -*- coding: utf-8 -*-
"""Functionality built on top of LocaleDB data."""

import folium
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import psycopg2
import psycopg2.extras
import pywt
import scipy
import scipy.signal
import sklearn
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics
import sklearn.preprocessing
import sklearn_extra.cluster
import sqlite3
import sys
import time
import tslearn
import tslearn.clustering
import tslearn.metrics
import tslearn.preprocessing

from collections.abc import Iterable
from numpy import linalg

from .util import plot_init


__all__ = ['LocaleDB']


# ----------------------------------------------------------------------------------------------------------------------
class UnknownLocaleError(Exception): pass
class UnknownDiseaseError(Exception): pass
class ObjectStateError(Exception): pass


# ----------------------------------------------------------------------------------------------------------------------
# class Result(object):
#     def __init__(self, res=None, ok=True, err_msg=None):
#         self.ok = ok
#         self.err_msg = err_msg
#         self.res = res


# ----------------------------------------------------------------------------------------------------------------------
class LocaleDB(object):
    """LocaleDB interface.

    Objects of this class grant access to the contents of a LocaleDB instance.
    """

    CURSOR_NAME_PREFIX = 'localedb-py'

    def __init__(self, pg_host='localhost', pg_port='5433', pg_usr='postgres', pg_pwd='sa', pg_db='localedb', do_connect=True):
        self.pg_host = pg_host
        self.pg_port = pg_port
        self.pg_usr  = pg_usr
        self.pg_pwd  = pg_pwd
        self.pg_db   = pg_db

        self.locale_id = None
        self.disease_id = None
        self.locale_fips = None  # not None for the US only (per the main.locale table)

        self.conn = None
        if do_connect:
            self.connect()

        self.cursor_num = -1

    def __del__(self):
        self.disconnect()

    def _exec(self, qry, vars=None, do_get=True, itersize=2000):
        with self.conn.cursor() as c:
            if itersize > 0:
                c.itersize = itersize
            c.execute(qry, vars)
            if do_get:
                return c.fetchall()

    def _get_dis_dyn_by_day_x(self, x, day_from=1, day_to=sys.maxsize, itersize=2000):
        self._req_disease() and self._req_locale()
        if day_from > day_to:
            raise ValueError('Incorrect day range')
        res = {}
        return np.array(
            self._exec(
                f'SELECT {x} FROM dis.dyn WHERE disease_id = %s AND locale_id = %s AND day_i BETWEEN %s AND %s ORDER BY day_i;',
                [self.disease_id, self.locale_id, day_from, day_to],
                itersize
            )
        )

    def _get_dis_dyn_norm(self, conf, dead, do_inc_delta=False):
        self._req_disease() and self._req_locale()
        # res = self.get_dis_dyn_delta(conf, dead)
        # if not res.ok:
        #     return res
        # delta = res.res
        delta = self.get_dis_dyn_delta_by_day(conf, dead)

        # norm1 = np.sum(arr1 ** 2)
        # norm2 = np.sum(arr2 ** 2)
        # norm = np.sum((arr1 - arr2) ** 2)

        return {
            'conf': linalg.norm(delta['conf']),
            'dead': linalg.norm(delta['dead']),
            'delta': None if not do_inc_delta else delta
        }

    def _get_dis_dyn_comp_stats_x(self, x, vals, day_from=1, day_to=sys.maxsize, itersize=2000):
        Y_obs = np.array(self._get_dis_dyn_by_day_x(x, day_from, day_to, itersize)).flatten()
        Y_hat = np.array(vals)
        if Y_obs.size != Y_hat.size:
            raise ValueError(f'The sizes of the observed ({Y_obs.size}) and predicted ({Y_hat.size}) time series do not match.')

        # Corr:
        corr = np.corrcoef(Y_obs, Y_hat)[0,1]
        if np.isnan(corr):
            corr = 0.0

        # MAE:
        mae = np.absolute(Y_obs - Y_hat).mean()

        # RMSE:
        rmse = np.linalg.norm(Y_obs - Y_hat) / np.sqrt(len(Y_obs))

        # SRMSE:
        ybar = Y_obs.mean()
        srmse = rmse / ybar

        # R2:
        u = np.sum((Y_hat - Y_obs)**2)
        v = np.sum((Y_obs - ybar)**2)
        r2 = 1.0 - u / v

        return { 'corr': corr, 'mae': mae, 'rmse': rmse, 'srmse': srmse, 'r2': r2 }

    @staticmethod
    def _get_nan_idx(x):
        return np.isnan(x), lambda z: z.nonzero()[0]

    def _get_next_cursor_name(self):
        self.cursor_num += 1
        return f'{self.CURSOR_NAME_PREFIX}-{self.cursor_num}'

    def _get_id(self, tbl, col='rowid', where_sql=None, where_vars=None):
        return self._get_num(conn, tbl, 'rowid', where_sql, where_vars)

    def _get_row_cnt(self, tbl, where_sql=None, where_vars=None):
        return self._get_num(tbl, 'COUNT(*)', where_sql, where_vars)

    def _get_new_conn(self, cursor_factory=psycopg2.extras.NamedTupleCursor):
        return psycopg2.connect(host=self.pg_host, port=self.pg_port, user=self.pg_usr, password=self.pg_pwd, database=self.pg_db, cursor_factory=cursor_factory)

    def _get_num(self, tbl, col, where_sql=None, where_vars=None):
        where_sql = '' if where_sql is None else f' WHERE {where_sql}'
        with self.conn.cursor() as c:
            c.execute(f'SELECT {col} FROM {tbl}{where_sql};', where_vars)
            row = c.fetchone()
            return row[0] if row else None

    def _req_disease(self):
        """Ensure the current diease has been set."""

        if self.disease_id is None:
            raise ObjectStateError('No disease has been set')

    def _req_locale(self, do_req_us=False):
        """Ensure the current locale has been set."""

        if self.locale_id is None:
            raise ObjectStateError('No locale has been set')
        if do_req_us and self.locale_iso_num != 840:
            raise ObjectStateError('A U.S. locale is required')

    def _set_pop_view_household(self, fips):
        self._exec(f"""
            DROP VIEW IF EXISTS pop_person_view;
            CREATE OR REPLACE TEMP VIEW pop_person_view AS
            SELECT p.*
            FROM pop.person AS p
            INNER JOIN pop.household AS h ON p.household_id = h.id
            WHERE h.stcotrbg LIKE '{fips}%';
        """)

    def _set_pop_view_household_geo(self, fips, geo_tbl):
        return
        self._exec(f"""
            DROP VIEW IF EXISTS pop_person_view;
            CREATE OR REPLACE TEMP VIEW pop_person_view AS
            SELECT p.*, g.gid AS household_geo_id
            FROM pop.person AS p
            INNER JOIN pop.household AS h ON p.household_id = h.id
            INNER JOIN geo.{geo_tbl} AS g ON ST_Contains(g.geom, h.coords)
            WHERE h.stcotrbg LIKE '{fips}%';
        """)

    @staticmethod
    def _smooth(x, window_len=11, window='hanning'):
        if x.ndim != 1:
            raise ValueError('One-dimensional array expected.')
        if x.size < window_len:
            raise ValueError('Input vector needs to be bigger than window size.')
        if window_len < 3:
            return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError('Unknown window type: flat, hanning, hamming, bartlett, blackman')
        s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval(f'np.{window}({window_len})')
        y = np.convolve(w / w.sum(), s, mode='same')
        return y[window_len:-window_len + 1]

    def clear_disease(self):
        """Clears the current disease.

        Returns:
            LocaleDB: self
        """

        self.disease_id = None
        return self

    def clear_locale(self):
        """Clears the current locale.

        Returns:
            LocaleDB: self
        """

        self.locale_id = None
        self.locale_fips = None
        return self

    def connect(self):
        """
        Returns:
            LocaleDB: self
        """

        self.disconnect()
        self.conn = self._get_new_conn()
        return self

    def disconnect(self):
        """
        Returns:
            LocaleDB: self
        """

        if self.conn is not None:
            self.conn.close()
            self.conn = None
        return self

    def get_dis_dyn_by_fips(self, fips='42___', cols='day day_i n_conf n_dead n_rec case_density r0 r0_ci90 test_n_pos test_n_neg test_r_pos beds_hosp_cap beds_hosp_usage_tot beds_hosp_usage_covid beds_icu_cap beds_icu_usage_tot beds_icu_usage_covid vax_n_init vax_n_done vax_r_init vax_r_done'.split(' '), day0=1, day1=99999999, do_interpolate=False, n_diff=0, smooth_window_len=0, smooth_window='hanning', do_scale=False, get_pandas=False):
        """Get disease dynamics for a specified locale or a set of locales given by the US FIPS code SQL wildcard."""

        ret = []

        ids = [r[0] for r in self._exec('SELECT id FROM main.locale WHERE fips LIKE %s ORDER BY fips ASC', [fips])]

        for id in ids:
            ret.append(self.get_dis_dyn_by_id(id, cols, day0, day1, do_interpolate, n_diff, smooth_window_len, smooth_window, do_scale))

        if get_pandas:
            ret = pd.DataFrame(ret)
            ret = ret.T
            ret.columns = ids
        else:
            ret = np.array(ret, dtype=float)

        return (ret, ids)

    def get_dis_dyn_by_id(self, id, cols='day day_i n_conf n_dead n_rec case_density r0 r0_ci90 test_n_pos test_n_neg test_r_pos beds_hosp_cap beds_hosp_usage_tot beds_hosp_usage_covid beds_icu_cap beds_icu_usage_tot beds_icu_usage_covid vax_n_init vax_n_done vax_r_init vax_r_done'.split(' '), day0=1, day1=99999999, do_interpolate=False, n_diff=0, smooth_window_len=0, smooth_window='hanning', do_scale=False, get_pandas=False):
        """Get disease dynamics."""

        self._req_disease()

        ret = pd.read_sql_query(f'SELECT {",".join(cols)} FROM dis.dyn WHERE disease_id = %s AND locale_id = %s AND day_i BETWEEN %s AND %s', self.conn, params=[self.disease_id, id, day0, day1])
        ret = ret.transpose().to_numpy(dtype=np.float)

        if do_interpolate:
            for i in range(len(cols)):
                if np.isnan(ret[i,]).all():
                    continue
                nans, ret_ = self.__class__._get_nan_idx(ret[i,])
                ret[i,nans] = np.interp(ret_(nans), ret_(~nans), ret[i,~nans])

        if n_diff > 0:
            ret_ = np.empty((ret.shape[0], ret.shape[1] - n_diff))
            for i in range(len(cols)):
                ret_[i,] = np.diff(ret[i,], n_diff)
            ret = ret_

        if smooth_window_len > 0:
            for i in range(ret.shape[0]):
                 ret[i] = self.__class__._smooth(ret[i], smooth_window_len, smooth_window)

        if do_scale:
            # ret = sklearn.preprocessing.minmax_scale(ret, axis=1)

            ret = np.transpose(ret, (1,0))
            # ret = sklearn.preprocessing.RobustScaler().fit_transform(ret)
            ret = sklearn.preprocessing.StandardScaler().fit_transform(ret)
            ret = np.transpose(ret, (1,0))

            # for i in range(ret.shape[0]):
            #     ret[i] = sklearn.preprocessing.RobustScaler().fit_transform(ret[i])

        if get_pandas:
            ret = pd.DataFrame(ret, columns=cols)

        return ret

    def get_dis_dyn_by_day_conf(self, day_from=1, day_to=sys.maxsize, itersize=2000):
        """Get number of confirmed cases by day."""

        return self._get_dis_dyn_by_day_x('n_conf', day_from, day_to, itersize)

    def get_dis_dyn_by_day_dead(self, day_from=1, day_to=sys.maxsize, itersize=2000):
        """Get number of deaths by day."""

        return self._get_dis_dyn_by_day_x('n_dead', day_from, day_to, itersize)

    def get_dis_dyn_by_date(self, date_from='2020.01.01', date_to='3000.01.01', itersize=2000):
        self._req_disease() and self._req_locale()
        return self._exec(
            'SELECT day, day_i, n_conf, n_dead, n_rec FROM dis.dyn WHERE locale_id = %s day >= %s AND day <= %s ORDER BY day_i;',
            [self.locale_id, date_from, date_to],
            itersize
        )

    def get_dis_dyn_by_day(self, do_get_conf=False, do_get_dead=False, day_from=1, day_to=sys.maxsize, itersize=2000):
        """Get disease dynamics by day."""

        self._req_disease() and self._req_locale()
        if day_from > day_to:
            raise ValueError('Incorrect day range')
        res = {}
        if do_get_conf:
            res['conf'] = np.array(
                self._exec(
                    'SELECT n_conf FROM dis.dyn WHERE disease_id = %s AND locale_id = %s AND day_i BETWEEN %s AND %s ORDER BY day_i;',
                    [self.disease_id, self.locale_id, day_from, day_to],
                    itersize
                )
            )
        if do_get_dead:
            res['dead'] = np.array(
                self._exec(
                    'SELECT n_dead FROM dis.dyn WHERE disease_id = %s AND locale_id = %s AND day_i BETWEEN %s AND %s ORDER BY day_i;',
                    [self.disease_id, self.locale_id, day_from, day_to],
                    itersize
                )
            )
        return res

    def get_dis_dyn_comp_stats(self, conf, dead, day_from=1, day_to=sys.maxsize, itersize=2000):
        return {
            'conf': self.get_dis_dyn_comp_stats_conf(conf, day_from, day_to, itersize),
            'dead': self.get_dis_dyn_comp_stats_dead(dead, day_from, day_to, itersize)
        }

    def get_dis_dyn_comp_stats_conf(self, vals, day_from=1, day_to=sys.maxsize, itersize=2000):
        return self._get_dis_dyn_comp_stats_x('n_conf', vals, day_from, day_to, itersize)

    def get_dis_dyn_comp_stats_dead(self, vals, day_from=1, day_to=sys.maxsize, itersize=2000):
        return self._get_dis_dyn_comp_stats_x('n_dead', vals, day_from, day_to, itersize)

    def get_dis_dyn_delta_by_day(self, conf=None, dead=None, day_from=1, day_to=sys.maxsize, itersize=2000):
        """Get once differenced disease dynamics cases by day."""

        self._req_disease() and self._req_locale()
        if day_from > day_to:
            raise ValueError('Incorrect day range')
        res = {}
        if conf:
            conf_obs = np.array(
                self._exec(
                    'SELECT n_conf FROM dis.dyn WHERE disease_id = %s AND locale_id = %s AND day_i BETWEEN %s AND %s ORDER BY day_i;',
                    [self.disease_id, self.locale_id, day_from, day_to],
                    itersize
                )
            )
            if len(conf_obs) != len(conf):
                raise ValueError('The sizes of the confirmed cases time series provided is incongruent with the observed one; the database may not contain enough data or the date range is incorrect.')
            res['conf'] = conf_obs - np.ndarray(conf)
        if dead:
            dead_obs = np.array(
                self._exec(
                    'SELECT n_dead FROM dis.dyn WHERE disease_id = %s AND locale_id = %s AND day_i BETWEEN %s AND %s ORDER BY day_i;',
                    [self.disease_id, self.locale_id, day_from, day_to],
                    itersize
                )
            )
            if len(dead_obs) != len(dead):
                raise ValueError('The sizes of the dead cases time series provided is incongruent with the observed one; the database may not contain enough data or the date range is incorrect.')
            res['dead'] = dead_obs - np.ndarray(dead)
        return res

    def get_health_stats_by_fips(self, fips='42___', get_pandas=False):
        """Returns health statistics for a locale or a set of locales."""

        ret = self._exec(
            '''
            WITH locale AS (SELECT id FROM main.locale WHERE fips LIKE %s)
            SELECT
                l.id,
                h.premat_death, h.unins_adults, h.pcp, h.prev_hosp_stays, h.adult_obesity, h.unemp_rate, h.child_in_pov, h.sex_trans_inf, h.mamm_screen, h.phys_inact, h.unins, h.dentists, h.unins_child, h.alcohol_driving_deaths, h.flu_vax,
                w.tavg_m1, w.tavg_m2, w.tavg_m3, w.tavg_m4, w.tavg_m5, w.tavg_m6, w.tavg_m7, w.tavg_m8, w.tavg_m9, w.tavg_m10, w.tavg_m11, w.tavg_m12
            FROM locale l LEFT JOIN
            (
                SELECT locale_id,
                    MAX(rawvalue) FILTER (WHERE measure_id =   1) AS premat_death,
                    MAX(rawvalue) FILTER (WHERE measure_id =   3) AS unins_adults,
                    MAX(rawvalue) FILTER (WHERE measure_id =   4) AS pcp,
                    MAX(rawvalue) FILTER (WHERE measure_id =   5) AS prev_hosp_stays,
                    MAX(rawvalue) FILTER (WHERE measure_id =  11) AS adult_obesity,
                    MAX(rawvalue) FILTER (WHERE measure_id =  23) AS unemp_rate,
                    MAX(rawvalue) FILTER (WHERE measure_id =  24) AS child_in_pov,
                    --MAX(rawvalue) FILTER (WHERE measure_id =  43) AS violent_crime_rate,
                    MAX(rawvalue) FILTER (WHERE measure_id =  45) AS sex_trans_inf,
                    MAX(rawvalue) FILTER (WHERE measure_id =  50) AS mamm_screen,
                    MAX(rawvalue) FILTER (WHERE measure_id =  70) AS phys_inact,
                    MAX(rawvalue) FILTER (WHERE measure_id =  85) AS unins,
                    MAX(rawvalue) FILTER (WHERE measure_id =  88) AS dentists,
                    MAX(rawvalue) FILTER (WHERE measure_id = 122) AS unins_child,
                    --MAX(rawvalue) FILTER (WHERE measure_id = 125) AS air_poll,
                    MAX(rawvalue) FILTER (WHERE measure_id = 134) AS alcohol_driving_deaths,
                    MAX(rawvalue) FILTER (WHERE measure_id = 155) AS flu_vax
                FROM health.health
                WHERE end_year = 2017
                GROUP BY locale_id
            ) h ON h.locale_id = l.id LEFT JOIN
            (
                SELECT locale_id,
                    MAX(tavg) FILTER (WHERE month =  1) AS tavg_m1,
                    MAX(tavg) FILTER (WHERE month =  2) AS tavg_m2,
                    MAX(tavg) FILTER (WHERE month =  3) AS tavg_m3,
                    MAX(tavg) FILTER (WHERE month =  4) AS tavg_m4,
                    MAX(tavg) FILTER (WHERE month =  5) AS tavg_m5,
                    MAX(tavg) FILTER (WHERE month =  6) AS tavg_m6,
                    MAX(tavg) FILTER (WHERE month =  7) AS tavg_m7,
                    MAX(tavg) FILTER (WHERE month =  8) AS tavg_m8,
                    MAX(tavg) FILTER (WHERE month =  9) AS tavg_m9,
                    MAX(tavg) FILTER (WHERE month = 10) AS tavg_m10,
                    MAX(tavg) FILTER (WHERE month = 11) AS tavg_m11,
                    MAX(tavg) FILTER (WHERE month = 12) AS tavg_m12
                FROM weather.weather
                WHERE year = 2020
                GROUP BY locale_id
            ) w ON w.locale_id = h.locale_id;
            ''', [fips]
        )

        if get_pandas:
            ret = pd.DataFrame(ret).astype('float')
            ret.set_index('id', inplace=True)

        return ret

    def get_locale_inf(self):
        pass

        # self._check_locale()
        # inf = self._exec(f'SELECT iso2, iso3 FROM main.locale WHERE id = ?;', [self.locale_id])[0]
        # return f'{inf.iso2} {inf.iso3}'

    def get_geo_counties(self, st_fips):
        return self._exec(f"SELECT gid, statefp10, countyfp10, geoid10, name10, namelsad10 FROM geo.co WHERE statefp10 = %s ORDER BY geoid10;", [st_fips])

    def get_geo_states(self):
        return self._exec('SELECT gid, statefp10, geoid10, stusps10, name10 FROM geo.st ORDER BY geoid10;')

    def get_pop_size(self):
        self._req_locale()
        return self._exec('SELECT pop FROM main.locale WHERE id = %s;', [self.locale_id])[0].pop

    def get_pop_size_synth(self, do_rise=False):
        """Get the size of the U.S. synthetic population that is currently loaded into the database.  The U.S. only
        restriction stems from the fact that currently no other country is covered.  This method is most useful for
        states and counties because synthetic population data is loaded on a per state basis.  Consequently, unless all
        the states are loaded, the entire U.S. synthetic population size will be artifically low.

        Returns:
            int: -2 for non-US locale; non-negative integer for US locales; -1 (or raise an exception) for locales not
                found.
        """

        if not self.is_locale_us():
            return -2

        # return self._get_row_cnt('pop_person_view')
        # return self._exec_get('WITH h AS (SELECT id FROM pop.household WHERE stcotrbg LIKE %s) SELECT COUNT(*) FROM pop.person p WHERE p.household_id IN (SELECT id FROM h);', [f'{self.locale_fips}%'])[0][0]
        # return self._exec_get('SELECT COUNT(*) FROM pop.person AS p INNER JOIN pop.household AS h ON p.household_id = h.id WHERE h.stcotrbg LIKE %s;', [f'{self.locale_fips}%'])[0][0]

        if self.locale_fips is None:  # entire US
            return self._exec('SELECT COUNT(*) FROM pop.person;')[0][0]
        elif len(self.locale_fips) == 2:  # US state
            return self._exec('SELECT COUNT(*) FROM pop.person p INNER JOIN pop.household h ON p.household_id = h.id INNER JOIN main.locale l ON h.st_id = l.id WHERE l.fips = %s;', [self.locale_fips])[0][0]
        elif len(self.locale_fips) == 5:  # US county
            return self._exec('SELECT COUNT(*) FROM pop.person p INNER JOIN pop.household h ON p.household_id = h.id INNER JOIN main.locale l ON h.co_id = l.id WHERE l.fips = %s;', [self.locale_fips])[0][0]
        else:
            if do_rise:
                raise ValueError('Incorrect FIPS code: {self.locale_fips}')
            else:
                return -1

    def get_rand_us_county_fips(self, n=1):
        """Select 'n' random U.S. county FIPS codes (without replacement).

        While fine for the present application, this approach should not be used with very large tables.
        """

        return self._exec('SELECT fips FROM main.locale WHERE iso_num = 840 AND admin1 NOTNULL AND admin2 NOTNULL AND fips NOTNULL ORDER BY random() LIMIT %s;', [n])

    def get_rand_us_state_fips(self, n=1):
        """Select 'n' random U.S. state FIPS codes (without replacement).

        While fine for the present application, this approach should not be used with very large tables.
        """

        return self._exec('SELECT fips FROM main.locale WHERE iso_num = 840 AND admin1 NOTNULL AND admin2 ISNULL AND fips NOTNULL ORDER BY random() LIMIT %s;', [n])

    def get_synth_pop(self, cols=['age'], limit=0, itersize=2000):
        self._req_locale(True)
        limit = f'LIMIT {limit}' if limit > 0 else ''
        if len(self.locale_fips) == 2:    # US state
            locale_id_col = 'st_id'
        elif len(self.locale_fips) == 5:  # US county
            locale_id_col = 'co_id'
        return self._exec(
            f'''
            SELECT {",".join(cols)}
            FROM pop.person p
            INNER JOIN pop.household h ON p.household_id = h.id
            INNER JOIN main.locale l ON h.{locale_id_col} = l.id
            WHERE l.id = %s
            ORDER BY p.id {limit};
            ''', [self.locale_id], itersize
        )

    def is_locale_us(self):
        return self.locale_id is not None and self.locale_iso_num == 840

    @staticmethod
    def pad_fips(fips):
        """Zero-pads the FIPS code provided.

        Many datasets incorrectly declare the FIPS code as integer instead of a string.  Consequently, Alaska's FIPS
        code of '02' becomes just 2.  This method fixes those issues for both US states (a two-digit string) and
        counties (a five-digit string).  The 'fips' code provided is not checked for correctness.

        Returns:
            str: Zero-padded FIPS code if original fips is 1 or 4 characters in length; the original FIPS code
                otherwise.
        """

        f = str(fips)
        if len(f) == 1 or len(f) == 4:
            return f'0{f}'
        return f

    def plot_clusters_map(self, ids, clusters, n_clusters, radius=4, radius_noisy_ratio=0.4, width='100%', height='100%', color_rest='silver', color_noisy='#333333', colors=list(mpl.colors.XKCD_COLORS.values()), txt_show=False, txt_fontsize=1.0, txt_offset=10, **kwargs):
        # list(mpl.colors.TABLEAU_COLORS.values())    # n=10
        # list(mpl.colors.CSS4_COLORS.values())       # n=148
        # list(mpl.colors.XKCD_COLORS.values())       # n=949
        # list(mpl.colors._colors_full_map.values())  # n=1163

        # clusters = [c - min(clusters) for c in clusters]

        if n_clusters <= 0 or n_clusters > len(colors):
            n_clusters = len(colors) - 1

        specs = []
        # for i in range(len(set(clusters))):
        color_idx = 0  # not incremented for sklearn's "noisy sample" (i.e., a -1 cluster)
        for (i, label) in enumerate(np.unique(clusters)):
            if label == -1:
                color = color_noisy
            else:
                color = colors[color_idx] if i <= n_clusters - 1 else color_rest
                color_idx += 1
            specs.append({ 'cluster': label, 'ids': [], 'color': color })
        for i in range(len(ids)):
            specs[clusters[i] + (0 if -1 not in np.unique(clusters) else 1)]['ids'].append(ids[i])
        specs = sorted(specs, key=lambda i: len(i['ids']), reverse=True)

        fg_txt = folium.FeatureGroup(name='Clustering: Text', show=txt_show)
        fg_mkr = folium.FeatureGroup(name=f'Clustering: Markers ({len(np.unique(clusters))})')

        with self.conn.cursor() as c:
            for spec in specs:
                for i in spec['ids']:
                    r = self._exec('SELECT COALESCE(lat, 0.0) AS lat, COALESCE(long, 0.0) AS long FROM main.locale WHERE id = %s', [int(i)])[0]
                    if spec['cluster'] == -1:  # sklearn's "noisy sample"
                        folium.CircleMarker(location=[r.lat, r.long], tooltip=f'<b>Cluster:</b> {spec["cluster"]}', radius=radius * radius_noisy_ratio, color=spec['color'], weight=1.50, opacity=0.50, fill=True, fill_opacity=0.35).add_to(fg_mkr)
                    else:
                        folium.CircleMarker([r.lat, r.long], tooltip=f'<b>Cluster:</b> {spec["cluster"]}', radius=radius, color=spec['color'], weight=1.50, opacity=0.50, fill=True, fill_opacity=0.35).add_to(fg_mkr)
                        if txt_fontsize > 0:
                            folium.Marker([r.lat + txt_offset, r.long + txt_offset], icon=folium.DivIcon(html=f'<div style="color: #777; font-size: {txt_fontsize}em;">{spec["cluster"]}</div>'), opacity=1.00).add_to(fg_txt)

        fig = folium.Figure(width, height)
        m = folium.Map(**kwargs)
        m.add_to(fig)

        folium.TileLayer('https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png', name='Stadia.AlidadeSmooth', attr='&copy; <a href="https://stadiamaps.com/">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors').add_to(m)
        folium.TileLayer('Stamen Toner').add_to(m)
        folium.TileLayer('Stamen Terrain').add_to(m)

        fg_txt.add_to(m, )
        fg_mkr.add_to(m)
        folium.LayerControl().add_to(m)

        return m

    def set_disease(self, name):
        """Sets the current disease by name.

        Returns:
            LocaleDB: self
        """

        self.disease_id = self._get_num('dis.disease', 'id', 'name = %s', [name])
        if self.disease_id is None:
            raise UnknownDiseaseError(f'Disease not found: {name}')
        return self

    def set_locale_by_name(self, admin0, admin1=None, admin2=None):
        """Sets the current locale by name.

        Returns:
            LocaleDB: self
        """

        with self.conn.cursor() as c:
            c.execute('SELECT id, iso_num, fips FROM main.locale WHERE admin0 = %s AND admin1 IS NOT DISTINCT FROM %s AND admin2 IS NOT DISTINCT FROM %s;', [admin0, admin1, admin2])
            if c.rowcount == 0:
                raise UnknownLocaleError(f'No locale found with the following name: {admin0}, {admin1}, {admin2}')
            r = c.fetchone()
            self.locale_id      = r.id
            self.locale_iso_num = r.iso_num
            self.locale_fips    = r.fips

        # if self.locale_fips is not None:
        #     self._set_pop_view_household(self.locale_fips)
        #     self._set_pop_view_household_geo(self.locale_fips, 'st')

        return self

    def set_locale_by_us_fips(self, fips):
        """Sets the current locale by U.S. FIPS code.

        Returns:
            LocaleDB: self
        """

        fips = LocaleDB.pad_fips(fips)
        with self.conn.cursor() as c:
            c.execute('SELECT id FROM main.locale WHERE iso_num = %s AND fips IS NOT DISTINCT FROM %s;', [840, fips])
            if c.rowcount == 0:
                raise UnknownLocaleError(f'No U.S. locale found with the following FIPS code: {fips}')
            self.locale_id = c.fetchone().id

        self.locale_iso_num = 840
        self.locale_fips = fips
        # self._set_pop_view_household(fips)
        # self._set_pop_view_household_geo(fips, 'st')

        return self
