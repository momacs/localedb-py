import numpy as np
import time

from localedb.localedb import LocaleDB


# ----------------------------------------------------------------------------------------------------------------------
def disp_locale_inf(db):
    t0 = time.perf_counter()
    print(f'id: {db.locale_id:>8}  iso_num: {db.locale_iso_num:>3}  fips: {db.locale_fips or "-":5}  pop: {db.get_pop_size():>12}  pop-synth: {db.get_pop_size_synth():>12}  ({time.perf_counter() - t0:.0f} s)', flush=True)

def disp_locale_dis_dyn_by_day(db):
    conf = db.get_dis_dyn_by_day_conf()
    print(f"{db.locale_id:>8}: n={conf.size:<3}  {conf.flatten().tolist()[:42]}")

def disp_synth_pop(db):
    c = db.get_synth_pop(['sex', 'age', 'WIDTH_BUCKET(age::INTEGER,ARRAY[18,60]) AS age_grp', 'income', 'CASE WHEN school_id IS NULL THEN 0 ELSE 1 END AS is_student', 'CASE WHEN workplace_id IS NULL THEN 0 ELSE 1 END is_worker'], limit=4)
    print(np.array(c).tolist())


# ----------------------------------------------------------------------------------------------------------------------
db = LocaleDB()
db.set_disease('COVID-19')

print('Random locale selection:')
print(db.get_rand_us_state_fips(3))
print(db.get_rand_us_county_fips(3))
print()

print('Basic population and synthetic population queries:')
db.set_locale_by_name('China')                            ; disp_locale_inf(db)
db.set_locale_by_name('Italy')                            ; disp_locale_inf(db)
db.set_locale_by_name('US')                               ; disp_locale_inf(db)

db.set_locale_by_name('US', 'Alaska')                     ; disp_locale_inf(db)
db.set_locale_by_us_fips('02')                            ; disp_locale_inf(db)
db.set_locale_by_name('US', 'Alaska', 'Anchorage')        ; disp_locale_inf(db)
db.set_locale_by_us_fips('02020')                         ; disp_locale_inf(db)

db.set_locale_by_name('US', 'Pennsylvania')               ; disp_locale_inf(db)
db.set_locale_by_us_fips('42')                            ; disp_locale_inf(db)
db.set_locale_by_name('US', 'Pennsylvania', 'Allegheny')  ; disp_locale_inf(db)
db.set_locale_by_us_fips('42003')                         ; disp_locale_inf(db)
print()

print('Disease dynamics queries:')
db.set_locale_by_name('China')                            ; disp_locale_dis_dyn_by_day(db)
db.set_locale_by_name('Italy')                            ; disp_locale_dis_dyn_by_day(db)
db.set_locale_by_name('US')                               ; disp_locale_dis_dyn_by_day(db)
db.set_locale_by_name('US', 'Alaska')                     ; disp_locale_dis_dyn_by_day(db)
db.set_locale_by_name('US', 'Alaska', 'Anchorage')        ; disp_locale_dis_dyn_by_day(db)
db.set_locale_by_name('US', 'Pennsylvania')               ; disp_locale_dis_dyn_by_day(db)
db.set_locale_by_name('US', 'Pennsylvania', 'Allegheny')  ; disp_locale_dis_dyn_by_day(db)
print()

print('Disease dynamics comparison:')
db.set_locale_by_name('US')
print(db.get_dis_dyn_comp_stats_conf([1, 1, 2, 2, 6], day_to=5))
print()

print('Synthetic population retrieval queries:')
db.set_locale_by_name('US', 'Pennsylvania')               ; disp_synth_pop(db)
db.set_locale_by_name('US', 'Pennsylvania', 'Allegheny')  ; disp_synth_pop(db)
db.set_locale_by_name('US', 'Pennsylvania', 'Adams')      ; disp_synth_pop(db)

db.set_locale_by_us_fips('02020')
conf = db.get_dis_dyn_by_day_conf(20,77)  # get the time series of confirmed COVID-19 cases from day 20 to day 77
print(conf.flatten().tolist())            # print the
