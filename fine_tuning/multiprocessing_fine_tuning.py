import os
import multiprocessing as mp

processes = ('min', 'mid', 'max')


def run_python(process):
    os.system('python fine_tuning.py --type {}'.format(process))


pool = mp.Pool(processes=3)
pool.map(run_python, processes)
