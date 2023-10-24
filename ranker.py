import time
import sqlite3
from sqlite3 import Connection
import pymc as pm
import numpy as np
from pytensor.tensor.subtensor import set_subtensor
import pytensor as pt

K = 15 

def init_db() -> Connection:
    print('Initializing db...', end='')
    conn = sqlite3.connect('names.db', isolation_level=None)
    cur = conn.cursor()

    cur.execute('PRAGMA foreign_keys = ON')

    cur.execute('begin')
    
    cur.execute('create table if not exists names (name text primary key)')
    if not cur.execute('select name from names').fetchone():
        with open("names.txt") as f:
            names = f.read()
        name_tuples = [(name,) for name in names.split('\n')]
        cur.executemany("insert into names (name) values (?)", name_tuples)

    cur.execute("""
    create table if not exists evaluations (
        winner text,
        loser text,
        constraint fk_winner foreign key (winner) references names(name),
        constraint fk_loser foreign key (loser) references names(name)
    )
    """)

    cur.execute("""
    create table if not exists estimates (
        name text,
        rating float,
        rating_std float,
        top_chance float,
        constraint fk_name foreign key (name) references names(name)
    )
    """)
    
    cur.execute('commit')
    print(' done.')

    return conn
    

def read_evaluations(conn: Connection) -> tuple[list[str], set[tuple[str, str]]]:
    cur = conn.cursor()
    name_records = cur.execute('select name from names').fetchall()
    names = [name_record for name_record, in name_records]
    
    evaluation_records = set(cur.execute('select winner, loser from evaluations').fetchall())

    return names, evaluation_records


def generate_estimates(names: list[str], evaluations: set[tuple[str, str]]) -> dict[str, tuple[float, float, float]]:
    observed_data = np.zeros((len(names), len(names)))
    for winner, loser in evaluations:
        winner_idx = names.index(winner)
        loser_idx = names.index(loser)
        observed_data[winner_idx, loser_idx] = 1

    mask = observed_data.astype(bool)
    
    with pm.Model() as model:
        # Define latent variables for the true values of the items
        name_vars = pm.Normal('names', mu=0, sigma=1, shape=len(names))

        deltas = 0.5 * pt.tensor.ones_like(mask)

        # Compute deltas based on evaluations
        for winner, loser in evaluations:
            winner_idx = names.index(winner)
            loser_idx = names.index(loser)
            delta = name_vars[winner_idx] - name_vars[loser_idx]
            deltas = set_subtensor(deltas[winner_idx, loser_idx], delta)
        p = pm.math.sigmoid(deltas)
        # Adjust the likelihood computation to only consider the compared pairs
        pm.Bernoulli('obs', p=p[mask], observed=observed_data[mask])

        # Perform MCMC
        trace = pm.sample(2000)

    samples = np.vstack(trace.posterior.names)
    estimates: dict[str, float] = dict()
    for name in names:
        name_idx = names.index(name)
        estimate = samples[:, name_idx].mean()
        estimate_std = samples[:, name_idx].std()
        top_k_chance = np.mean([name_idx in list(reversed(sample.argsort()))[:K] for sample in samples])

        estimates[name] = (estimate, estimate_std, top_k_chance)
    
    return estimates


def write_estimates(conn: Connection, estimates: dict[str, tuple[float, float]]):
    estimate_tuples = [(name, rating, rating_std, top_chance)
                       for name, (rating, rating_std, top_chance) in estimates.items()]

    conn.execute('begin')
    conn.execute('delete from estimates')
    conn.executemany('''
    insert into estimates (name, rating, rating_std, top_chance) values (?, ?, ?, ?)
    ''', estimate_tuples)
    conn.execute('commit')


if __name__ == "__main__":
    conn = init_db()
    while True:
        print('Reading evaluations... ', end='')
        names, evaluations = read_evaluations(conn)
        print('done.\nGenerating estimates... ', end='')
        estimates = generate_estimates(names, evaluations)
        print('Writing estimates...', end='')
        write_estimates(conn, estimates)
        print('done.\nSleeping.' )

        time.sleep(1)