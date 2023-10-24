import streamlit as st
import sqlite3
from sqlite3 import Connection
import random
import itertools as it
import numpy as np

K = 15 

st.set_page_config(page_title="Names", page_icon="🥔", layout="wide", initial_sidebar_state="auto", menu_items=None)

def names_left_to_rank(conn: Connection) -> list[tuple[str, str]]:
   return conn.cursor().execute('''
   select n1.name, n2.name
   from names n1
   join names n2 on n1.name != n2.name
   left join evaluations e1 on n1.name = e1.winner and n2.name = e1.loser
   left join evaluations e2 on n1.name = e2.loser and n2.name = e2.winner
   where e1.winner is null and e2.loser is null
   order by random()
   ''').fetchall()


def choose_names_to_rank_random(conn: Connection) -> tuple[str, str]:
    name1, name2 = names_left_to_rank(conn)[0]
    
    return random.sample([name1, name2], k=2)


def choose_names_to_rank_top(conn: Connection) -> tuple[str, str]:
    """
    rank pairs by abs(name1_top_k_chance - .5) + abs(name1_top_k_chance - .5)
    """
    name_pairs_left = names_left_to_rank(conn)
    top_chances = {name: top_chance for name, _, _, top_chance in get_estimates(conn)}
    
    pair_scores = np.array([1 / (abs(top_chances[n1]) + abs(top_chances[n2])) for n1, n2 in name_pairs_left])
    pair_scores = np.exp(pair_scores) / sum(np.exp(pair_scores))
    idx = np.random.choice(np.array(range(len(name_pairs_left))), p=pair_scores)
    return name_pairs_left[idx]


def insert_evaluation(conn: Connection, winner: str, loser: str):
    conn.execute('insert into evaluations (winner, loser) values (?, ?)', (winner, loser))


def evaluation_progress(conn: Connection) -> tuple[int, int]:
    evaluated = conn.cursor().execute('select count(*) from evaluations').fetchone()[0]
    names = conn.cursor().execute('select name from names').fetchall()

    return evaluated, len(list(it.combinations(names, r=2)))


def get_estimates(conn: Connection) -> list[tuple[str, float, float, float]]:
    return conn.cursor().execute('''
      select name, round(rating, 2), round(rating_std, 2), 100 * top_chance from estimates
    ''').fetchall()


if __name__ == "__main__":
    conn = sqlite3.connect('names.db', isolation_level=None)
    
    ranker_col, display_col = st.columns([0.5, 0.5])

    with ranker_col:
        name1, name2 = choose_names_to_rank_top(conn)

        color = random.choice("red orange green blue violet".split())
        st.markdown(f"## Do you prefer :{color}[{name1}] or :{color}[{name2}]?")

        left_col, right_col = st.columns(2)
        with left_col: 
            st.button(name1, key="left_button", on_click=lambda: insert_evaluation(conn, name1, name2))
        with right_col:
            st.button(name2, key="right_button", on_click=lambda: insert_evaluation(conn, name2, name1))

        evaluated_count, total_count = evaluation_progress(conn) 
        st.text(f'Evaluated {evaluated_count}/{total_count} name pairs')

    estimates = sorted(get_estimates(conn), key=lambda x: x[1], reverse=True)
    with display_col:
        st.header('Current Ranking')
        for idx, (name, value, std, top_k_chance) in enumerate(estimates, start=1):
            estimate_str = f'{value}±{std};'
            st.text(f'{idx:<2}. {name:<10} ({estimate_str:<11} {round(top_k_chance):>2}% chance of being top {K})')
