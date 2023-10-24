pipenv run python3 ranker.py &
sleep 5 &&
pipenv run streamlit run ui.py
