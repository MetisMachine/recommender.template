import uuid
import numpy as np
import pandas as pd
from random import randint
from datetime import datetime
from movies.constants import VOTES_SCHEMA

def make_indicies(col_type, unique_list):
  """Given a list of unique things, create and map an ascending index."""
  count = 0
  rows = []
  col1 = col_type + '_id'
  col2 = col_type + '_int'
  for n in unique_list:
    rows.append({col1: n, col2: count})
    count += 1
  return pd.DataFrame(rows).set_index(col1)


def write_data(data, schema, skafos):
  """Write data out to the data engine."""
  print('Saving {} records with the data engine'.format(len(data)), flush=True)
  res = skafos.engine.save(schema, data).result()
  print(res, flush=True)


def make_fake_votes(num_movies, num_users, skafos):
  """Generate fake votes as a starter set for recommendations."""
  users = [uuid.uuid4() for u in range(0, num_users)]
  movies = np.array(range(0, num_movies), dtype=np.int)
  fake_votes = []
  # For each user loop through on a random subset of movies and assign a random vote
  for u in users:
    for m in np.random.choice(movies, size=num_movies // 2, replace=False):
      time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
      fake_votes.append({'user_id': str(u), 'movie_id': str(m), 'vote': randint(1, 2), 'vote_time': time})
    print("Writing %s votes to cassandra.." % len(fake_votes), flush=True)
    write_data(fake_votes, VOTES_SCHEMA, skafos)
    fake_votes.clear()
