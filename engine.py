import os
import uuid
import numpy as np
import pandas as pd
from random import randint
from datetime import datetime
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel
from movies.constants import *
from movies.logger import get_logger
from skafossdk import *

log = get_logger('recommender')
ska = Skafos()
BATCH_SIZE = os.environ['BATCH_SIZE']


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
  # Save out using the data engine
  data_length = len(data)
  log.info('Saving {} records with the data engine'.format(data_length))
  res = skafos.engine.save(schema, data).result()
  log.debug(res)


# VOTES TABLE 
# In order to get recommendations from the model, a votes table must be created with a schema defined in constants.py
# Unless that table exists, build an initial votes table with fake data.. 
def make_fake_votes(num_movies, num_users):
  users = [uuid.uuid4() for u in range(0, num_users)]
  movies = np.array(range(0, num_movies))
  fake_votes = []
  for u in users:
    for m in np.random.choice(movies, size=num_movies // 2):
      time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
      fake_votes.append({'user_id': str(u), 'movie_id': str(m), 'vote': randint(1,2), 'vote_time': time})
    print("Writing %s votes to cassandra.." % len(fake_votes), flush=True)
    write_data(fake_votes, VOTES_SCHEMA, ska)
    fake_votes.clear()

# Generate fake votes, create table, and write to the table
# If a "votes" table exists already, comment the next line out
make_fake_votes(num_movies=20, num_users=8)


# Use Skafos Data Engine to Pull in Votes data
log.info('Setting up view and querying movie list')
res = ska.engine.create_view("ratings", {"table": "votes"}, DataSourceType.Cassandra).result()
ratings_query = "SELECT * FROM ratings"
ratings = ska.engine.query(ratings_query).result()['data']
log.info("Ingested {} movie ratings".format(len(ratings)))

# Make a data frame of user ratings
ratings_df = pd.DataFrame(ratings).dropna()

# Create indicies for movies and users
user_ind = make_indicies('user', unique_list=ratings_df.user_id.unique())
movie_ind = make_indicies('movie', unique_list=ratings_df.movie_id.unique())

# Join on index to get user_int and movie_int
ratings_df = ratings_df.set_index('movie_id').join(movie_ind)\
  .set_index('user_id').join(user_ind)

# Convert votes to -1 and 1
ratings_df['vote'] = ratings_df.vote.apply(lambda x: 1 if x == 2 else -1)

# Build interactions object (building torch tensors underneath)
interactions = Interactions(item_ids=ratings_df.movie_int.astype(np.int32).values,
                            user_ids=ratings_df.user_int.astype(np.int32).values,
                            num_items=len(ratings_df.movie_int.unique()),
                            num_users=len(ratings_df.user_int.unique()),
                            ratings=ratings_df.vote.astype(np.float32).values)

# Build Explicit Matrix Factorization Model
model = ExplicitFactorizationModel(loss='logistic', n_iter=10)
model.fit(interactions)

# Get predictions out for each user
full_movies = movie_ind.movie_int.unique()
recommendations = []
# convert datetime to string to ensure serialization success
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
batch_count = 0

for device, user_row in user_ind.iterrows():
  # Get list of all movies this user voted on
  user = user_row.user_int
  user_votes = ratings_df[ratings_df.user_int == user].movie_int.unique()
  # Calculate difference in the two lists - rate those movies only
  m = np.setdiff1d(full_movies, user_votes)
  user_rank = 0
  # for each movie and prediction for a given user, create a recommendation row
  for movie, pred in zip(m, model.predict(user_ids=user, item_ids=m)):
    batch_count += 1
    user_rank += 1
    log.debug('movie {}'.format(user_rank))
    # For each prediction, make a recommendation row
    recommendations.append({'user_id': device,
                            'rank': user_rank,
                            'movie_id': movie_ind[movie_ind.movie_int == movie].index[0],
                            'pred_rating': float(pred),
                            'pred_time': timestamp})
    if batch_count % int(BATCH_SIZE) == 0:
      write_data(recommendations, RECOMMEND_SCHEMA, ska)
      # Clear the recommendation set
      recommendations.clear()
  # clean up anything remaining in a partial batch
  if recommendations:
    log.info('writing out a final partial batch') 
    write_data(recommendations, RECOMMEND_SCHEMA, ska)

