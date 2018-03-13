import os
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel
from movies.constants import *
from movies.helpers import *
from movies.logger import get_logger
from skafossdk import *

## Setup the logger and initialize Skafos
log = get_logger('recommender')
ska = Skafos()

## Extract Batch Size and Keyspace from the environment
if 'BATCH_SIZE' in os.environ:
  BATCH_SIZE = os.environ['BATCH_SIZE']
else:
  BATCH_SIZE = 10

## Set optional keypace (make sure data sharing rights have been activated if between two different organizations)
if 'KEYSPACE' in os.environ:
  KEYSPACE = os.environ['KEYSPACE']
else:
  KEYSPACE = None

## Set the recommendation schema
rec_schema = RECOMMEND_SCHEMA
if KEYSPACE:
  rec_schema["namespace"] = KEYSPACE

## VOTES TABLE SETUP
# In order to get recommendations from the model, a votes table must be created with a schema defined in constants.py
# Unless that table exists, we will build an initial votes table with fake movie data..
if not KEYSPACE:
  log.info("Generating fake votes data")
  make_fake_votes(num_movies=12, num_users=3, skafos=ska)

## Use Skafos Data Engine to get votes
log.info('Setting up view and querying movie list')
if KEYSPACE:
  res = ska.engine.create_view("ratings", {"keyspace": KEYSPACE, "table": "votes"}, DataSourceType.Cassandra).result()
else:
  res = ska.engine.create_view("ratings", {"table": "votes"}, DataSourceType.Cassandra).result()

ratings = ska.engine.query("SELECT * FROM ratings").result()['data']
log.info("Ingested {} movie ratings".format(len(ratings)))

## Make a data frame of user ratings and print head
ratings_df = pd.DataFrame(ratings).dropna()
print(ratings_df.head(), flush=True)

## Create indicies for movies and users
user_ind = make_indicies('user', unique_list=ratings_df.user_id.unique())
movie_ind = make_indicies('movie', unique_list=ratings_df.movie_id.unique())

## Join on index to get user_int and movie_int
ratings_df = ratings_df.set_index('movie_id').join(movie_ind)\
  .set_index('user_id').join(user_ind)

## Convert votes to -1 and 1
# This may need to be adjusted based on how you store your votes
# Thumbs up must be 1 and Thumbs down must be -1
ratings_df['vote'] = ratings_df.vote.apply(lambda x: 1 if x == 2 else -1)

## Build interactions object (building torch tensors underneath)
log.info("Building interactions object")
interactions = Interactions(item_ids=ratings_df.movie_int.astype(np.int32).values,
                            user_ids=ratings_df.user_int.astype(np.int32).values,
                            num_items=len(ratings_df.movie_int.unique()),
                            num_users=len(ratings_df.user_int.unique()),
                            ratings=ratings_df.vote.astype(np.float32).values)

## Build Explicit Matrix Factorization Model
# We use logistic loss since the interaction rating is binary (-1, 1)
log.info("Training the recommendation engine model using Explicit Matrix Factorization")
model = ExplicitFactorizationModel(loss='logistic', n_iter=10)
model.fit(interactions)

# Prepare to get predictions out for each user
full_movies = movie_ind.movie_int.unique()
recommendations = []
# Convert datetime to string to ensure serialization success
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
batch_count = 0

for device, user_row in user_ind.iterrows():
  # Get list of all movies this user voted on
  log.info("Generating recommendations for user {}".format(device))
  user = user_row.user_int
  user_votes = ratings_df[ratings_df.user_int == user].movie_int.unique()
  # Calculate difference in the two lists - rate those movies only
  m = np.setdiff1d(full_movies, user_votes)
  user_rank = 0
  # for each movie and prediction for a given user, create a recommendation row
  for movie, pred in zip(m, model.predict(user_ids=user, item_ids=m)):
    batch_count += 1
    user_rank += 1
    log.debug('...movie {}'.format(user_rank))
    # For each prediction, make a recommendation row
    recommendations.append({'user_id': device,
                            'rank': user_rank,
                            'movie_id': movie_ind[movie_ind.movie_int == movie].index[0],
                            'pred_rating': float(pred),
                            'pred_time': timestamp})
    if batch_count % int(BATCH_SIZE) == 0:
      write_data(recommendations, rec_schema, ska)
      # Clear the recommendation set
      recommendations.clear()
  # clean up anything remaining in a partial batch
  if recommendations:
    log.info('...writing out a final partial batch')
    write_data(recommendations, rec_schema, ska)

log.info('Done.')
