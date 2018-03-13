# recommender.template
This template shows example usage of the Metis Machine platform for the purpose of building a recommendation engine model. Given a dataset of user IDs, item IDs, and ratings (either boolean or range), this template allows the user to get item recommendations for each user stored in a separate table. Here we use user votes (thumbs up and down) on various movies as the example use case.

## Dependencies
1. User must have python 3.6
2. User must have skafos CLI and skafos SDK installed (`brew install skafos`)
3. User must have git installed and a github account created --> https://git-scm.com

## Project Structure
- *movies*
  - `__init__.py`
  - `constants.py`
  - `logger.py`
  - `helpers.py`
- *metis.config.yml*
- *environment.yml*
- *README.md*
- *engine.py*

## Flow
- The files in *movies* contain helper functions, various constants (table schemas), and a logger.
- *engine.py* is the main script that runs for this task. It performs the following:
  - Generates fake user-movie votes and stores them in a table (unless a keyspace is provided).
  - Pulls in votes data from table and does some reshaping and cleaning.
  - Trains a recommender model using Explicit Matrix Factorization (using PyTorch based implementation called [Spotlight](https://github.com/maciejkula/spotlight))
  - Generates movie recommendations for each user
  
## Options
These are the environment variables that could be set from the terminal in the project directory using the Skafos CLI. 

**optional**

- BATCH_SIZE, default=10: Number of rows written to the database at a time.
- KEYSPACE, default=None: Namespace in cassandra if the user has a set of tables to use that are pre-built.

## Deployment
Once the user is ready to fire it off (after following the dependency steps above):
1. CREATE: a github repository and attach it to the skafos app here --> https://github.com/apps/skafos
2. OPEN: one of the files and make some sort of change (add a comment, add new functionality, change config file, etc)
3. From Project Directory RUN:
```bash
  $ git init
  $ git add .
  $ git commit -am "<message>"
  $ git remote add origin git@github.com:<organization>/<repository-name>.git
```
