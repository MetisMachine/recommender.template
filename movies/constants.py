VOTES_SCHEMA = {
  "table_name": "votes",
   "options": {
       "primary_key": ["user_id", "movie_id"],
  },
  "columns": {
     "user_id": "text",
     "movie_id": "text",
     "vote": "tinyint",
     "vote_time": "timestamp"
  }
}

RECOMMEND_SCHEMA = {
  "table_name": "recommendations",
  "options": {
      "primary_key": ["user_id", "rank"],
      "order_by": ["rank asc"]
  },
  "columns": {
      "user_id": "text",
      "rank": "int",
      "movie_id": "text",
      "pred_rating": "float",
      "pred_time": "timestamp"
  }
}


