# Recommendation-System
What is recommendation 
system?
■ It is an information filtering technique that can 
provide information to a users, which they may 
be interested in.
■ Examples : Netflix (Video), Amazon (Shopping), 
LinkedIn (Social Network)
Dataset
1. Data : 100000 ratings by 943 users on 1682 items. 
2. Item: Information about the items (movies)
3. Genre: A list of the genres
Tweets from Twitter
1. Creating tagword
2. Search and download tweets 
using tagwords and do the 
same for each movie in the 
list
3. Clean tweets and perform 
sentiment analysis for each 
tweet for each movie
4. Average the sentiment score 
for each movie combining 
tweets for each movie
Tweets from Twitter
1. Tweepy for Twitter API
2. Creating tagword
3. Downloaded tweets for movies
4. Cleaning tweets
5. Vader Analysis
Sentiment Analysis
■ We've used sentiment analysis to get the sentiment score from the twitter.
■ These tweets are stored in a data frame, text cleaning of the data and tweets is 
done for better outcomes.
■ The clean Tweet data is stored in a data frame text cleaning
■ Using SentimentIntensityAnalyzer() function, compound score is generated 
based on the tweet words
■ The Sentiment scores are then labelled as: Positive >0, Negative <0, and the 
remaining are sorted as Neutral==0
Collaborative Filtering
■ It's a method based on the user's historic data It doesn't need anything other than 
user's historic data.
■ Because it’s based on historical data, the core assumption here is that the users 
who have agreed in the past tend to also agree in the future. In terms of user 
preference, it usually expressed by two categories.
■ For implementing the cosine similarity in the collaborative filtering the angle 
between two vectors is calculated.
■ When we display lower angle displays higher similarity.
Collaborative Filtering
■ We use matrix factorization in the 
collaborative filtering system to 
reduce the size of actual matrix into 
smaller matrices to simplify 
complex calculations.
■ Using MF each User is 
associated with the Movie rating, 
Using a vector latent feature
Hybrid Recommendation System
■ It has two main branches item based and user based
■ It works with combining two or more filtering 
techniques in different ways to increase the 
performance and accuracy of the recommender 
system.
■ Pytorch layers were used to buid the hybrid 
recommender system using neural embodiment
■ Neural embedding layer is a tensor, a vector and 
vector composed of random numbers 
like conventional matrix.
Result and Conclusion
■ RMSE Decreases as the number of latent features increases from 20-50.
■ Top 10 recommendation list was generated from each model. Based on the 
comparison result, all metrics, except recall, were better in the list generated 
by Hybrid Recommender.
■ Top 20 recommendation List was generated from each model. Based on 
the comparison result, metrics of the Hybrid Recommender were better than the 
ones of matrix factorization.
■ Hybrid Recommender Exhibited better performance than the other method. Thus, It 
can be a potential method to improve the traditional recommender systems 
that drive industry today.
