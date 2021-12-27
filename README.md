	Abstract
The primary goal of this research was to develop collaborative filtering-based recommender and hybrid recommender systems. We utilised the Matrix Factorization approach to develop the collaborative filtering-based recommender, and the Neural Embedding layer from the Py Torch package to produce the hybrid recommender. Furthermore, we created an advanced collaboration system that uses the Kaggle API to directly get the data set.

	Introduction
A machine-learning system that recommends suitable goods or products to consumers is known as a hybrid recommender system (HRS). HRS is becoming increasingly popular in a variety of business industries. HRS is expanding in the travel and healthcare industries, in addition to retail and e-commerce. HRS can anticipate future trends using a variety of approaches, including matrix factorization, based on historical data. Seeing good suggestions when browsing a movie, for example, promotes an immersive movie-watching experience as well as consumer satisfaction. It expedites the process of locating all of the preferred films and provides a list of comparable interests to movies and genres. A sufficient amount of data is required for developing a RS. A movie recommender requires user preferences, most recently seen movies, movie ratings, and other variables relating to user behaviour toward any movie in the database. RS employs three filtering algorithms: (1) content-based filtering, which makes use of attribute information; (2) collaborative filtering, which makes use of user-item interaction; and (3) hybrid system, which makes use of both information. Collaborative Filtering is classified into three types: (1) Memory-based, which computes the similarity measure between users or objects based on user history data ratings. (2) Model-based collaborative Filtering, in which embeddings represent each user and each movie in the data; (3) Neural-based collaborative Filtering, in which embeddings represent each user and each movie in the data. The purpose of this research is to examine and contrast the performance of Collaborative Filtering and Hybrid-based approaches in producing movie recommendations. The Collaborative Filtering method is divided into two parts: movie similarity and movie rating prediction. While the Collaborative Filtering-based technique has a Content-based add-on to the Hybrid-based approach. It makes use of both movie and rating data and is divided into four sections: 1) Text preprocessing, 2) Term weighting, 3) Movie clustering, and 4) Collaborative Filtering. The hybrid-based technique outperformed the standard Matrix factorization approach in terms of performance.
The following are a few examples of machine learning recommendation systems: 
	Popularity based 
	Classification based  
	Content based  
	Collaborative filtering   
	User based nearest collaborative filtering  
	Item based nearest neighbor collaborative filtering 
	Singular value decomposition and matrix factorization  
	Hybrid 








	Methodology
3.1 MODEL DESIGNING
A Collaborative Filtering-based Recommender System was built using Matrix Factorization. In contrast, Pytorch was utilised to construct a Hybrid RS. Figures 1 and 2 depict the full process or algorithms used to the two methodologies employed in this project.

  
Credits: https://www.researchgate.net/figure/Main-procedures-of-matrix-factorization-based-algorithms_fig7_303556519
Figure 1 - Matrix Factorization Algorithm
 
Credits: https://www.semanticscholar.org/paper/Hybrid-recommendation-model-based-on-incremental-Wang-Zhang/3f092f1895077dc64cec5efd5d29b0e1dbf1017c
Figure 2 - Hybrid-Based algorithm
3.2 HYBRID RECOMMENDER SYSTEM
Figure 3 depicts a Hybrid Recommender that combines information from both Collaborative Filtering and content-based techniques. The sentiment ratings (positive, neutral, or negative) and genres are used in the content-based filtering. Collaborative Filtering, on the other hand, is based on user-generated movie ratings. Movie features, user ratings, and the embedding layers discussed in section 3.5 can all be combined. The combined data were input into a neural network, which produces a value based on an activation function. If the output does not match an anticipated value, the neural network modifies its weights such that the result is closer to the predicted value after a fresh run. 

Each run is referred to as an era. The predicted output value is the rating given to the movie by the users who saw it. If a trained neural network can reliably predict the user rating, it is feasible to provide a list of films that the user would enjoy.
 Figure 3 - Hybrid Recommender System Design

3.3  DATASET DESCRIPTION
Two datasets from the'ml-100k' datasets collection released by the Group Lens Research Experiment at the University of Minnesota were used in this project. The initial dataset, dubbed u.data, is a collection of 100000 ratings provided to 1682 movies by 943 individuals. This collection contains instances with a user id, an item id, a rating, and a timestamp. The second dataset, u.item, contains information on movie aspects such as genres, movie id, movie title, release date, and video release date. The 19 categories that characterise each film are vital in building a Hybrid Recommender. Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, and Western are among the genres included in this dataset.

We utilised the following strategies to get the required results for the Project tasks:
	Finding Similarity in Collaborative Recommender Using Cosine Similarity
	Designing a Collaborative Recommender System Using Matrix Factorization
	Py Torch Embedding Layer Hybrid Recommender
3.4  MATRIX FACTORIZATION
One of the most extensively used techniques for dimensionality reduction is matrix factorization. It works by compressing the original sparse [user][item] and introducing them as a distinct matrix, which displays the user and things as undefined feature vectors. This equation depicts the breakdown process, where f represents the number of latent factors:
ratings Matrix[user][item] = sum (userFeature[f][user] * item Feature[f][item])

This equation will provide two matrices: userFeature of size (user * f) and itemFeature of size (f * item). Figure 4 shows the complete method.
 
Figure 4 - Matrix Factorization 
3.5   NEURAL EMBEDDING LAYER
The Neural Embedding Layer is a tensor, or vector of vectors, consisting of random integers in the same way that a traditional matrix is. The number of vectors corresponds to the number of rows, and the size of each vector corresponds to the number of columns. When we instantiate an object using Pytorch, we must give the number of vectors (embeddings) as well as the size of each vector. On the left side of Figure 5, for example, we have an object initialised with five vectors of size 3. This object has five vectors, each with three random values. The right side of the same image shows an application of this to the Hybrid Recommender system. We can build an embedding layer with five vectors, one for each user, and three latent characteristics. If we feed the embedding layer a list of users, this object will return the relevant vector for each user.
 
Figure 5 – Latent features generation with embedding layers
3.6 SENTIMENT SCORE CALCULATION
The sentiment ratings of each post collected from Twitter were calculated using Vader, a popular sentiment analysis model. Vader's Sentiment Intensity Analyzer library generated a dictionary output with sentiment (key) and polarity scores (value). The weight of the compound can be used as an indicator to identify the likely emotion of each tweet. Scores between -0.05 and 0.05 are considered neutral, whereas anything above 0.05 is positive and anything below 0.05 is negative. Pandas groupby aggregate mean function was used to obtain the average (compound) score for each movie.
3.7 ACTIVATION FUNCTION OF HYBRID RECOMMENDER SYSTEM
The activation function of the Rectified Linear Unit (ReLU) was utilised to construct the Hybrid Recommender system. The ReLU activation function is used in many neural network topologies. Some of the explanations behind this tendency include: 1) The computation of ReLU is easier than that of Tanh and Sigmoid activation functions: 2) ReLU can produce true zero values, resulting in a desired quality known as sparse representation, which can be used to expedite learning: 3) ReLU optimises the model better since it is more linear than the tanh and sigmoid functions. Figure 6 depicts the rectified linear activation function graph, and its formula is (1):
f(z)= {0,for z<0 z,for z≥0    			      (1)

 
Figure 6 - ReLU graph







4. Analysis Steps 
This stage is critical in our assignment since we can only achieve our result based on analysis, and in the following steps, we will see our output in both statistical and graphical form. Only when the analytic stages are completed can we obtain the top five themes.
	Here, we were working with the dataset collected from MovieLens. Initally, these data were extracted from separate files namely, u.data, u.item and u.genre.  
	All the movie related information was availabe in u.item file, like, movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western. 
	The u.data file has information related to the user and ratings i.e. user id | item id | rating | timestamp. 
	In order to deal with the quantifiable values we fetched total tweets and other related data for Twitter tweets.
	During the extraction process, a restriction of 5 tweets per movie was imposed. According to the tweets collected in Figure 7, "The Big Squeeze" received the most positive reviews. The song "Til There Was You" came next, with an emotion score of 93 percent.
 
Figure 7 - Top 10 movies

	According to Figure 8, the film with the lowest emotion score is "Killer: A Journal of Murder (1995)," with -0.84, followed by "Faster Pussycat! Kill! Kill!" with -0.83.

 
Figure 8 - Bottom 10 movie Graph

Analysis Steps Using a bar diagram, showing the top-5 topics for any twitter handler of your choice
	Statistical View:
 

	Graphical View:
 





4.1 EVALUATION METRICS
To compare the two models, two measures were used: Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) (MAE). RMSE (2) calculates a model's inaccuracy in predicting quantitative data.


RMSE= √(1/n ∑_(i=1)^n(〖y_i-y ̂_i)〗^2 )                                                 (2)

In this equation, y1, y2,..., yn are the predicted values, y1, y2,..., yn are the observed values, and n is the number of observations. ns. MAE (5) is one of numerous metrics for describing and evaluating the quality of a machine learning modality. el. In this context, error refers to the subtraction of the predicted value from the actual value as below.

Prediction Error=Actual Value-Predicted Value    (3) 

Each record is converted to positive by the prediction error. This equation is used to calculate the Absolute Error (4).

Absolute Error → |Prediction Error|                                    (4)

Finally, we compute the mean of all absolute errors observed (Average sum of all fundamental errors).

MAE=  (∑_(i=1)^n |y_i-y ̂_i |)/n 				         (5)       

Here, y1 denotes the predicted value, y1 is the observed value, and n denotes the number of observations. .

Precision (6) is computed by dividing the total number of true positives and false positives by the number of true positives.

  Precision=(True Positives)/(True Positives+ False Positives)                               (6)

The number of genuine positives divided by the total number of true positives and false negatives yields recall (7).

Recall=(True Positives)/(True Positives+ False Negatives)                                     (7)

The F-score (8) combines accuracy and recall into a single statistic that accounts for both.

F_score=(2*Precision*Recall)/(Precision+Recall)                                                    (8)

4.2 Results and Analysis                                                                                          
Two distinct experiments were carried out in this section. The computed RMSE for both models (Matrix Factorization and Hybrid Recommender) were derived when the model was run with latent vectors of sizes 20, 30, 40, and 50. For each model, a line diagram was drawn with the RMSE x Latent Feature sizes.

The Matrix Factorization model exhibited a linear relationship between the amount of latent features and RMSE, as seen in Figures 9 and 10. Furthermore, as the number of latent characteristics rises, the RMSE drops. The RMSE with the lowest value was about 1.85.

The Neural Network graph in Figure 10 shows that RMSE drops as the number of embedding layers increases until it reaches 40, at which point RMSE begins to keep constant at 0.91 with less variability. By comparing the two outcomes, we can conclude that the Hybrid Recommender model produced a lower RMSE value than the Matrix Factorization model.


 
Figure 9 - Matrix Factorization - RMSE x Numb	er of latent features
 

Figure 10 - Hybrid Recommender - RMSE x Number of latent features

The model with the lowest RMSE from the previous part was chosen to be utilised further in this section.

The following experiments were carried out for each model:

	We made a list of the top 20 recommendations for a user.
	Checked to see whether the user had viewed any of the movies in the suggested list.
	The RMSE, MAE, Precision, Recall, and F1-Score were determined based on these ideas.
	The procedure was repeated for all users.
	WE computed the average of all metrics.

Table 1 displays the metrics for each model's top-10 suggestion list. The RMSE and MAE values reported by the Hybrid Recommender were lower than those presented by the Matrix Factorization model. The recall of the Matrix Factorization model was 14% greater than the recall of the Hybrid Recommender model. The accuracy and F1-score of the Neural Network Recommender, on the other hand, were 53% and 37% higher, respectively.


Metrics of Top-10 recommendation
 	Matrix Factorization	Hybrid Recommender
RMSE	1.289	0.879
MAE	1.099	0.826
Precision	0.143	0.220
Recall	0.862	0.755
F1-score	0.229	0.315
Table 1 - Metrics of Top-10 recommendation

Table 2 displays the metrics for each model's Top 20 suggestion list. The Hybrid Recommender model outperformed the Matrix Factorization model on all measures. Precision, recall, and F1-score of Neural Collaborative Filtering were 24%, 57%, and 24% higher, respectively, than those of the old technique.


Metrics of Top-20 recommendation
 	Matrix Factorization	Hybrid Recommender
RMSE	1.435	0.925
MAE	1.238	0.834
Precision	0.209	0.259
Recall	0.739	0.903
F1-score	0.295	0.367
Table 2 - Metrics of Top-20 recommendation


5. Conclusion 

To create a Recommender System, two separate recommendation systems were created (RS). During the development process, the key approaches employed and studied were traditional Matrix Factorization and Hybrid Recommender (powered by the newly developed Neural Collaborative Filtering).

Three experiments were carried out to assess the performance of each model. According to the provided graphs, RMSE reduces as the number of latent features grows from 20 to 50. Each model provided a top ten suggestion list. All metrics, except recall, were superior in the list created by Hybrid Recommender, according to the comparative results. Each model produced a list of the top 20 recommendations. According to the comparative results, the metrics of the Hybrid Recommender outperformed those of Matrix Factorization.

In conclusion, Hybrid Recommender performed better than the other technique. As a result, it has the ability to enhance the standard recommender systems that drive business today.


6. References
	https://tweettopicexplorer.neoformix.com/#n=NYTimes
	https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
	https://github.com/c17hawke/flask-based-wordcloud-generator
