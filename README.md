# GoodReadsRecs

Our goal is to build a recommendation system based on historical user ratings from the GoodReads dataset. All the data can be found: [here](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home). 

From the review, book, and interaction data we derive 3 main matrices (see Preprocessing for exact implementation). 

* Book Data: This BxT matrix shows the number of people that gave a book i, tag j. 

* Review Data: A TF_IDF matrix of the top 10 reviews for each book

* Interaction Data: The most critical one. This UxB matrix shows the rating user i gave to book j. While Goodreads does not allow 0 ratings, we set 0 ratings to 0.5 to make storage easier. 

All 3 matrices are enormous, hundreds of thousands of rows and columns so traditional storage is infeasible and wasteful since data is sparse. So we use sparse matrices to maintain these. 

## Collab Filtering

Find similar users to a given user and find what those users like

## Content Filtering

Two ways to go about this.

* User-based, given a book, find other books which similar people liked. Similar to Collab. 

* Characteristic based. Using Review Data, tag data, etc, find similar books in terms of attributes. Dimensionality reduction is needed here due to vast amount of dimensions in  things like TF_IDF matrices. 

## SVD 

This solution is by and far the best one of out the 3 frameworks we use. 

SVD is the state-of-the-art non-deep solution to the recommendation problem. Netflix uses this exact system to make recommendations, and we can leverage this powerful technique to make great recommendations.

The first thing to do is to get our interaction data in the
correct format. We are given a dataframe like so:
What we need is a pivot table of this data. A matrix I that
MxN, where M is the number of users and N is the number
of books. The ith row and jth column tell us the rating user
I gave to book J.
Naturally, this gives us very sparse data, as the majority of
users have rated a fraction of the whole universe of books.
And this is where we make an optimization of the standard
SVD algorithm, inspired by Simon Funk’s Netflix solution.
This allows us to use the entirety of the data and process it in
a reasonable time. This solution is orders of magnitude faster.
But first, we consider what SVD does. It decomposes a
matrix into 3 parts A = UDV. A in our case is our interactions
data, U would represent our user space and V represents our
book space. Obviously, a perfect one doesn’t exist, so SVD
aims to minimize the MSE of this fitting operation.
Note that the right part is a dot product between row i and
column of matrices U and V respectively.

With millions of dimensions, this can take quite long. But using our sparse nature, we only consider non-zero dimensions. A simple but effective speedup.

This is a supervised learning algorithm, actually, with training, validation, and testing data. So we split our interaction data BY USER. This is critical, as we do not want part of one
user to be in another sample, as this would corrupt our test
set. After training, we predict, and our results are excellent.
When predicting the ratings of users the machine had never
seen before, we get a mean absolute error of 0.89 and an
MSE of 1.46. **This tells us that the machine can predict what
a user will rate a book within 0.89 points.** With this, we can
recommend books quite easily, as the rating user I will give
book J is given by multiplying row I of U by column J of V
scaled by the diagonal elements.

We selected the parameters by using cross validation but had
to limit the data we did it on to make this practical. We tested
various n-factors and regularization terms and settled on reg:
0.005 and factors 15. Though various different combinations
did not actually change performance too much.
