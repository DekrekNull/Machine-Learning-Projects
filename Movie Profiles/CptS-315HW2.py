# Derek Wright 11766151
# CptS 315 Homework Assignment 2
# 
#  100004 ratings
# 1296 tag applications
# 9125 movies
# 671 users, at least 20 movies rated each
# genres: Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Romance, Sci-Fi, Thriller, War, Western, (no genres listed)

# ratings.csv header and example line :
#   "userId,movieId,rating,timestamp"
#   "1,31,2.5,1260759179"

# movies.csv header and example line :
#   "movieId,title,genres"
#   "1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy"

# tags.csv header and example line : 
#   "userId,movieId,tag,timestamp"
#   "15,339,sandra 'boring' bullock, 1138537770"
# 
# Instructions
# A) Construct the profile of each movie: at least using ratings *Other info optional (e.g. genres, tags).
# B) Copmute similarity score for all movie-movie pairs using centered cosine.
# C) Compute the neighborhood set N for each movie. Size of 5 highest similarity score movies, ties broken by lexicographic ordering of movieId.
# D) Estimate the ratings of other users who didn't rate each movie using the neighborhood set.
# E) Compute the recommended movies for each user. Top 5 with highest ratings, ties broken by lexicographic ordering.
# Output) Top 5 recommendations for each user.
#   Line format: "User-id1 movie-id1 movie-id2 movie-id3 movie-id4 movie-id5"

# run_code.sh should execute code: compile, link, and execute
# Assume relative file paths e.g. "./filename.txt" or "./hw2/filename.txt"
# all files should be stored in a zipped file named 11766151.zip

# Global Variables
from copy import deepcopy
import csv
from operator import contains

RATING_FILE="./data/ratings.csv"
MOVIE_FILE="./data/movies.csv"
TAG_FILE="./data/tags.csv"
OUT_FILE="./data/output.txt"

# Change this the number of movies to make profile, only profiles will be made for movies with an ID less than this value
# The full set of data take FOREVER to calculate scores for, I'm not sure if my time complexity is particularly bad
# or if this is expected, but I made it as efficient as I could. It can do up to 1500 movies in a few minutes, but the full 9000 takes a couple hours
MOVIE_LIMIT = 200000

# Number of users that have ratings for the movies
USER_COUNT = 671
# Dictionary that stores the similarity scores of each movie pair, 
# Key: tuple (movie1, movie2), Value: similarity score of movie1 and movie2
SIM_SCORES = {}
# Dictionary that stores the 5 closes neighbors for each movie
# Key: MOVIE_ID index for the movieid, Value: list of 5 closest neighbors
MOVIE_NEIGHBORS = {}

# Array of mvoie profiles, each value is a tuple
# (movieId, array of user ratings with indiced being the userid)
MOVIE_PROFILES = []
# array of movieids, index of each movie is the key for that movie in the MOVIE_NEIGHBOR dictionary
MOVIE_IDS = []
# Array that stores the estimated rating for each movie in the same format as the MOVIE_PROFILES array
ESTIMATED_RATINGS = []
# Array that stores the recommended movies for each user where the userid is the index and the value is the list
# of 5 recommended movies
USER_RECOMMENDATIONS = [0] * USER_COUNT

# Calculates the absolute value of a list of ratings for a movie
def Absolute(ratings):
    absolute = 0.0
    for rating in ratings:
        absolute += pow(rating, 2)
    return absolute ** 0.5

# Calculates the dot product between two lists of movie ratings
def DotProduct(r1, r2) :
    product = 0.0
    for i  in range(0, len(r1)):
        product += (r1[i] * r2[i])
    return float(product)

# Calculates and returns the normalized form of a given lsit of movie ratings
def Normalize(ratings):
    total = 0.0
    count = 0
    for r in ratings:
        total += r
        if r > 0 :
            count = count + 1
    if count == 0:
        return ratings
    mean = total / count
    norm = deepcopy(ratings)
    for i in range(0, len(ratings)):
        if ratings[i] > 0:
            norm[i] = ratings[i] - mean
    return norm

# Calculates the CenteredCosine between two lists of movie ratings
def CenteredCosine(r1, r2):
    norm1 = Normalize(r1)
    norm2 = Normalize(r2)
    dotProduct = DotProduct(norm1, norm2)
    absolute1 = Absolute(norm1)
    absolute2 = Absolute(norm2)
    if absolute1 == 0.0:
        return 0.0
    if absolute2 == 0.0:
        return 0.0
    
    return round((dotProduct / (absolute1 * absolute2)), 5)

# Reads the data file and adds the movies with all of their ratings into an array
def MakeMovieProfiles():
    with open(RATING_FILE) as dataFile:
        csvReader = csv.reader(dataFile)
        next(csvReader)
        for row in csvReader:
            movie = int(row[1])
            user = int(row[0]) - 1
            rating = float(row[2])
            if movie < MOVIE_LIMIT:
                if movie not in MOVIE_IDS:
                    MOVIE_IDS.append(movie)
                    MOVIE_PROFILES.append((movie, [0] * USER_COUNT))
                index = MOVIE_IDS.index(movie)
                pair = MOVIE_PROFILES[index]
                movieProfile = pair[1]
                movieProfile[user] = rating

# Calculates the similarity scores of each pair of movies        
def ComputeScores():
    length = len(MOVIE_PROFILES)
    for i in range(0,length):
        PrintProgressBar(i,length)
        pair1 = MOVIE_PROFILES[i]
        for j in range(i + 1, length):
            pair2 = MOVIE_PROFILES[j]
            pair = (pair1[0], pair2[0])
            score = CenteredCosine(pair1[1], pair2[1])
            SIM_SCORES[pair] = score       

# Compares two tuples of (movieID, score) and returns true if the first one is greater
def PairGreater(p1, p2):
    if (p1[1] > p2[1]):
        return True
    elif p1[1] == p2[1]:
        if p1[0] > p2[0]:
            return True
    return False

# Returns the name of the movie in the pair of movies that is not the same as the given movie name
def OtherMovie(movie, pair):
    if movie == pair[0]:
        return pair[1]
    return pair[0]

# Retrieves the top five movies for a user
def GetUserTopFive(user):
    top5 = [(0,0), (0,0), (0,0), (0,0), (0,0)]
    for estimatedProfile in ESTIMATED_RATINGS:
        movie = estimatedProfile[0]
        ratings = estimatedProfile[1]
        estimatedRating = ratings[user]
        pair = (movie, estimatedRating)
        if PairGreater(pair, top5[4]):
            top5.pop()
            top5.append(pair)
            top5.sort(key= lambda x:x[1], reverse=True)
    return top5

# Retrieves the recommendations for each user and stores them into the array
def GetRecommendations():
    for i in range(0, USER_COUNT - 1):
        top5 = GetUserTopFive(i)
        USER_RECOMMENDATIONS[i] = list(map(lambda x : x[0], top5))

# Retrieves the top 5 neighbors for a given movie
def GetTopFiveNeighbors(movie1):
    top5 = [(0,-1), (0,-1), (0,-1), (0,-1), (0,-1)]
    for moviePair in SIM_SCORES:
        if contains(moviePair, movie1):
            movie2 = OtherMovie(movie1, moviePair)
            scorePair = (movie2, SIM_SCORES[moviePair])
            if PairGreater(scorePair, top5[len(top5) - 1]):
                top5.pop()
                top5.append(scorePair)
                top5.sort(key= lambda x:x[1], reverse=True)
    return top5

# Retrieves the neighbors for each movie and stores them into the array
def GetNeighbors():
    for pair in MOVIE_PROFILES:
        top5 = GetTopFiveNeighbors(pair[0])
        MOVIE_NEIGHBORS[pair[0]] = list(map(lambda x : x[0], top5))

# Calculates the estimated rating for a given movie
def EstimateRating(movie, user):
    totalSim = 0
    weightedAverage = 0
    neighbors = MOVIE_NEIGHBORS[movie]
    for movie2 in neighbors:
        index = MOVIE_IDS.index(movie2)
        movie2Profile = MOVIE_PROFILES[index]
        movie2Ratings = movie2Profile[1]
        rating2 = movie2Ratings[user]
        if (movie, movie2) in SIM_SCORES:
            simScore = SIM_SCORES[(movie, movie2)]
        else:
            simScore = SIM_SCORES[(movie2, movie)]
        totalSim += simScore
        weightedAverage += (rating2 * simScore)
    if totalSim == 0:
        return 0
    return weightedAverage / totalSim

# Calculates the estimated rating for each movie and stores them into the array
def EstimateRatings():
    for pair in ESTIMATED_RATINGS:
        movie = pair[0]
        ratings = pair[1]
        length = len(ratings) - 1
        for user in range(0, length):
            if ratings[user] == 0:
                ratings[user] = EstimateRating(movie, user)

# Creates the empty array for the estimated ratings
def MakeEmptyEstimateList():
    for movie in MOVIE_IDS:
        ESTIMATED_RATINGS.append((movie, [0] * USER_COUNT))

# Writes the top 5 movie recommendations for each user to a text file
def WriteOut():
    with open(OUT_FILE, 'w') as f:
        for userId in range(1, USER_COUNT):
            r = USER_RECOMMENDATIONS[userId - 1]
            f.write(str(userId) + ' ')
            for i in range(0,5):
                f.write(str(r[i]) + ' ')
            f.write('\n')

# Prints a progress bar to the console to indicate the progress of a task
def PrintProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100* (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total:
        print()

def main():
    MakeMovieProfiles()
    print("Profiles Created")
    ComputeScores()
    print("Scores Computed")
    GetNeighbors()
    print("Neighbors Found")
    MakeEmptyEstimateList()
    EstimateRatings()
    print("Missing Ratings Estimated")
    GetRecommendations()
    WriteOut()

if __name__ == '__main__':
    main()