# Understanding-Human-Sentiments-using-AI-Based-Text-Analysis-Techniques
This project show the sentiment analysis of text data using NLP and Dash. I used Amazon reviews dataset to train the model and further scrap the reviews from Etsy.com in order to test my model.

## What all you need for this project?
1.Python3

2.Spyder (64 bit)

3.[Amazon Dataset](http://deepyeti.ucsd.edu/jianmo/amazon/)

## How I made this project?
1.This project has been built using Python3 to help predict the sentiments with the help of Machine Learning and an interactive dashboard to test reviews.

2.To start, I downloaded the dataset and extracted the JSON file.

3.Next, I took out a portion of 7,92,000 reviews equally distributed into chunks of 24000 reviews using pandas.

4.The chunks were then combined into a single CSV file called balanced_reviews.csv.

5.This balanced_reviews.csv served as the base for training my model which was filtered on the basis of review greater than 3 and less than 3.

6.Further, this filtered data was vectorized using TF_IDF vectorizer.

7.After training the model to a 90% accuracy, the reviews were scrapped from Etsy.com in order to test our model.

8.Finally, I built a dashboard in which we can check the sentiments based on input given by the user or can check the sentiments of reviews scrapped from the website.

## What do you mean by CountVectorizer?
Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of term/token counts. It also enables the ​pre-processing of text data prior to generating the vector representation. This functionality makes it a highly flexible feature representation module for text.

CountVectorizer creates a matrix in which each unique word is represented by a column of the matrix, and each text sample from the document is a row in the matrix. The value of each cell is nothing but the count of the word in that particular text sample.

## What do you mean by TF-IDF Vectorizer?
TF-IDF is an abbreviation for Term Frequency Inverse Document Frequency. This is very common algorithm to transform text into a meaningful representation of numbers which is used to fit machine algorithm for prediction.
It works by increasing proportionally to the number of times a word appears in a document, but is offset by the number of documents that contain the word. So, words that are common in every document, such as this, what, and if, rank low even though they may appear many times, since they don’t mean much to that document in particular.

## What do you mean by Plotly Dash?
It is an open source library for creating interactive web-based visualizations. 

The library is built on top of well established open source frameworks like flask for serving the pages and React.js for the javascript user interface. 
The unique aspect of this library is that you can build highly interactive web application solely using python code. 

Having knowledge of HTML and javascript is useful but certainly not required to get a nice display with a minimal amount of coding.

## What do you mean by Web Scrapping?
Web scraping is an automated method used to extract large amounts of data from websites. The data on the websites are unstructured. Web scraping helps collect these unstructured data and store it in a structured form.

## Running the project
### *Step-1*:
  1.Download the dataset and extract the JSON data in your project folder.
  2.Run the  ***Dataextraction.py*** file .This will extract data from the JSON file into equal sized chunks and then combine them into a single CSV file called ***balanced_reviews.csv***.
### *Step-2*:
  1.Run the ***data_cleaning_preprocessing_and_vectorizing.py*** file. This will clean and filter out the data.
  2.Next the filtered data will be fed to the **TF-IDF Vectorizer** and then the model will be pickled in a ***pickle_model.pkl*** file and the Vocabulary of the trained model will be stored as ***feature.pkl***. Keep these two files in your project folder itself.
### *Step-3*:
  1.Then run the ***etsy_scrapper.py*** file.You can adjust the range of pages and product to be scrapped as it takes a long long time for the entire data to be scrapped.
   A small sized data is sufficient to check the accuracy of our model.
  2.The scrapped data will be stored in csv as well as db file.
### *Step-3*:
  1.Finally, run the ***UI.py*** file that will start up the Dash server and we can check the working of our model either by typing or either by selecting the preloaded scrapped reviews.
  
 **NOTE- For the favicon , make sure that you make a separate folder with the name assets in your project folder and put the favicon image to this .There's no code for this , your browser will automatically load this. **

