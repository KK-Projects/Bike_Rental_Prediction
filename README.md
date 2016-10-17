## Bike Rental Project

This is a school project to forecast the bike rental using Machine learning algorithms.
The project is subscribed in Kaggle projects competition. The description is the following :

You are provided bike rental data over a period of two years. The training set contains the first 19 days of each month.
The test set contains days from the 20th to the end of the month.
The goal is to predict the number of bikes that were rented for each time point of the test set.

###File descriptions

- train.csv - the training set
- test.csv - the test set
- my_submission.csv - a sample submission file in the correct format

###Data fields

- instant : id of the record (integer)
- dteday : date of the record (yr-month-day)
- season : season of the record (integer, 1-4)
        - 1: Spring
        - 2: Summer
        - 3: Fall
        - 4: Winter
- yr - year of the record(integer, 0-1)
        - 0: 2011
        - 1: 2012
- mth : month of the record (integer, 1-12)
- hr : hour of the record (integer, 0-23)
- holiday : whether the day is a holiday or not (integer, 0-1)
- weekday : day of the week (integer, 1-7)
- workingday : whether the day is a working day (neither holiday nor weekend) or not (integer, 0-1)
- wheathersit :  weather situation (integer, 1-4)
        - 1: clear, few clouds, or partly cloudy
        - 2: mist (no precipitation)
        - 3: light rain or light snow
        - 4: heavy rain, hail, or snow.
- temp : temperature in Celsius, normalized by dividing by the highest temperature recorded over these two years (float, [0, 1]).
- atemp : apparent temperature in Celsius, normalized by dividing by the highest apparent temperature over these two years (float, [0, 1]). Apparent temperature quantifies the temperature perceived by humans, combining wind chill, humidity, and actual temperature.
- hum : percentage of humidity (float, [0, 1]).
- windspeed : wind speed, normalized by dividing by the highest speed recorded over these two years (float, [0, 1])
- cnt : number of bikes rented (integer) Train set only, value to predict.
