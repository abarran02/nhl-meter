# NHL Meter

### Abstract

The objective of this project is to provide a system that can provide live estimations about the outcomes of an National Hockey League (NHL) game. Using statistical and machine learning methods, we were able to create a model that can ???

### Introduction

Inspired by visualizations like the Win Probability graph provided by ESPN, we sought out to create a similar metric based on Hockey play by play data sourced from the NHL. We would like to use information about the state of the game, such as the current score, 'momentum,' and other factors that can be derived from the data to determine the odds of the home team winning at any arbitrary point in the game. Sports are inherently unpredictable, however we can still make educated guesses the chances of specific outcomes based on information we have.

### Problem Definition

Given the Play-By-Play data for a specific NHL match up to a certain time point, what is the probability of the Home or Away team winning? With the rise of the sports betting industry, these types of endeavors have grown in popularity and interest on both the bettor and bookkeepers sides, as bettors seek new edges and bookkeepers seek to set more accurate lines. Estimations of NHL outcomes are not new, but from our research there is no investigation into the dynamic estimation as the game is played.

### Implementation / Analysis

Our data is sourced from the official NHL API. This allows you to get historical play-by-play data from games since the 2007 season using HTTP requests. Using a Game ID, you can download all the games for a specific season. Thankfully, there exists an existing python implementation that scrapes this data. 'hockey-scraper' by harryshomer on pypi allows you to input a season or a date range, and will download the season information then all associated plays with each game in that season. This is a somewhat large amount of data, taking multiple days to download due to API rate limiting. Post-parquet compression, the data including play-by-play and shifts (documenting when a player comes on or off the ice) took up approximately half a gigabyte.

Analyzing the format of the data, we can view useful information about the game. A "play" is recorded whenever a notable event occurs. These events can take forms like a shot on goal, a sucessful goal, the result of a face off, penalties, or other associated game events. For every play, all players that were involved are listed, as well as all other players on the ice at the time. Spatial data is included, such as the X and Y coordinates on the rink where the event happened, and of course information about the game itself such as the current score, the teams, and the time remaining in the period. 
