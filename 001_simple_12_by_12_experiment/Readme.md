# 12x12 Simple Map - Count Stones after one Move

This is my initial test to train a model related to reversi.
The goal is to train the model to predict correct moves for player one and how many stones
player one will own after the move.

## Input

The input is a 12x12 standard reversi board with two players.
Input will be one hot encoded and differenciate between:
- Player One
- Player Two
- No Player

The input will be fed as a 2D data structure with 3 layers.

## Output

The desired output is a 12x12 array. Each field should contain the
number of stones player one ownes after it executed a single move on
the given field.
Fields that are no valid moves output -10.

## Goal

Get confortable with tensorflow and get a first test on how well a model can
be trained to understand the basic rules of reversi.

## Subfolders/Subprojects

Short explanations of the different subfolders/subprojects.

### data_generation

Used to execute selfplay between two ai_trival clients on a 12x12 standard map.
The games are played at search depth 1 to 10, logs are saved to a logs folder.

This data is then processed using a reversi java client and converted to csv
files containing the inptu/output needen for training the model.
