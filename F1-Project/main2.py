import model2 as m #importing loads the data, splits the years, fits the indexes/normalising and builds the model + loader

#1. train on 2018-2023
m.training()

#2. eval on 2024 - lap by lap with its own predictions + predicted standings for each race
m.run_races(m.model, m.val, 'eval')

#3. test on 2025 - same as eval
m.run_races(m.model, m.test, 'test')
