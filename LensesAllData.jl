include("./ID3.jl")
include("./Arff.jl")

import Arff
importall ID3

# Load the data
arff = Arff.loadarff("./data/lenses.arff", should_parse=false)
x = convert(Array{Symbol}, arff.data[:, 1:4])
y = convert(Array{Symbol}, arff.data[:, 5:5])
feature_names = String["Age", "Spectacle-prescript", "Astigmatism", "Tear-prod-rate"]

tree = decision_tree(x, y, feature_names)

# Induced Tree:
(1) Split on Tear-prod-rate -> (2) normal, (3) reduced,
(2) Split on Astigmatism -> (4) yes, (5) no,
(3) classify: none 1.0,
(4) Split on Spectacle-prescript -> (6) hypermetrope, (7) myope,
(5) Split on Age -> (8) young, (9) presbyopic, (10) pre-presbyopic,
(6) Split on Age -> (11) young, (12) presbyopic, (13) pre-presbyopic,
(7) classify: hard 1.0,
(8) classify: soft 1.0,
(9) Split on Spectacle-prescript -> (14) hypermetrope, (15) myope,
(10) classify: soft 1.0,
(11) classify: hard 1.0,
(12) classify: none 1.0,
(13) classify: none 1.0,
(14) classify: soft 1.0,
(15) classify: none 1.0,
