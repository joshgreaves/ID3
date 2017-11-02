include("./ID3.jl")
include("./Arff.jl")

import Arff
importall ID3

# Load the data
arff = Arff.loadarff("./data/cars.arff", should_parse=false)
x = convert(Array{Symbol}, arff.data[:, 1:end-1])
y = convert(Array{Symbol}, arff.data[:, end:end])
feature_names = String["Age", "Spectacle-prescript", "Astigmatism", "Tear-prod-rate"]

tree = ID3.create_tree_inner(x, y)

for i in 1:size(x)[1]
    println(ID3.classify(x[i, :], tree), ": ", y[i, 1])
end
