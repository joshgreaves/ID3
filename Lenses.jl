include("./ID3.jl")
include("./Arff.jl")
include("./DataPrep.jl")

import Arff
importall ID3
importall DataPrep

# Load the data
arff = Arff.loadarff("./data/lenses.arff", should_parse=false)
x = convert(Array{Symbol}, arff.data[:, 1:4])
y = convert(Array{Symbol}, arff.data[:, 5:5])

# Split into test and validation set
train_x, train_y, test_x, test_y = splitdata(x, y)
train_x, train_y, val_x, val_y = splitdata(train_x, train_y)
feature_names = String["Age", "Spectacle-prescript", "Astigmatism", "Tear-prod-rate"]

tree = decision_tree(train_x, train_y, feature_names, validation=(val_x, val_y))

correct = 0
for i in 1:size(test_x)[1]
    prediction = classify(test_x[i, :], tree)
    println(prediction, ", ", test_y[i, 1])
    if test_y[i, 1] == classify(test_x[i, :], tree)[1]
        correct += 1
    end
end

println("Accuracy: ", correct, "/", length(test_y), " = ", correct / length(test_y))
