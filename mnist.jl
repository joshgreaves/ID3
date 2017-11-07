include("./ID3.jl")
include("./Arff.jl")
include("./DataPrep.jl")
include("./CrossValidation.jl")

import Arff
importall ID3
importall DataPrep
importall CrossValidation

using MLDatasets

# Load the data
train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

num_train = length(train_y)
num_test = length(test_y)
num_features = 28 * 28

# Binning it
threshold = 0.5

#train_x
temp = fill(:less, num_train, num_features)
for i in 1:num_train
    temp[i, reshape(train_x[:, :, i], :) .> threshold] = :more
end
train_x = temp
train_y = map(Symbol, train_y)

#test_x
temp = fill(:less, num_test, num_features)
for i in 1:num_test
    temp[i, reshape(test_x[:, :, i], :) .> threshold] = :more
end
test_x = temp
test_y = map(Symbol, test_y)

x = vcat(train_x, test_x)
y = reshape(vcat(train_y, test_y), :, 1)

feature_names = ["Pixel" * string(i) for i in 1:num_features]

train_fn(x, y) = decision_tree(x, y, feature_names, pruning=true)
function classify_fn(model, x::Matrix{Symbol}, y::Matrix{Symbol})
    num_data = size(x)[1]
    correct = 0
    for i in 1:num_data
        pred, conf = classify(x[i, :], model)
        if pred == y[i, 1]
            correct += 1
        end
    end
    return correct / num_data
end

train_acc, test_acc, trees = cross_validate(x, y, train_fn, classify_fn, n=10)
# tree = decision_tree(x, y, feature_names)
# tree2 = decision_tree(x, y, feature_names, pruning=true)
