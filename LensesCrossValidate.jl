include("./ID3.jl")
include("./Arff.jl")
include("./DataPrep.jl")
include("./CrossValidation.jl")

import Arff
importall ID3
importall DataPrep
importall CrossValidation

# Load the data
arff = Arff.loadarff("./data/lenses.arff", should_parse=false)
x = convert(Array{Symbol}, arff.data[:, 1:4])
y = convert(Array{Symbol}, arff.data[:, 5:5])
feature_names = String["Age", "Spectacle-prescript", "Astigmatism", "Tear-prod-rate"]

train_fn(x, y) = decision_tree(x, y, feature_names)
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

acc = cross_validate(x, y, train_fn, classify_fn, n=24)
println(acc)
