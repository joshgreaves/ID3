include("ID3.jl")
include("Arff.jl")
include("DataPrep.jl")
include("CrossValidation.jl")

import Arff
importall ID3
importall DataPrep
importall CrossValidation

# Load the data
mappings = Dict("'y'" => :y, "'n'" => :n, :? => :?,
                "'democrat'" => :democrat, "'republican'" => :republican)
arff = Arff.loadarff("./data/voting.arff", mappings=mappings, should_parse=false)
x = convert(Array{Symbol}, arff.data[:, 1:end-1])
y = convert(Array{Symbol}, arff.data[:, end:end])
feature_names = String["handicapped-infants",
                       "water-project-cost-sharing",
                       "adoption-of-the-budget-resolution",
                       "physician-fee-freeze",
                       "el-salvador-aid",
                       "religious-groups-in-schools",
                       "anti-satellite-test-ban",
                       "aid-to-nicaraguan-contras",
                       "mx-missile",
                       "immigration",
                       "synfuels-corporation-cutback",
                       "education-spending",
                       "superfund-right-to-sue",
                       "crime",
                       "duty-free-exports",
                       "export-administration-act-south-africa"]

train_fn(x, y) = decision_tree(remove_unknowns(x, y), y, feature_names, pruning=true)
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

train_acc, test_acc = cross_validate(x, y, train_fn, classify_fn, n=10)
tree = decision_tree(remove_unknowns(x, y), y, feature_names)
tree2 = decision_tree(remove_unknowns(x, y), y, feature_names, pruning=true)
