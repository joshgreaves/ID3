module ID3

# For my implementation, I do not allow my DecisionTree to
# make decisions on real values, only on nominal values.

export create_tree

abstract type DTNode end

struct DecisionNode <: DTNode
    index::Integer
    children::Dict{Symbol, DTNode}
end

struct LeafNode <: DTNode
    class::Symbol
end

struct DecisionTree
    root::DTNode
    datanames::Vector{AbstractString}
end

function count(x::Matrix{Symbol})
    result = Dict{Symbol, Int64}()
    for i in 1:length(x)
        if !(x[i, 1] in keys(result))
            result[x[i, 1]] = 0
        end
        result[x[i, 1]] += 1
    end
    return result
end

function calc_gain(x::Vector{Symbol}, y::Vector{Symbol})
    num_data = length(x)

    # Loop through the data, collecting counts of features, and corresponding labels
    x_keys = unique(x)
    y_keys = unique(y)
    x_counts = Dict{Symbol, Int64}([x => 0 for x in x_keys])
    y_counts = Dict{Symbol, Int64}([x => 0 for x in y_keys])
    comb_counts = Dict{Tuple{Symbol, Symbol}, Int64}([(u, v) => 0 for u in x_keys, v in y_keys])

    for i in 1:num_data
        x_counts[x[i]] += 1
        y_counts[y[i]] += 1
        comb_counts[(x[i], y[i])] += 1
    end

    info = 0
    for key in x_keys
        key_count = x_counts[key]
        ratio = key_count / num_data
        temp_info = 0
        for y_key in y_keys
            comb_ratio = comb_counts[(key, y_key)] / key_count
            if comb_ratio != 0
                temp_info -= comb_ratio * log2(comb_ratio)
            end
        end
        info += ratio * temp_info
    end
    return info
end

function create_tree(x::Matrix{Symbol}, y::Matrix{Symbol},
                     names::Vector{<:AbstractString})
end

function create_tree_inner(x::Matrix{Symbol}, y::Matrix{Symbol})
    # Base case: all data points are the same
    # TODO
    # Base case: There is only one class remaining
    y_counts = count(y)
    y_keys = keys(y_counts)
    if length(y_keys) == 1
        return LeafNode(first(y_keys))
    end

    # Get the number of features
    num_classes = length(y_keys)
    num_data = length(y)

    # Calculate Information
    info = 0
    for key in y_keys
        count = y_counts[key]
        ratio = count / num_data
        info -= ratio * log2(ratio)
    end
    println("Information: ", info)

    # Loop through each feature
    best_index = 0
    max_gain = 0
    for i in 1:size(x)[2]
        gain = info - calc_gain(x[:, i], y[:, 1])
        if gain > max_gain
            max_gain = gain
            best_index = i
        end
    end
    println("Best index is: ", best_index, " with information gain of ", max_gain)

    # Split on that data point
    node = DecisionNode(best_index, Dict{Symbol, DTNode}())
    split_keys = unique(x[:, best_index])
    for key in split_keys
        indices = x[:, best_index] .== key
        node.children[key] = create_tree_inner(x[indices, :], y[indices, :])
    end

    return node
end

function classify(x::Vector{Symbol}, node::DecisionNode)
    return classify(x, node.children[x[node.index]])
end

function classify(x::Vector{Symbol}, node::LeafNode)
    return node.class
end

end
