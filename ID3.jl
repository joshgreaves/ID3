module ID3

# For my implementation, I do not allow my DecisionTree to
# make decisions on real values, only on nominal values.

# TODO
# 4. DT explains why it made a choice
# 5. Do pruning

import Base.+
import Base.*
import Base.max
import Base.size

export decision_tree, classify, DecisionTree

abstract type DTNode end

struct DecisionTreeResult
    result::Dict{Symbol, <:AbstractFloat}
end
function +(l::DecisionTreeResult, r::DecisionTreeResult)
    result = DecisionTreeResult(Dict{Symbol, AbstractFloat}())
    for key in keys(l.result)
        result.result[key] = l.result[key]
    end
    for key in keys(r.result)
        if !(key in keys(result.result))
            result.result[key] = 0
        end
        result.result[key] += r.result[key]
    end
    return result
end
function *(l::AbstractFloat, r::DecisionTreeResult)
    for key in keys(r.result)
        r.result[key] = r.result[key] * l
    end
    return r
end
function max(r::DecisionTreeResult)
    max_s = :none
    max_p = 0
    for key in keys(r.result)
        if r.result[key] > max_p
            max_p = r.result[key]
            max_s = key
        end
    end
    return max_s, max_p
end

struct DecisionTree
    root::DTNode
    datanames::Vector{AbstractString}
end
function size(tree::DecisionTree)
    return size(tree.root)
end

struct DecisionNode <: DTNode
    index::Integer
    children::Dict{Symbol, DTNode}
    probs::Vector{Tuple{<:DTNode, <:AbstractFloat}}
end
function size(node::DecisionNode)
    return 1 + sum(map(size, values(node.children)))
end

struct LeafNode <: DTNode
    class::DecisionTreeResult
end
function LeafNode(x::Dict{Symbol, <:AbstractFloat})
    return LeafNode(DecisionTreeResult(x))
end
function LeafNode(x::Symbol)
    return LeafNode(Dict(x => 1.0))
end
function size(node::LeafNode)
    return 1
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

function validate(x::Matrix{Symbol}, y::Matrix{Symbol}, node::DTNode)
    correct = 0
    num_data = size(x)[1]
    for i in 1:num_data
        pred = classify(x[i, :], node)
        if y[i, 1] == max(pred)[1]
            correct += 1
        end
    end
    return correct / num_data
end

function decision_tree(x::Matrix{Symbol}, y::Matrix{Symbol},
                     names::Vector{<:AbstractString};
                     validation::Tuple{Matrix{Symbol}, Matrix{Symbol}}=(Matrix{Symbol}(0, 0), Matrix{Symbol}(0, 0)))
    return DecisionTree(create_tree_inner(x, y, size(x)[2], val=validation),
                        names)
end

function create_tree_inner(x::Matrix{Symbol}, y::Matrix{Symbol},
                           remaining_splits::Integer;
                           val::Tuple{Matrix{Symbol}, Matrix{Symbol}}=(Matrix{Symbol}(0, 0), Matrix{Symbol}(0, 0)))
   # Get the number of features
   y_counts = count(y)
   y_keys = keys(y_counts)
   num_classes = length(y_keys)
   num_data = length(y)

   # Base case: There is only one class remaining
   if length(y_keys) == 1
       return LeafNode(first(y_keys))
   end

    # Base case: no splits remaining
    if remaining_splits == 0
        result = DecisionTreeResult(Dict{Symbol, Float64}())
        for key in keys(y_counts)
            result.result[key] = y_counts[key] / num_data
        end
        return LeafNode(result)
    end

    # Calculate Information
    info = 0
    for key in y_keys
        count = y_counts[key]
        ratio = count / num_data
        info -= ratio * log2(ratio)
    end
    # println("Information: ", info)

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
    # println("Best index is: ", best_index, " with information gain of ", max_gain)

    # Split on that data point
    node = DecisionNode(best_index, Dict{Symbol, DTNode}(),
                        Vector{Tuple{<:DTNode, <:AbstractFloat}}())
    split_keys = unique(x[:, best_index])
    for key in split_keys
        indices = x[:, best_index] .== key
        child = create_tree_inner(x[indices, :], y[indices, :],
                                  remaining_splits - 1, val=val)
        node.children[key] = child
        prob = sum(indices) / num_data
        push!(node.probs, (child, prob))
    end

    # If using a validation set, test to see if accuracy increases
    if length(val[1]) > 0
        acc = validate(val[1], val[2], node)

        most = 0
        for key in keys(y_counts)
            ratio = y_counts[key] / num_data
            if ratio > most
                most = ratio
            end
        end

        println(most, ", ", acc)
        if most > acc
            println("Creating leaf node instead")
            result = DecisionTreeResult(Dict{Symbol, Float64}())
            for key in keys(y_counts)
                result.result[key] = y_counts[key] / num_data
            end
            return LeafNode(result)
        end
    end

    return node
end

function classify(x::Vector{Symbol}, node::DecisionTree)
    result = classify(x, node.root)
    return max(result)
end

function classify(x::Vector{Symbol}, node::DecisionNode)
    if x[node.index] in keys(node.children)
        return classify(x, node.children[x[node.index]])
    else
        lst = [tup[2] * classify(x, tup[1]) for tup in node.probs]
        return reduce((a, b) -> a + b, lst)
    end
end

function classify(x::Vector{Symbol}, node::LeafNode)
    return node.class
end

end
