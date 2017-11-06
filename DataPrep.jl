module DataPrep

export partition, splitdata, shuffledata, getbatch, remove_unknowns

function remove_unknowns(x::Matrix{<:Any}, y::Matrix{<:Any}; unk::Any=:?)
    x = copy(x)

    classes = unique(y)
    num_data = size(x)[1]
    num_features = size(x)[2]

    bests = Dict{Symbol, Vector{Symbol}}()
    # Collect the mean for each class
    for class in classes
        indices = y[:, 1] .== class
        class_xs = x[indices, :]

        best = Vector{Symbol}(num_features)
        for i in 1:num_features
            counts = Dict{Symbol, Int64}()
            features = unique(class_xs[:, i])
            for feature in features
                counts[feature] = sum(class_xs[:, i] .== feature)
            end

            best_s = :none
            best_n = 0
            for key in keys(counts)
                if counts[key] > best_n && key != unk
                    best_s = key
                    best_n = counts[key]
                end
            end

            best[i] = best_s
            bests[class] = best
        end
    end

    # Now we have the bests, loop through the data altering it
    for i in 1:num_data
        for j in 1:num_features
            if x[i, j] == unk
                x[i, j] = bests[y[i, 1]][j]
            end
        end
    end
    
    return x
end

function partition(x::Matrix{<:Any}, y::Matrix{<:Any}, n::Integer)
    num_data = size(x)[1]
    partition_size = num_data / n
    xs = Vector{typeof(x)}(n)
    ys = Vector{typeof(y)}(n)
    last = 1
    for i in 1:n
        next = Int32(floor(partition_size * i))
        xs[i] = x[last:next, :]
        ys[i] = y[last:next, :]
        last = next + 1
    end
    return xs, ys
end

function splitdata(data::Array{<:Any, 2}, split::AbstractFloat=0.75)
    data_points = size(data)[1]
    indices = shuffle(1:data_points)
    split_index = Int32(floor(data_points * split))
    return (data[indices[1:split_index],:], data[indices[(split_index+1):end],:])
end

function splitdata(data::Array{<:Any, 2}, labels::Array{<:Any, 2},
                   split::AbstractFloat=0.75)
    data_points = size(data)[1]
    indices = shuffle(1:data_points)
    split_index = Int32(floor(data_points * split))
    return data[indices[1:split_index],:], labels[indices[1:split_index],:],
        data[indices[(split_index+1):end],:], labels[indices[(split_index+1):end],:]
end

function shuffledata(data::Array{<:Any, 2})
    num = size(data)[1]
    indices = shuffle(1:num)
    return data[indices,:]
end

function shuffledata(data::Array{<:Any, 2}, labels::Array{<:Any, 1})
    num = size(data)[1]
    indices = shuffle(1:num)
    return data[indices,:], labels[indices]
end

function shuffledata(data::Array{<:Any, 2}, labels::Array{<:Any, 2})
    num = size(data)[1]
    indices = shuffle(1:num)
    return data[indices,:], labels[indices,:]
end

function getbatch(data::Matrix{<:Any}, labels::Matrix{<:Any};
                  batch_size::Integer=32)
    num_data = size(data)[1]
    indices = shuffle(1:num_data)
    return data[indices[1:batch_size], :], labels[indices[1:batch_size], :]
end

function smoothline(data::Array{<:Number, 1}; window=10)
    result = Array{Float64, 1}(Int32(floor(size(data)[1] / 10)))
    for i in 1:length(result)
        result[i] = mean(data[((i-1)*window+1):(i*window)])
    end
    return result
end

end
