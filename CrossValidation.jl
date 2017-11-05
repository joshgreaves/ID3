module CrossValidation
# A simple package for running cross validation for a particular
# learning algorithm and dataset

include("./DataPrep.jl")
import DataPrep.partition
import DataPrep.shuffledata

export cross_validate

function cross_validate(x::Any, y::Any, train_fn::Function,
                        classify_fn::Function; n::Integer=10)
    # Shuffle, then split data into n pieces
    xs, ys = shuffledata(x, y)
    xs, ys = partition(xs, ys, n)

    acc = Vector{Float64}(n)
    for i in 1:n
        indices = [true for j in 1:n]
        indices[i] = false
        trained = train_fn(vcat(xs[indices]...), vcat(ys[indices]...))
        acc[i] = classify_fn(trained, xs[i], ys[i])
    end
    return mean(acc)
end

end
