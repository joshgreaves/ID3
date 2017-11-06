module CrossValidation
# A simple package for running cross validation for a particular
# learning algorithm and dataset

include("./DataPrep.jl")

importall DataPrep

export cross_validate

function cross_validate(x::Any, y::Any, train_fn::Function,
                        classify_fn::Function; n::Integer=10)
    # Shuffle, then split data into n pieces
    xs, ys = shuffledata(x, y)
    xs, ys = partition(xs, ys, n)

    train_acc = Vector{Float64}(n)
    test_acc = Vector{Float64}(n)
    for i in 1:n
        indices = [true for j in 1:n]
        indices[i] = false
        train_y = vcat(ys[indices]...)
        train_x = vcat(xs[indices]...)
        trained = train_fn(train_x, train_y)
        train_acc[i] = classify_fn(trained, train_x, train_y)
        test_acc[i] = classify_fn(trained, xs[i], ys[i])
    end
    return train_acc, test_acc
end

end
