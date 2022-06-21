"""
    bst = build_model(dtrain::DMatrix, 100; num_class=2, param=param, metrics=metrics, seed=0)

Build a XGBoost model from a DMatrix with optimized parameters and save the model to disk.\\

Pass parameters as an array of pairs or a dictionary:\\
Dict(\\
"max_depth"=>6,\\
"eta"=>0.3,\\
"alpha"=>0.0,\\
"gamma"=>0.0,\\
"lambda"=>1.0,\\
"min_child_weight"=>1.0,\\
"max_delta_step"=>0.0\\
)

Use the runCVtrain function to find the best parameters.
Additional options: watchlist = [], feval = Union{}, group = [], kwargs...
"""
function build_model(dtrain::DMatrix, rounds::Int64; num_class=2, param=param, metrics=metrics, seed=0)

    #Required input form: xgboost(dtrain, 100; num_class=5, param=param, metrics=metrics, seed=0)

    bst = xgboost(dtrain, rounds; num_class=num_class, metrics=metrics, param=param, seed=seed)
    name = join(["bst", string.(rounds), "model"], "." )
    save(bst, name)
    println("Wrote booster to disk file: $name\n")

    return(bst)

end
