"""
    cm, ev = make_confusion_matrix(true::Array, predicted::Array)

Create a confusion matrix for true vs. predicted sample values. Return the confusion matrix (cm) and a vector marking prediction errors (ev).
# Example
```
  x = [1 1 3 2 3 1]
  y = [1 2 3 2 1 1]

  cm, ev = make_confusion_matrix(x,y)

cm
3×3 Matrix{Int64}:
 2  1  0
 0  1  0
 1  0  1

ev
1×6 Matrix{Int64}:
 0  1  0  0  1  0

```

"""
function make_confusion_matrix(x::Array, y::Array)

    if length(x) !== length(y)
        error("Input vectors must be the same size!")
    end

    if typeof(x[1]) == String || typeof(y[1]) == String
        error("Input must currently be numeric!")
    end

    ux = sort(unique(x)); uy = sort(unique(y))

    if typeof(x) == Array{Float32, 1} || typeof(x) == Array{Float64, 1}
        x=Int64.(x)
        y=Int64.(y)
    end

    println("Values in labels and prediction sets are now:\nLabels: $ux\nPredictions: $uy\n")

    if minimum(x) == 0
        print("Recoding 0-based true labels indexing to 1-based indexing...\n")
        x = x .+ 1
    end    

    if minimum(y) == 0
        print("Recoding 0-based predicted labels indexing to 1-based indexing...\n")
        y = y .+ 1
    end    

    v = x .== y
    ev = replace(v, 0=>1, 1=>0)
    sev = sum(ev)
    cm = zeros(length(ux), length(uy))

    for i in 1:length(x)
            cm[x[i], y[i]] += 1
    end

    cm = convert.(Int64, cm)
    
    return(cm, ev)

end
