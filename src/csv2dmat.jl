"""
     dmatrix, features, ids = csv2dmat(filename::String; weight="none", file_has_ids=true, file_has_labels=true)

Read a csv file containing numeric observations in rows, features in cols, and optional ids and/or known labels in the first column(s). All columns must be labelled. Return a sparse XGBoost DMatrix and an array of features. All input should be numeric and labels should be  0,1, ..., N.

Class weights can be specified in a two-column text file where each line contains class [0 ... N-class] and weight. Use weight="filename.txt" to use a file, weight="equal" to apply equal weighting to all classes or weight="none" (default) for no weighting.

Example input (id and label columns are optional):\n 
id,label,trait1,trait2\\
id1,0,1,0\\
id5,2,0,0\\
id7,1,0,1\\
\\
\n

If you are preparing a standard data table for prediction using a prebuilt model, set file_has_ids=true and file_has_labels=false to create the Dmatrix.

Example input:\n 
ID,trait1,trait2\\
M1,1,0\\
X2,0,0\\
SAM3,0,1\\
\\
\n
"""
function csv2dmat(filename::String; weight="none", data_only::Bool=false, file_has_ids::Bool=true, file_has_labels::Bool=true)

    weight=weight
    features = []
    ids = []

    printstyled("File has ids is set as $file_has_ids...\n", color=:yellow)
    printstyled("File has labels is set as $file_has_labels...\n", color=:yellow)

    println("Converting $filename to DMatrix format...")

    if data_only == true
        
        d = readdlm(filename, ',', Int)
        println("Processing a labelless $(size(d)) data matrix...")
        d = Matrix(d)
        xsp = sparse(d)
        dtrain =  DMatrix(xsp)
        println("Created label-free data matrix.")
        
        return dtrain, features, ids
        
    end

    df = CSV.read(filename, DataFrame, normalizenames=true, delim=",")

    if file_has_ids == true && file_has_labels == true
        
        ids = df[:,1]
        df = df[:,2:end]
        println("Removed the ids from column one...\nContinuing formatting... ")
        
    elseif file_has_ids == true && file_has_labels == false

        ids = df[:,1]
        df = df[:, 2:end]
        println(df)
        features = Vector(names(df))
        d = Matrix(df)
        xsp = sparse(d)
        dtrain =  DMatrix(xsp)
        println("Created label-free Dmatrix for prediction.\n\n")

        return dtrain, features, ids

    elseif file_has_ids == false && file_has_labels == false
        
        features = Vector(names(df))
        d = Matrix(df)
        xsp = sparse(d)
        dtrain =  DMatrix(xsp)
        println("For faster performance create an unlabelled disk file and use data_only=true.")
        println("Created label-free Dmatrix  with no ids for prediction.\n\n")

        return dtrain, features, ids

    end
    
    x = Matrix(df[!,2:end])
    y = df[!,1]
    sx = size(x); uy = sort(unique(y))
    println("Input features matrix size = $sx\nInput labels are coded as: $uy")
    
    if minimum(y) == 1
        y .- 1
        println("Minimum label value was 1. Recoded 1-based labels to 0-based labels.")
    end

    xsp = sparse(x)
    
    if weight == "equal" || isfile(weight)
        
        ynz = y .+ 1  #prevent divid by 0
        
        if weight == "equal"
            yc = countmap(ynz)
            ycc = values(yc) ./ minimum(values(yc))
            ycc = inv.(ycc)
            wd = Dict(zip(keys(yc), ycc))
        end
        
        if isfile(weight)
            println("Using weight values from file $weight ...")
            wd = Dict()
            open(weight) do f
                for i in eachline(f)
                    j = split(i, r"\t|\s+|\,")
                    c = parse(Int8, j[1] ) + 1
                    w = parse(Float64, j[2] )
                    wd[c] = w
                end
            end
        end

        if minimum(keys(wd)) > 1  #check user input, still recoded to 1-based
            error("Labels in $weight should start with zero!")
        end

        wv = [wd[i] for i in ynz]
        
        print("Applying weights as follows (label => weight): ")
        for (k,v) in sort(wd)
            k = k-1
            print("$k => $v, ")
        end
        println()

        dtrain =  DMatrix(xsp, label = y, weight = wv)

    else

        dtrain =  DMatrix(xsp, label = y)

    end

    features = Vector(names(df)[2:end])

    return dtrain, features, ids

end
