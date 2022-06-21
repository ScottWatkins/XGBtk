"""
    dmat, labels = libsvm2dmat(filename::String, (rows::Int, cols::Int))

Read data in libsvm format and return a DMatrix and the labels. You must provide the shape of the expected output matrix as a tuple (rows, columns)
Example libsvm file:
1 101:1.2 102:0.03
0 1:2.1 10001:300 10002:400
0 0:1.3 1:0.3
1 0:0.01 1:0.3
0 0:0.2 1:0.3

First column contains labels as 0,1 or [0,1] probabilities. The remaining columns are feature_index:feature_value pairs representing a sparse matrix.
"""
function libsvm2matrix(fname::String, shape)

    dmx = zeros(Float32, shape)
    label = Float32[]
    fi = open(fname, "r")
    cnt = 1

    for line in eachline(fi)
        line = split(line, " ")
        push!(label, parse(Float64, line[1]))
        line = line[2:end]
        
        for itm in line
            itm = split(itm, ":")
            dmx[cnt, parse(Int, itm[1]) + 1] = parse(Int, itm[2])
        end

        cnt += 1

    end

    close(fi)

    return (dmx, label)

end
