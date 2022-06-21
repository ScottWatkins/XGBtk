"""
    dtrain, dtest, param = scale_pos(dtrain::DMatrix, dtest::DMatrix, param)

Scale positive weights (1) for an unbalanced dataset (count(0) >> count(1)).
"""
function scale_pos(dtrain::DMatrix, dtest::DMatrix, param)  
    label = get_info(dtrain, "label")                       
    ratio = sum(label .== 0) / sum(label .== 1)
    param["scale_pos_weight"] = ratio
    return dtrain, dtest, param
end

