"""
    predictions, cm, miss = prediction(bst::Booster, data::Dmatrix; obj="binary:logistic", num_class::Int=0, num_obs::Int=0,  output_margin::Bool = false, ntree_limit::Integer = 0, training::Integer=0, ids::Array=[] )

Predict a data set using an established model and analyze the results. Set the objective (obj) to binary:logistic, multi:softmax, or multi:softprobs -- based on how you created the model -- and set the num_class and num_obs to match the input matrix. 

If obj=multi:softprobs is used, the difference between the highest and next best probability for each observation and the N-class probability matrix will be included in the output. If the input matrix contains labels, a confusion matrix and error rate comparing known and predicted labels are returned. If provided, an ordered list of ids will be appended to the prediction output. The order of ids must match the dmatrix! 
"""
function prediction(bst::Booster, data::DMatrix; obj::String="binary:logistic", num_class::Int=2, num_obs::Int64, ntree_limit::Int64=0, training::Int=0, ids::Array=[])

    if obj == "multi:softprobs"
        error("\n\nNo objective called multi:softprobs. Did you mean multi:softprob\n")
    end

    predout = []
    
    if length(obj) == 0 || num_class == 0 || num_obs == 0
        error("\nPlease provide the model objective (e.g. obj=\"multi:softprob\"),\nthe number of classes (e.g. num_class=5) and the number of observations\n(e.g. num_obs=14572) for analysis.\n\n")
    end

    label = Int.(get_info(data, "label"))
    label_count = length(label)

    if label_count > 0
        println("Input dmatrix contains $label_count labels ...")
        if label_count != num_obs
            error("\nThe number of labels ($label_count) is inconsistent with the user-specified num_obs ($num_obs)\n\n")
        end
        nclass = length(unique(label))
        if nclass != num_class
            error("\nThe number of classes ($nclass) is inconsistent with the user-specified num_class ($num_class)\n\n")
        end

        if minimum(label) == 0
            print("Recoding 0-based DMatrix labels indexing to 1-based indexing ...\n")
            label = label .+ 1
        end
    end
    
    println("Running predictions using bst model ...")

    xgbpred = XGBoost.predict(bst::Booster, data::DMatrix; ntree_limit=ntree_limit, training=training)
    probs = permutedims(reshape(Float64.(xgbpred), num_class, num_obs)) #reshape to two col output

    cm = nothing; err = nothing; dmarg_norm = nothing;

    function compare(label, pred)

        if length(label) !== length(pred)
            error("Length of labels doesn't match number of predictions")
        end

        nobs = length(label)
        missclass = label .!= pred
        err = round(sum(missclass) / nobs, digits=4)
        cm, ev = make_confusion_matrix(label, pred)

        return (cm, err)

    end


    if obj == "multi:softprob"

        pred = [argmax(probs[i,:]) for i in 1:size(probs,1)]
        sprobs = sort(probs, dims=2, rev=true)
        diffprob = round.( (sprobs[:,1] .- sprobs[:,2]) .* 100, digits=1)

        if label_count > 0
            println("Comparing labels to predicted labels...")
            cm, err = compare(label, pred)
            pred = hcat(label, pred)
            println("Observed error rate: $err")
        end
        
        predout = hcat(pred, diffprob, probs)

        println("Multi:softprob prediction columns are: [ids], [label], prediction, bestprobdiff, N-class_probabilities...")

    elseif obj == "multi:softmax"

        if label_count > 0
            println("Comparing labels to predicted labels...")
            cm, err = compare(label, pred)
            pred = hcat(label, pred)
        end

        predout = pred

    # elseif obj == "binary:logistic"

    #     binpred = pred .> 0.5
    #     cm = make_confusion_matrix((label .+ 1.0), (binpred .+ 1.0))       
    #     missclass = label .!= binpred
    #     err = sum(missclass) / nobs
    #     pred  = hcat(label, binpred, pred)

    # else

    #     println("Analysis currently limited to binary logistic, multi:softmax, and multi:softprobs.")

    # end

#    if length(ids) > 1
#        hcat(ids,pred)
#    end

#    if length(label) == 0
#        return pred
#    else
#        success = round((1 - err) * 100, digits=2)
#        printstyled("\nModel classification success: $success percent.\n\n", color=:green)
#        return(pred, cm, missclass)
    end

    if length(ids) > 0        # add labels if provided
        predout = hcat(ids, predout)
    end

    return predout, cm, err, dmarg_norm

end

