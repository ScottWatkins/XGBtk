"""
    softprobs_bayes_correction(predictions::Matrix)

Apply a naive Bayes correction to the multi:softprob prediction probabilities and re-predict the predicted labels. The input is the table returned by the runCVtrain function with the following columns: [truelabels, predicted_labels, mxn_prob_matrix]. Output is updated to a table with columns: [truelabels, predicted_labels, bayes_predicted_labels bayes_corrected_mxn_prob_matrix].
"""
function softprobs_bayes_correction(predmat)

    ac = countmap(predmat[:,1])
    e = [ac[k]/size(predmat,1) for k in sort(collect(keys(ac)))]
    m = predmat[:, 3:end]
    bc = zeros(size(m))

    for i in 1:(size(m,1))
        denom = m[i,:] ⋅ e #calc denom as dot product
        for j in 1:size(m,2)
            bc[i,j] = (m[i,j] * e[j]) / denom
        end
    end

    newpreds = [ argmax(bc[i,:]) for i in 1:size(bc,1)]
    bc = hcat(predmat[:,1:2], newpreds, bc)

    return(bc)
end
