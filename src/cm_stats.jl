"""
    cm_stats(cm::Matrix; percent=false)

Get typical statistics from a confusion matrix. If matrix is larger than 2x2, analyze each variable class separately against all others. Set percent=true to output percentages.
"""
function cm_stats(cm::Matrix; percent=false)
    
    size(cm,1) != size(cm,2) ? error("Input matrix is not square!") : println("Reading input matrix ... size = $(size(cm))\n")

    cmd = diag(cm); cml = tril(cm, -1); cmu = triu(cm, 1);
    cms = sum(cm); cmds = sum(diag(cm));    
    acc_all = round(sum(cmd)/cms, digits=4);
    err_all = round(1 - acc_all, digits=4)

    printstyled("Total samples: $cms\n", color=:green)
    printstyled("Overall accuracy: $acc_all\n", color=:green)
    printstyled("Overall error rate: $err_all\n\n\n", color=:green)

    println("Table variables order correspond to input matrix order.")

    r = zeros(size(cm,1), 12) 

    for i in 1:size(cm,1)             #analysis by class

        tp = diag(cm)[i]
        tn = cms - ( sum(cm[:,i]) + sum(cm[i,:]) - tp )
        fp = sum(cm[:,i]) - tp
        fn = sum(cm[i,:]) - tp

        acc = (tp + tn) / (tp + tn + fp + fn);  r[i,1] = round(acc, digits=4);
        err = 1 - acc;          r[i,3] = round(err, digits=4);
        sen = tp / (tp + fn);   r[i,4] = round(sen, digits=4);        #sensitivity,tp rate, recall
        spec  = tn / (tn + fp); r[i,5] = round(spec, digits=4);       #specificity,selectivity, true neg rate
        bal_acc = (sen + spec) / 2; r[i,2] = round(bal_acc, digits=4);
        ppv = tp / (tp + fp);   r[i,6] = round(ppv, digits=4);        #pos. pred. value, precision 
        npv = tn / (fn + tn);   r[i,7] = round(npv, digits=4);        #neg. pred. value
        fdr = fp / (tp + fp);   r[i,8] = round(fdr, digits=4);        #false discovery rate
        fpr = fp / (fp + tn);   r[i,9] = round(fpr, digits=4);        #false pos. rate, fall out
        fnr = fn / (tp + fn);   r[i,10] = round(fnr, digits=4);       #false neg rate, miss rate
        prev = (tp + fn) / (tp + tn + fp + fn); r[i,11] = round(prev, digits=4);
        f1 = (2*tp) / (2*tp + fp + fn); r[i,12] = round(f1, digits=4); #harmonic mean of sen and ppv

    end

    labels = ["Accuracy","BalancedAccuracy","ErrorRate","Sensitivity","Specificity","PosPredValue","NegPredValue","FDR","FalsePosRate(α)","FalseNegRate(Β)","Prevalence","F1_score"]

    r = DataFrame(r', :auto)
    insertcols!(r, 1, :Statistic=>labels)

    if size(r,2) == 3
        select!(r, Not(:x2))
        rename!(r, :x1=>:Value)
        insertcols!(r, 3, :Percent=>round.(r[:,2] * 100, digits=1))
    end

    if percent == true
        rp = r[:, 2:end] .* 100
        r = insertcols!(rp, 1, :Statistic=>r[:,1])
    end

    return(r)

end
