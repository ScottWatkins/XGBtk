"""
pred, cvdat, cvheader, trainerr, bestcvround = runCVtrain(dtrain; num\\_rounds=100, nfold=5, seed=0, subsample=1, reps=1, num\\_class=2, metrics=["mlogloss"], param=param, verbosity=1, nthread=1, max\\_depth=[6,1,6], eta=[0.3,0.1,0.3], alpha=[0.0,0.1,0.0], gamma=[0.0,0.1,0.0], lambda=[1.0,1.0,1.0], max\\_delta\\_step=[0,1,0], min\\_child\\_weight=[0,1,0])

Run cross-validation using user-defined search parameters. Input data is a DMatrix. Pass an initial parameter set as a dictionary. Pass the evaluation metrics as an array (e.g. metrics = ["auc", "logloss"]). The metrics must match the objective (e.g. multi:softprob uses mlogloss, binary:logistic uses logloss).

Set a range of values to scan for one or more tunable parameters e.g., max_depth [start, increment, stop]. Any combination of listed range parameters are allowed to be tuned. The default range is simple the xgboost default for that parameter. Range can be passed as a vector array [2,1,12] or tuple (2,1,12)

Required: \\
    num\\_rounds .... number of training rounds [100] \\
    nfold ........ the cross-validaton fold [5] (20%) \\
    num\\_class .... the number of label classes [2]  \\
    param  ........ a Dict with the initial set of parameters  \\
  \n
  \n

Options: \\
    verbosity ......  verbosity [1] \\
    randseed  .......  record the random seed for every run [false] \\
    nthread   .......,  number of threads [1]  (all) \\
    reps ........... repeat the run; useful for optimizing final settings [1] \\
 \n
 \n
Examples:\\

Test depth using other default parameters:\\
(pred, cvdat, cvheader, err, bestcvround) = runCVtrain(dtrain; num\\_rounds=10, nfold=5, num\\_class=2, max\\_depth=(2,1,20)

Test depth and learning rate (eta) using other default parameters:\\
pred, cvdat, cvheader, err, bestcvround = runCVtrain(dtrain, metrics=metrics, param=param, num\\_class=5, num\\_rounds=10, nfold=5, eta =[0.2,0.1,0.4], max\\_depth=[7,1,8] ) 

runs eta at 0.2, 0.3, 0.4 at max_depth values of 7 and 8. Example output:
 
 ((7, 0.2, 0.0, 0.0, 1.0, 0, 0), (10, 0.587757, 0.535275))\\
 ((7, 0.3, 0.0, 0.0, 1.0, 0, 0), (10, 0.465590, 0.397316))\\
 ((7, 0.4, 0.0, 0.0, 1.0, 0, 0), (10, 0.402149, 0.320444))\\
 ((8, 0.2, 0.0, 0.0, 1.0, 0, 0), (10, 0.560147, 0.500348))\\
 ((8, 0.3, 0.0, 0.0, 1.0, 0, 0), (10, 0.443602, 0.364027))\\
 ((8, 0.4, 0.0, 0.0, 1.0, 0, 0), (10, 0.382213, 0.286316))\\
\n
Outputs: all outputs are collected in arrays that can be indexed into to get values for each cv run. First, look at the bestcvround tuple pairs ((param1, param2, ...)(bestround, min-test[m]logloss, train[m]logloss)) showing the training round with the minimum (m)logloss error for the test data set. This is the approximate stopping point to prevent overfitting. The best round should be less than the total number of rounds tested, otherwise run more rounds.  If you test i x j x k  = 27 conditions, and you want the predictions and probabilities for the 17th run, the data are in pred[17], bestcvround[17], etc.

"""
function runCVtrain(dtrain::DMatrix; num_rounds::Int64=100, nfold::Int64=5, num_class::Int64=2, seed::Int64=0, metrics::Array=[], param::Union{Dict, Vector{Pair{String, Any}}}=Dict(), max_depth=[6,1,6], eta=[0.3,0.1,0.3], alpha=[0.0,0.1,0.0], gamma=[0.0,0.1,0.0], lambda=[1.0,0.1,1.0], max_delta_step=[0,1,0], min_child_weight=[0,1,0], subsample=1, verbosity=1, nthreads=1, randseed=false, reps=1, early_stop=40, print_every_n=1)
    
    param_default = Dict("max_depth"=>6, "eta"=>0.3, "alpha"=>0.0, "gamma"=>0.0, "lambda"=>1.0, "max_delta_step"=>0, "min_child_weight"=>0)

    metrics_default = ["mlogloss"]

    if length(param) == 0
        param = param_default
    end

    if length(metrics) == 0
        metrics = metrics_default
    end

    num_rounds=num_rounds; nfold=nfold; num_class=num_class; seed=seed; early_stop=early_stop
    pred_array = []; cvdata_array = []; err_array = []; k = 0;
    cvlabels = []; bestcvtestround = []; print_every_n = print_every_n;

    param = Dict(param)

    println("Initials values: num_rounds=$num_rounds, nfold=$nfold, $metrics, $param\n")
    for r in 1:reps
        for i in collect(max_depth[1]:max_depth[2]:max_depth[3])   #set ranges to scan; uncomment corresponding var below
            for j in collect(eta[1]:eta[2]:eta[3])
                for k in collect(alpha[1]:alpha[2]:alpha[3])
                    for g in collect(gamma[1]:gamma[2]:gamma[3])
                        for l in collect(lambda[1]:lambda[2]:lambda[3])
                            for d in collect(max_delta_step[1]:max_delta_step[2]:max_delta_step[3])
                                for c in collect(min_child_weight[1]:min_child_weight[2]:min_child_weight[3])
                            
                                    param["max_depth"] = i       #number of splits [6]
                                    param["eta"] = j             #learning rate [0.3]
                                    param["alpha"] = k           #L1 reg. term [0], higher is cons.
                                    param["gamma"] = g           #min_split_loss [0], higher is cons.
                                    param["lambda"] = l          #L2 reg term [1], higher is cons.
                                    param["max_delta_step"] = d  # [0], try 1-10 for unbalance prob. prediction
                                    param["min_child_weight"] = c #stop partioning [0], higher is cons. 

                                    println("Parameters for this run are...\n", param, "\n")

                                    if randseed == true
                                        seed = Int(floor((rand()*1e4)))
                                    end

                                    pred, cvdata, cvlabel, err, bestcvround = nfold_cv(dtrain, num_rounds, nfold, num_class=num_class, metrics=metrics, param=param, seed=seed, subsample=subsample, verbosity=verbosity, nthread=nthreads, early_stop=early_stop, print_every_n=print_every_n)

                                    push!(pred_array, pred)
                                    push!(cvdata_array, cvdata)
                                    push!(err_array, err)
                                    push!(cvlabels, cvlabel)
                                    push!(bestcvtestround, [i, j, k, g, l, d, c, seed, bestcvround[1], bestcvround[2], bestcvround[3], bestcvround[4], bestcvround[5], err] )

                                end
                            end
                        end
                    end
                end  
            end
        end
    end

    print("-"^35); print("Summary");println("-"^33)
    printstyled("Labels for parameter values below are:
[max_depth, eta, alpha, gamma, lambda, max_delta_step, min_child_weight, seed, round, min-test-logloss, min-test-logloss-std, train-logloss@min, train-logloss@min-std, classification error]\n\n", color=:green)

    r = size(bestcvtestround, 1)

    bestcvtestround = hcat(bestcvtestround...)'

    mintestlossidx = argmin([ bestcvtestround[i,10] for i in 1:r])

    best = bestcvtestround[mintestlossidx, :]
    bestminerr = err_array[mintestlossidx]
    minerr = err_array[argmin(err_array)]
    minerr_params = bestcvtestround[argmin(err_array), :]

    printstyled("Model with minimum test-[m]logloss:\nParameters: $best\nClassification error: $bestminerr percent\n\n", color=:green)
    printstyled("Model with minimum classification error:\nParameters: $minerr_params\nClassification error: $minerr percent\n\n", color=:green)

    if minerr == "NA"
        printstyled("Error value was NA. You may need to set the \"objective\" in your *initial* input parameters (e.g. \"objective\"=>\"multi:softprob\").\n", color=:yellow)
    end

    println("-"^75)
    evaldata = DataFrame(bestcvtestround, :auto)
    
    rename!(evaldata, :x1=>"max_depth", :x2=>"eta", :x3=>"alpha", :x4=>"gamma", :x5=>"lambda", :x6=>"max_delta_step", :x7=>"min_child_weight", :x8=>"seed", :x9=>"round", :x10=>"min_cv_test", :x11=>"min_cv_test_std", :x12=>"cv_train", :x13=>"cv_train_std", :x14=>"error_rate")
    runid = collect(1:size(evaldata,1))
    insertcols!(evaldata, 1, :runid=>runid) 
    evaldata = evaldata[:, [1,11,12,13,14,15,10,2,3,4,5,6,7,8,9]]
    
    return pred_array, cvdata_array, cvlabels, err_array, evaldata

end

