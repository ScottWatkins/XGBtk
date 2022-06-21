"""

    info, gainplot, freqplot, covplot = plot_feature_info(bst, features, n=10, rename=())
Get the feature importance from a XGBooster model. Create bar plots for gain, freq, and coverage. View plots using display(plotname).
Note: the booster is created with the build_model function, and tne features list is created with the  csv2dmat function.
"""
function plot_feature_info(bst::Booster, features::Vector; n::Int=10, newnames::Dict=Dict())

    f_info = DataFrame(importance(bst))
    idx = parse.(Int, replace.(f_info.fname, "f"=>""))
    label = replace.([features[i+1] for i in idx], "_"=>" ")
    
    insertcols!(f_info, :feature=>label)

    if length(newnames) > 0
        for i in f_info.feature
            if haskey(newnames, i)
                f_info.feature = replace.(f_info.feature, i=>newnames[i])
            end
        end
    end
    
    fnames = "  " .* f_info.feature[1:n]
    fnames = lpad.(fnames, 12, " ")

    tfnames =[ i[1:12] for i in fnames]

    gplot = bar(tfnames, f_info.gain[1:n], color=:firebrick, label="Gain", xguidefontsize=11, xrotation=45)

    fplot = bar(tfnames, f_info.freq[1:n], color=:dodgerblue, label="Frequency", xguidefontsize=11, xrotation=45)

    cplot = bar(tfnames, f_info.cover[1:n], color=:olive, label="Coverage", xguidefontsize=11, xrotation=45) 


    if n > 18
        printstyled("Not all labels will display if n > 18\n", color = :yellow)
    end

    return(f_info, gplot, fplot, cplot)

end
