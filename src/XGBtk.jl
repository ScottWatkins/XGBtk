module XGBtk

using SparseArrays, DataFrames, CSV, Plots, DelimitedFiles
using Printf, LinearAlgebra, StatsBase, Statistics, Random

export build_model, confusion_matrix, csv2dmat, libsvm2dmat
export load_model, make_confusion_matrix, plot_feature_info
export prediction, runCVtrain, scale_pos, softprobs_bayes_correction

using XGBoost

include("build_model.jl")
include("cm_stats.jl")
include("csv2dmat.jl")
include("libsvm2dmat.jl")
include("load_model.jl")
include("make_confusion_matrix.jl")
include("plot_feature_info.jl")
include("prediction.jl")
include("runCVtrain.jl")
include("scale_pos.jl")
include("softprobs_bayes_correction.jl")

end
