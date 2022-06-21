"""
    bst = load_model("filename")

Load a saved Booster.

"""
function load_model(file::String)
    if isfile(file)
        bst = Booster(model_file=file)
    else
        error("File $file not found.")
    end
    
    println("Loaded xgb booster file $file.")

    return(bst)

end

