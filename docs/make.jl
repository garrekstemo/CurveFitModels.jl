push!(LOAD_PATH, "../src/")

using Documenter, CurveFitModels

makedocs(
    sitename="CurveFitModels.jl",
    authors = "Garrek Stemo",
    modules = [CurveFitModels],
    pages = [
        "Home" => "index.md",
        "Model Functions" => "funcs.md",
        "Special Functions" => "specialfuncs.md"
    ]
    )
deploydocs(
    repo = "github.com/garrekstemo/Models.jl.git"
)