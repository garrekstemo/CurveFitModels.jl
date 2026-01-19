push!(LOAD_PATH, "../src/")

using Documenter, CurveFitModels

makedocs(
    sitename="CurveFitModels.jl",
    authors = "Garrek Stemo",
    modules = [CurveFitModels],
    pages = [
        "Home" => "index.md",
        "Model Functions" => "functions.md",
        "Special Functions" => "specialfunctions.md"
    ]
    )
deploydocs(
    repo = "github.com/garrekstemo/CurveFitModels.jl.git"
)