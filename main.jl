import Pkg
folder = joinpath(@__DIR__, "Материалы")
Pkg.activate(folder)
using CSV, DataFrames, Statistics

raw = CSV.read(joinpath(folder, "data", "raw", "result_data_n_2.csv"), DataFrame)
data = filter(row -> isequal(row.ITTFL, "ДА"), raw)
@assert length(unique(data.SUBJID)) == nrow(data)

data = select(data, :SUBJID, :TRT01P, :V1_AGE, :V1_SEX,
              :V1_NIHSS_VALUE, :V4_NIHSS_VALUE)

data.TRT_DRUG = Int.(data.TRT01P .== "Препарат")
data.SEX_FEMALE = Int.(data.V1_SEX .== "Ж")

ages = collect(skipmissing(data.V1_AGE))
age_mean = mean(ages)
age_sd = std(ages)
data.AGE_Z = (data.V1_AGE .- age_mean) ./ age_sd

data.DELTA_NIHSS = data.V4_NIHSS_VALUE .- data.V1_NIHSS_VALUE

groups = combine(groupby(data, :TRT01P), nrow => :N,
                 :V1_AGE => mean => :MEAN_AGE)
sex = combine(groupby(data, [:TRT01P, :V1_SEX]), nrow => :N)

output = joinpath(folder, "results")
mkpath(output)
CSV.write(joinpath(output, "data_prepared.csv"), data; bom=true)
CSV.write(joinpath(output, "groups.csv"), groups; bom=true)
CSV.write(joinpath(output, "sex.csv"), sex; bom=true)

println("Пациентов: ", nrow(data), "; показателей: ", ncol(data))
println("Средний возраст: ", age_mean, "; СКО возраста: ", age_sd)
show(stdout, MIME("text/plain"), groups)
println("\nГотово. Таблицы сохранены в ", output)
