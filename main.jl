import Pkg
Pkg.activate(@__DIR__)
using CSV, DataFrames, Statistics, Distributions, HypothesisTests, LinearAlgebra, SHA
ENV["GKSwstype"] = "100"
using Plots

root = @__DIR__
out = joinpath(root, "results")
fig = joinpath(root, "figures")
mkpath(out)
mkpath(fig)
raw = CSV.read(joinpath(root, "data", "result_data_n_2.csv"), DataFrame)
data = filter(r -> isequal(r.ITTFL, "ДА"), raw)
@assert nrow(raw) == length(unique(raw.SUBJID))
@assert all(data.V1_RANDOMIZED .== "ДА")
@assert all(data.TRT01P .== data.V1_RAND_ARM)
data = select(data, :SUBJID, :TRT01P, :ITTFL, :V1_AGE, :V1_SEX,
    :V1_NIHSS_VALUE, :V4_NIHSS_VALUE, :V4_VISDY, :V4_NIHSS_CHG,
    :CTEND_REASON, :CTEND_ALIVE, :CTEND_DEATHDY, :CTEND_DTH120)
@assert Set(data.TRT01P) == Set(["Препарат", "Плацебо"])
@assert Set(data.V1_SEX) == Set(["Ж", "М"])
@assert !any(ismissing, data.V1_AGE)
data.TRT_DRUG = Int.(data.TRT01P .== "Препарат")
data.SEX_FEMALE = Int.(data.V1_SEX .== "Ж")
age_mean, age_sd = mean(data.V1_AGE), std(data.V1_AGE)
data.AGE_Z = (data.V1_AGE .- age_mean) ./ age_sd
data.DELTA_NIHSS = data.V4_NIHSS_VALUE .- data.V1_NIHSS_VALUE
data.PAIR_NIHSS = .!ismissing.(data.DELTA_NIHSS)
@assert all(isequal.(data.DELTA_NIHSS, data.V4_NIHSS_CHG))
for f in [:V1_NIHSS_VALUE, :V4_NIHSS_VALUE]
    @assert all(x -> 0 <= x <= 42, skipmissing(data[!, f]))
end
pairs = data[data.PAIR_NIHSS, :]
@assert !any(coalesce.(pairs.CTEND_DEATHDY .< pairs.V4_VISDY, false))
groups = ["Препарат", "Плацебо"]
save(name, df) = CSV.write(joinpath(out, name * ".csv"), df; bom=true)
save("analytic_data", data)
save("complete_pairs", pairs)
save("standardization", DataFrame(variable=["V1_AGE"], n=[nrow(data)], mean=[age_mean], sd=[age_sd]))

function describe(x)
    a = Float64.(collect(skipmissing(x)))
    n = length(a)
    m, s = mean(a), std(a)
    q1, med, q3 = quantile(a, [0.25, 0.5, 0.75])
    margin = quantile(TDist(n-1), 0.975) * s / sqrt(n)
    skew = mean((a .- m).^3) / mean((a .- m).^2)^1.5
    outside = count(v -> v < q1-1.5*(q3-q1) || v > q3+1.5*(q3-q1), a)
    (;n, missing=length(x)-n, mean=m, sd=s, median=med, q1, q3,
        min=minimum(a), max=maximum(a), ci_low=m-margin, ci_high=m+margin,
        skewness=skew, tukey_outside=outside)
end

desc = DataFrame()
for (population, frame, fields) in [
    ("ITT", data, [:V1_AGE, :V1_NIHSS_VALUE, :V4_NIHSS_VALUE, :DELTA_NIHSS]),
    ("Pairs", pairs, [:V1_AGE, :V1_NIHSS_VALUE, :V4_NIHSS_VALUE, :DELTA_NIHSS, :V4_VISDY])]
    for g in vcat(["Все"], groups), f in fields
        sub = g == "Все" ? frame : frame[frame.TRT01P .== g, :]
        push!(desc, merge((;population, group=g, variable=String(f)), describe(sub[!, f])); cols=:union)
    end
end
save("descriptive", desc)

flow = DataFrame()
sex = DataFrame()
for g in vcat(["Все"], groups)
    sub = g == "Все" ? data : data[data.TRT01P .== g, :]
    absent = .!sub.PAIR_NIHSS
    push!(flow, (;group=g, n=nrow(sub), pairs=sum(sub.PAIR_NIHSS), missing=sum(absent),
        missing_death=count(absent .& coalesce.(sub.CTEND_REASON .== "Смерть", false)),
        missing_dropout=count(absent .& coalesce.(sub.CTEND_REASON .== "Выбывание", false)),
        missing_unknown=count(absent .& ismissing.(sub.CTEND_REASON)),
        death120=count(isequal(1), sub.CTEND_DTH120),
        paired_death120=count(sub.PAIR_NIHSS .& coalesce.(sub.CTEND_DTH120 .== 1, false))))
    for level in ["Ж", "М"]
        n = count(==(level), sub.V1_SEX)
        push!(sex, (;group=g, sex=level, n, percent=100*n/nrow(sub)))
    end
end
save("flow", flow)
save("sex", sex)

function welch(x, y)
    a, b = Float64.(collect(skipmissing(x))), Float64.(collect(skipmissing(y)))
    t = UnequalVarianceTTest(a, b)
    ci = confint(t)
    (;n1=length(a), n0=length(b), difference=mean(a)-mean(b),
        ci_low=ci[1], ci_high=ci[2], statistic=t.t, df=t.df, p=pvalue(t),
        smd=(mean(a)-mean(b))/sqrt((var(a)+var(b))/2))
end
function chi_result(mat)
    expected = sum(mat; dims=2) * sum(mat; dims=1) / sum(mat)
    @assert minimum(expected) >= 5
    t = ChisqTest(mat)
    (;statistic=t.stat, df=(size(mat,1)-1)*(size(mat,2)-1), p=pvalue(t), min_expected=minimum(expected))
end
function holm(p)
    adjusted = zeros(length(p))
    running = 0.0
    for (i,j) in enumerate(sortperm(p))
        running = max(running, (length(p)-i+1)*p[j])
        adjusted[j] = min(1.0, running)
    end
    adjusted
end

baseline = DataFrame()
for f in [:V1_AGE, :V1_NIHSS_VALUE]
    x = data[data.TRT01P .== groups[1], f]
    y = data[data.TRT01P .== groups[2], f]
    push!(baseline, merge((;variable=String(f)), welch(x,y)); cols=:union)
end
sexmat = [count((data.V1_SEX .== level) .& (data.TRT01P .== g)) for level in ["Ж","М"], g in groups]
sextest = chi_result(sexmat)
adjusted = holm(vcat(baseline.p, sextest.p))
baseline.p_holm = adjusted[1:2]
save("baseline_tests", baseline)
save("sex_test", DataFrame([merge(sextest, (;p_holm=adjusted[3]))]))

missing_desc = DataFrame()
for complete in [true, false], f in [:V1_AGE, :V1_NIHSS_VALUE]
    push!(missing_desc, merge((;complete, variable=String(f)), describe(data[data.PAIR_NIHSS .== complete, f])); cols=:union)
end
save("missing_descriptive", missing_desc)
missingmat = [count((data.PAIR_NIHSS .== level) .& (data.TRT01P .== g)) for level in [true,false], g in groups]
save("missing_test", DataFrame([chi_result(missingmat)]))

drug = pairs[pairs.TRT01P .== groups[1], :]
placebo = pairs[pairs.TRT01P .== groups[2], :]
primary = DataFrame([welch(drug.DELTA_NIHSS, placebo.DELTA_NIHSS)])
save("primary_welch", primary)
paired_tests = DataFrame()
for g in groups
    x = Float64.(pairs[pairs.TRT01P .== g, :DELTA_NIHSS])
    t = OneSampleTTest(x)
    ci = confint(t)
    push!(paired_tests, (;group=g, n=length(x), mean_change=mean(x),
        ci_low=ci[1], ci_high=ci[2], statistic=t.t, df=length(x)-1, p=pvalue(t)))
end
paired_tests.p_holm = holm(paired_tests.p)
save("paired_tests", paired_tests)
mw = ApproximateMannWhitneyUTest(Float64.(drug.DELTA_NIHSS), Float64.(placebo.DELTA_NIHSS))
save("rank_test", DataFrame(u=[mw.U], p=[pvalue(mw)]))

X = hcat(ones(nrow(pairs)), pairs.TRT_DRUG, pairs.V1_NIHSS_VALUE, pairs.AGE_Z, pairs.SEX_FEMALE)
y = Float64.(pairs.V4_NIHSS_VALUE)
beta = X \ y
residuals = y - X*beta
bread = inv(X'X)
leverage = vec(sum((X*bread).*X; dims=2))
vcov_hc3 = bread * (X' * Diagonal((residuals ./ (1 .- leverage)).^2) * X) * bread
se = sqrt.(diag(vcov_hc3))
df = length(y) - size(X,2)
critical = quantile(TDist(df), 0.975)
model = DataFrame(term=["Intercept","TRT_DRUG","V1_NIHSS_VALUE","AGE_Z","SEX_FEMALE"],
    estimate=beta, se_hc3=se, ci_low=beta-critical*se, ci_high=beta+critical*se,
    statistic=beta./se, p=2 .* ccdf.(TDist(df), abs.(beta./se)))
save("ancova_hc3", model)
save("model_diagnostics", DataFrame(n=[length(y)], df=[df], max_leverage=[maximum(leverage)],
    residual_sd=[std(residuals)], r2=[1-sum(residuals.^2)/sum((y.-mean(y)).^2)]))
save("model_residuals", DataFrame(fitted=X*beta, residual=residuals))

packages = DataFrame(name=String[], version=String[], direct=Bool[])
for (_, info) in Pkg.dependencies()
    push!(packages, (info.name, isnothing(info.version) ? "stdlib" : string(info.version), info.is_direct_dep))
end
sort!(packages, :name)
save("packages", packages)
open(joinpath(out, "provenance.txt"), "w") do io
    println(io, "Julia ", VERSION)
    println(io, "Source SHA256: ", bytes2hex(sha256(read(joinpath(root,"data","result_data_n_2.csv")))))
    println(io, "Raw rows: ", nrow(raw), "; raw columns: ", ncol(raw))
    println(io, "ITT rows: ", nrow(data), "; paired rows: ", nrow(pairs))
    println(io, "SD uses n-1; quantiles type 7; mean CI uses Student t, 95%.")
    println(io, "Primary: two-sided Welch comparison of changes among observed pairs.")
    println(io, "Ancova: OLS with HC3 standard errors, t reference with n-p df; supplementary only.")
end

default(fontfamily="Arial", size=(1100,650), dpi=160, guidefontsize=13,
    tickfontsize=11, legendfontsize=11, titlefontsize=15, margin=6Plots.mm)
colors = ["#246A8C", "#C47B38"]
for (name, variable, xlabel, bins) in [
    ("age", :V1_AGE, "Возраст, лет", 38:4:86),
    ("baseline", :V1_NIHSS_VALUE, "NIHSS на первом визите, баллы", 2.5:2:30.5)]
    p = plot(xlabel=xlabel, ylabel="Доля внутри группы", legend=:topright)
    for (g,c) in zip(groups,colors)
        histogram!(p, data[data.TRT01P .== g, variable]; bins=bins, normalize=:probability,
            alpha=0.5, color=c, label=g, linecolor=:white)
    end
    savefig(p, joinpath(fig,name*".png"))
end
p = plot(xlabel="Изменение NIHSS V4 − V1, баллы", ylabel="Доля внутри группы", legend=:topleft)
for (g,c) in zip(groups,colors)
    histogram!(p, pairs[pairs.TRT01P .== g,:DELTA_NIHSS]; bins=-18.5:2:19.5,
        normalize=:probability, alpha=0.5, color=c, label=g, linecolor=:white)
end
vline!(p,[0];label="Без изменения",color=:black,linestyle=:dash)
savefig(p,joinpath(fig,"change.png"))
p = plot(xticks=([1,2],["Первый визит","Четвёртый визит"]), xlims=(0.8,2.2),
    ylabel="Средний NIHSS, баллы", legend=:topright)
for (g,c) in zip(groups,colors)
    sub=pairs[pairs.TRT01P .== g,:]
    summaries=[describe(sub[!,f]) for f in [:V1_NIHSS_VALUE,:V4_NIHSS_VALUE]]
    plot!(p,[1,2],[s.mean for s in summaries]; yerror=[s.ci_high-s.mean for s in summaries],
        label=g*" (n="*string(nrow(sub))*")",color=c,marker=:circle,linewidth=3)
end
savefig(p,joinpath(fig,"dynamics.png"))
f=flow[flow.group .!= "Все",:]
p=bar(1:2,100 .* f.missing ./ f.n;xticks=(1:2,groups),xlims=(0.4,2.6),
    label=false,color=colors,ylabel="Нет конечной оценки, %",ylims=(0,30),bar_width=0.55)
for i in 1:2
    annotate!(p,i,100*f.missing[i]/f.n[i]+1.5,text(string(f.missing[i])*" из "*string(f.n[i]),11))
end
savefig(p,joinpath(fig,"missing.png"))
p=scatter(X*beta,residuals;label=false,alpha=0.35,markersize=3,color=colors[1],
    xlabel="Предсказанный NIHSS на V4",ylabel="Остаток модели, баллы")
hline!(p,[0];color=:black,label=false,linestyle=:dash)
savefig(p,joinpath(fig,"residuals.png"))
println("Готово: ", nrow(data), " пациентов; ", nrow(pairs), " полных пар.")
show(stdout, MIME("text/plain"), primary)
println("\nРезультаты: ", out)


using CSV, DataFrames, Statistics, Distributions, Plots
ENV["GKSwstype"] = "100"

let
    d = CSV.read(joinpath(@__DIR__, "results", "analytic_data.csv"), DataFrame)
    complete = d[.!ismissing.(d.DELTA_NIHSS), :]
    groups = ["Препарат", "Плацебо"]
    palette = ["#246A8C", "#C47B38"]
    destination = joinpath(@__DIR__, "figures")
    mkpath(destination)
    @assert nrow(d) == 700
    @assert nrow(complete) == 560
    @assert all(isequal.(complete.DELTA_NIHSS,
        complete.V4_NIHSS_VALUE .- complete.V1_NIHSS_VALUE))
    default(fontfamily="Arial", size=(1100, 650), dpi=180,
        guidefontsize=15, tickfontsize=13, legendfontsize=13,
        titlefontsize=15, margin=7Plots.mm, framestyle=:box,
        background_color=:white, gridalpha=0.15)

    p = plot(xlabel="Возраст, лет", ylabel="Доля внутри группы, %",
        xlims=(44.5, 76.5), xticks=45:4:77,
        ylims=(0, 0.25), yticks=(0:0.05:0.25, string.(0:5:25)),
        legend=:topleft)
    for (g, c) in zip(groups, palette)
        values = d[d.TRT01P .== g, :V1_AGE]
        histogram!(p, values; bins=44.5:4:76.5, normalize=:probability,
            fillalpha=0.35, linealpha=1, linewidth=1.2,
            color=c, linecolor=c, label=g * " (n=" * string(length(values)) * ")")
    end
    savefig(p, joinpath(destination, "section3_age.png"))

    p = plot(xticks=(1:2, ["Препарат\n(n=349)", "Плацебо\n(n=351)"]),
        xlims=(0.4, 2.6), ylims=(0, 22), yticks=0:2:22,
        ylabel="NIHSS на первом визите, баллы", legend=false)
    for (i, g) in enumerate(groups)
        values = Float64.(d[d.TRT01P .== g, :V1_NIHSS_VALUE])
        q1, median_value, q3 = quantile(values, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        low = minimum(filter(x -> x >= q1 - 1.5iqr, values))
        high = maximum(filter(x -> x <= q3 + 1.5iqr, values))
        outsiders = filter(x -> x < low || x > high, values)
        rectangle = Shape([i-0.25, i+0.25, i+0.25, i-0.25], [q1, q1, q3, q3])
        plot!(p, rectangle; color=palette[i], fillalpha=0.30,
            linecolor=palette[i], linewidth=2, label=false)
        plot!(p, [i, i], [low, q1]; color=palette[i], linewidth=2, label=false)
        plot!(p, [i, i], [q3, high]; color=palette[i], linewidth=2, label=false)
        plot!(p, [i-0.14, i+0.14], [low, low]; color=palette[i], linewidth=2, label=false)
        plot!(p, [i-0.14, i+0.14], [high, high]; color=palette[i], linewidth=2, label=false)
        plot!(p, [i-0.25, i+0.25], [median_value, median_value];
            color=palette[i], linewidth=4, label=false)
        annotate!(p, i+0.40, median_value,
            text(string(Int(median_value)), 13, palette[i]))
        if !isempty(outsiders)
            scatter!(p, fill(i, length(outsiders)), outsiders;
                color=palette[i], markersize=4, label=false)
        end
    end
    savefig(p, joinpath(destination, "section3_baseline_box.png"))

    p = plot(xlabel="Изменение NIHSS: четвёртый визит минус первый, баллы",
        ylabel="Доля внутри группы, %", xlims=(-17.5, 17.5),
        xticks=-16:4:16, ylims=(0, 0.22),
        yticks=(0:0.05:0.20, string.(0:5:20)), legend=:topright)
    for (g, c) in zip(groups, palette)
        values = complete[complete.TRT01P .== g, :DELTA_NIHSS]
        histogram!(p, values; bins=-17.5:1:17.5, normalize=:probability,
            fillalpha=0.35, linealpha=1, linewidth=1.2,
            color=c, linecolor=c, label=g * " (n=" * string(length(values)) * ")")
    end
    vline!(p, [0]; color="#333333", linestyle=:dash, linewidth=1.5,
        label="Без изменения")
    savefig(p, joinpath(destination, "section3_change.png"))

    p = plot(xticks=([1, 2], ["Первый визит", "Четвёртый визит"]),
        xlims=(0.75, 2.25), ylims=(0, 14), yticks=0:2:14,
        ylabel="Средний NIHSS и 95% ДИ, баллы", legend=:bottomleft)
    for (g, c) in zip(groups, palette)
        sub = complete[complete.TRT01P .== g, :]
        means = [mean(sub[!, f]) for f in [:V1_NIHSS_VALUE, :V4_NIHSS_VALUE]]
        margins = [quantile(TDist(nrow(sub)-1), 0.975) * std(sub[!, f]) / sqrt(nrow(sub))
            for f in [:V1_NIHSS_VALUE, :V4_NIHSS_VALUE]]
        plot!(p, [1, 2], means; yerror=margins,
            color=c, marker=:circle, markersize=6, linewidth=3,
            label=g * " (n=" * string(nrow(sub)) * ")")
    end
    savefig(p, joinpath(destination, "section3_dynamics.png"))

    sizes = [count(==(g), d.TRT01P) for g in groups]
    absent = [count(ismissing, d[d.TRT01P .== g, :V4_NIHSS_VALUE]) for g in groups]
    percentages = 100 .* absent ./ sizes
    p = bar(1:2, percentages; xticks=(1:2, groups), xlims=(0.4, 2.6),
        ylims=(0, 30), yticks=0:5:30, bar_width=0.5,
        color=palette, linecolor=palette, fillalpha=0.85, label=false,
        ylabel="Нет конечной оценки NIHSS, %")
    for i in 1:2
        label = replace(string(round(percentages[i], digits=1)), "." => ",") *
            "%\n" * string(absent[i]) * " из " * string(sizes[i])
        annotate!(p, i, percentages[i]+2.6, text(label, 13, "#222222"))
    end
    savefig(p, joinpath(destination, "section3_missing.png"))
    println("Построены 5 графиков раздела 3: ", destination)
end
