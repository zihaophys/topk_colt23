# For Gaussian Bandits, run top-k arms with different algorithms, 
# including LinGame, LinGapE, FWSampling, Menard, and compare the results.

# Codes are modified from Andrea Tirinzoni's GitHub Codes "bandie-elimination"
# Parallelization is added to speed up the experiment.

using JLD2;
using Distributed;
using Printf;
using IterTools;
using Distributions;
using DelimitedFiles

@everywhere include("thresholds.jl")
@everywhere include("peps.jl")
@everywhere include("elimination_rules.jl")
@everywhere include("stopping_rules.jl")
@everywhere include("sampling_rules.jl")
@everywhere include("runit.jl")
@everywhere include("experiment_helpers.jl")
@everywhere include("utils.jl")
@everywhere include("envelope.jl")

@everywhere nbins = 1000;
@everywhere seed = rand(1145:1145140);
@everywhere δ = 0.01


# for reproducibility
@everywhere rng = MersenneTwister(rand(1145:1145140))

#%%%%%%%%%%%% Problem Instance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


@everywhere ExpName = "10choose3"
@everywhere m = 3
@everywhere θ = [-1.9, -0.6, -0.5, -0.4, -0.3, -0.1, 0.0, 0.1, 0.4, 1.8]
@everywhere scale = 0.0008895381 # parameter for Menard's algorithm Lazy Mirror Ascent



########################################################################################

EXEC_TIME = Vector()
SAMPLE_COMP = Vector()
ERR_R = Vector()

@everywhere K = length(θ)
@everywhere d = K
@everywhere arms = Vector{Float64}[]
@everywhere for k = 1:K
    v = zeros(d)
    v[k] = 1.0
    push!(arms, v)
end

@everywhere μ = [arm'θ for arm in arms]
@everywhere topm_arms = istar(Topm(arms, m), θ)
@everywhere β = GK16() # choose the threshold of loglikelihood ratio test as log((logt+1)/δ)
@everywhere pep = Topm(arms, m); # choose the PEP as top m arms

# set maximum number of samples. 
# Because we are running fixed-confidence experiment, some algorithm cannot stop in finite time almost surely. 
# So we set a maximum number of samples to avoid infinite loop. We use 1e6 as the maximum number of samples.
@everywhere max_samples = 1e6 

@everywhere policies = [LinGapE(NoElimSR), LinGame(CTracking, NoElimSR, false), FWSampling(NoElimSR), Menard(CTracking, scale, NoElimSR)]

function sim(policy, io=stdout)
    N = zeros(nbins, K)
    N, R = @distributed ((x,y) -> (vcat(x[1],y[1]),vcat(x[2],y[2]))) for bin in 1:nbins 
        n, r = runit(rand(1145:1145140), policy, Force_Stopping(max_samples, LLR_Stopping()), NoElim(), θ, pep, β, δ)
        n, r
    end
    # println(N)
    write(io, "\n")
    write(io, "$(policy)")
    write(io, "\n")
    writedlm(io, sum(N, dims=2))

    proportion = zeros(K)
    for bin in 1:nbins
        if R[bin] == true 
            proportion += N[bin, :]/sum(N[bin, :])
        end
    end
    proportion = proportion/nbins
    print("Results for $(policy), average on $(nbins) runs \n")
    print("number of samples: $(sum(N)/nbins) \n")
    print("realized error rate: $(1.0-sum(R)/nbins) \n")
    write(io, "$(1.0-sum(R)/nbins)")
    write(io, "\n")
    append!(SAMPLE_COMP, sum(N)/nbins)
    append!(ERR_R, 1.0-sum(R)/nbins)
    println()
end

println()
dir = string("data/", ExpName, "k", m, "d", string(δ)[3:end], ".txt")
open(dir, "w") do io 
    append!(EXEC_TIME, ([@elapsed sim(policy, io) for policy in policies])) 
end

DELIM = Vector{Any}(1:4)
for i in 1:4
    if ERR_R[i] < 1e-5
        DELIM[i] = " "
    else
        DELIM[i] = " ($(round(ERR_R[i]*100; digits=1))\\%)"
    end
    SAMPLE_COMP[i] = round(Int, SAMPLE_COMP[i])
    EXEC_TIME[i] = round(EXEC_TIME[i]; digits=1)
end

println(ExpName, " ", ", delta = ", δ)

println()
println("Sample Complexity")
println("m-LinGapE, MisLid, FWS, LMA")
println(SAMPLE_COMP[1], DELIM[1], " & ", SAMPLE_COMP[2], DELIM[2]," & ",SAMPLE_COMP[3], DELIM[3], " & ",SAMPLE_COMP[4], DELIM[4])
println()


