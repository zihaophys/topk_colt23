# For Bernoulli Bandits, run top-k arms with algorithms FWSampling and Menard, 
# and compare the results.

# Codes are modified from Andrea Tirinzoni's GitHub Codes "bandie-elimination"
# Modification is made to run Bernoulli Bandits
# We do not run LinGapE and LinGame for Bernoulli Bandits 
# since they are original designed for Linear Bandits with Gaussian Noise. 
# Although the extension to Bernoulli Bandits is possible, it can not be fairly 
# compared with other algorithms due to the heuristic stopping threshold used in their practical implementation.

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
@everywhere δ = 0.1


@everywhere max_samples = 1e6
@everywhere rng = MersenneTwister(rand(1145:1145140))
@everywhere β = GK16()

#%%%%%%%%%%%% Problem Instance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@everywhere ExpName = "5choose2"
@everywhere m = 2
@everywhere θ = [0.1, 0.2, 0.3, 0.4, 0.5]
@everywhere scale = 0.0047242023

###########################################################################

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
@everywhere pep = Topm(arms, m);
@everywhere policies = [FWSampling(NoElimSR), Menard(CTracking, scale, NoElimSR)]

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

DELIM = Vector{Any}(1:2)
for i in 1:2
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
println("FWS, LMA")
println(SAMPLE_COMP[1], DELIM[1], " & ", SAMPLE_COMP[2], DELIM[2])
println()