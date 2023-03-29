using Statistics
using Dates
using Distributed
using DelimitedFiles

######################## Input Parameter List #############################

@everywhere typeDistribution = "Bernoulli"
# @everywhere typeDistribution = "Gaussian"
@everywhere sigma = 1
@everywhere include("algorithm.jl")
nbins = 1000

#%%%%%%%%%%%% Choosing Policies %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

budget = 1000000
@everywhere policies = [TSKKT_point5_ids, TSKKT_1_ids, TSPPS_ids]
    
#%%%%%%%%%%%%%%%%%% File IO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SaveOrNot = "yes"

#%%%%%%%%%%%% Problem Instance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@everywhere ExpName = "exp1"
@everywhere k = 2
@everywhere theta = [0.1, 0.2, 0.3, 0.4, 0.5]

###########################################################################


@everywhere theta = sort(theta, rev=true)
@everywhere optimal_allocation = Oracle(ExpName)
println(optimal_allocation)


function sim(policy, theta, M, budget::Integer, nbins, io=stdout)

    K = length(theta)
    
    randomSeeds = [200:100:(nworkers()+1)*100;]
    rngList = Array{Any}(nothing, (nworkers(), K))
    for a in 1:nworkers()
        for k in 1:K
            rngList[a, k] = MersenneTwister(randomSeeds[a]+k)
            # rngList[a, k] = MersenneTwister(rand(114514:1145140))
        end
    end
    
    Eps = @distributed (vcat) for bin in 1:nbins 
        eps = policy(theta, M, budget, rngList)
        eps
    end
    write(io, "$(theta), $(typeDistribution). Select $(M).")
    write(io, "\n")
    write(io, "$(policy)")
    write(io, "\n")
    writedlm(io, Eps)

end

println("Parallel Computing with $(nworkers()) workers.")
FileIO = "w"
dir = string("data/", ExpName, typeDistribution, ".txt")

open(dir, FileIO) do io 
    @time for policy in policies
            sim(policy, theta, k, budget, nbins, io)
    end
end




