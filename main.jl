# Run Top-Two Algorithms closely related with our paper, including 
# TS-PPS with IDS, TS-PPS with 0.5-tuning, TS-KKT with IDS, TS-KKT with 0.5-tuning,
# TS-KKT(0.5) with IDS, TSk-KKT(1) with IDS, 
# KLLUCB, UGapE and Uniform are also included due to convenient implementation

using Statistics
using Dates
using Distributed
using Random

######################## Input Parameter List #############################

# @everywhere typeDistribution = "Bernoulli"
@everywhere typeDistribution = "Gaussian"
@everywhere sigma = 1
@everywhere include("algorithm.jl")
nbins = 1000

#%%%%%%%%%%%% Choosing Policies %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

deltaOrBudget = 0.001
@everywhere policies = [TSPPS_ids, TSPPS_const, TSKKT_point5_ids, TSKKT_1_ids, KLLUCB, UGapE, Uniform]

#%%%%%%%%%%%%%%%%%% File IO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SaveOrNot = "yes"
EXEC_TIME = Vector()
SAMPLE_COMP = Vector()
ERR_R = Vector()

#%%%%%%%%%%%% Problem Instance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@everywhere ExpName = "5choose2"
@everywhere k = 2
@everywhere theta = [0.1, 0.2, 0.3, 0.4, 0.5]

###########################################################################

@everywhere theta = sort(theta, rev=true)


println("Parallel Computing with $(nworkers()) workers.")

@everywhere optimal_allocation = Oracle(ExpName)
print("theta = $(theta) with $(typeDistribution) distribution. Select $(k). \n")

FileIO = "w"

SaveOrNot = "no"
SaveOrNot = "yes"

if SaveOrNot == "yes"
    if deltaOrBudget < 1
        # print("Theoretical lower bound on the sample complexity: $(log(1/deltaOrBudget)/minC) \n")
        println()
        dir = string("data/", ExpName, typeDistribution, "k", k, "d", string(deltaOrBudget)[3:end], ".txt")
    else
        println()
        dir = string("data/", ExpName, typeDistribution, "k", k, "B", string(deltaOrBudget), ".txt")
    end
    open(dir, FileIO) do io 

    append!(EXEC_TIME, ([@elapsed sim(policy, theta, k, deltaOrBudget, nbins, io) for policy in policies]))
    end
else 
@time for policy in policies
        sim(policy, theta, k, deltaOrBudget, nbins)
end
end



DELIM = Vector{Any}(1:7)
for i in 1:7
    if ERR_R[i] < 1e-5
        DELIM[i] = " "
    else
        DELIM[i] = " ($(round(ERR_R[i]*100; digits=1))\\%)"
    end
    SAMPLE_COMP[i] = round(Int, SAMPLE_COMP[i])
    EXEC_TIME[i] = round(EXEC_TIME[i]; digits=1)
end

println(ExpName, " ", typeDistribution, ", delta = ", deltaOrBudget)

println()
println("Execution Time")
println("TS-PPS-IDS, TS-PPS-0.5, TS-KKT(0.5)-IDS, TS-KKT(1)-IDS, KL-LUCB, UGapE, Uniform")
println(EXEC_TIME[1], " & ", EXEC_TIME[2], " & ",EXEC_TIME[3], " & ",EXEC_TIME[4], EXEC_TIME[5], " & ",EXEC_TIME[6], " & ",EXEC_TIME[7])
println()

println()
println("Sample Complexity")
println("TS-PPS-IDS, TS-PPS-0.5, TS-KKT(0.5)-IDS, TS-KKT(1)-IDS, KL-LUCB, UGapE, Uniform")
println(SAMPLE_COMP[1], DELIM[1], " & ", SAMPLE_COMP[2], DELIM[2]," & ",SAMPLE_COMP[3], DELIM[3]," & ",SAMPLE_COMP[4], DELIM[4]," & ",SAMPLE_COMP[5], DELIM[5]," & ",SAMPLE_COMP[6], DELIM[6]," & ",SAMPLE_COMP[7], DELIM[7])
println()


