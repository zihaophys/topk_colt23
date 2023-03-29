using DelimitedFiles
using Random


# fixed budget simulator
function sim(policy, theta, M, budget::Integer, nbins, io=stdout)
    K = length(theta)
    
    randomSeeds = [200:100:(nworkers()+1)*100;]
    rngList = Array{Any}(nothing, (nworkers(), K))
    for a in 1:nworkers()
        for k in 1:K
            rngList[a, k] = MersenneTwister(randomSeeds[a]+k)
        end
    end
    # println(rngList)
    
    N = zeros(1, K)
    R = zeros(1, K)
    N, R = @distributed ((x,y) -> ((+)(x[1],y[1]),(+)(x[2], y[2]))) for bin in 1:nbins 
        n, r = policy(theta, M, budget, rngList)
        n, r
    end
    if SaveOrNot == "yes"
        write(io, "$(theta), $(typeDistribution). Select $(M).")
        write(io, "\n")
        write(io, "$(policy)")
        write(io, "\n")
        writedlm(io, sum(R, dims=1)/nbins)
    end    

    proportion = N[1,:]/nbins/budget
    # print allocation ratio
    println()
    print("Results for $(policy), average on $(nbins) runs \n")
    print("realized allocation ratio: $(proportion) \n")
    println()
end

# fixed confidence simulator 
function sim(policy, theta, M, delta::Real, nbins, io=stdout)
    K = length(theta)
    
    randomSeeds = [200:100:(nworkers()+1)*100;]
    rngList = Array{Any}(nothing, (nworkers(), K))
    for a in 1:nworkers()
        for k in 1:K
            rngList[a, k] = MersenneTwister(randomSeeds[a]+k)
        end
    end
    
    N = zeros(nbins, K)
    N, R = @distributed ((x,y) -> (vcat(x[1],y[1]),vcat(x[2],y[2]))) for bin in 1:nbins 

        for a in 1:nworkers()
            for k in 1:K
                seed_add = abs(rand(rngList[a,k], Int32)) + bin
                rngList[a, k] = MersenneTwister(randomSeeds[a]+k+seed_add)
            end
        end
        
        n, r = policy(theta, M, delta, rngList)
        n, r
    end
    
    if SaveOrNot == "yes"
        write(io, "$(theta), $(typeDistribution). Select $(M).")
        write(io, "\n")
        write(io, "$(policy)")
        write(io, "\n")
        writedlm(io, sum(N, dims=2))
    end
        
    
    proportion = zeros(K)
    for bin in 1:nbins
        if R[bin] == true 
            proportion += N[bin, :]/sum(N[bin, :])
        end
    end
    proportion = proportion/nbins
    # print allocation ratio
    print("Results for $(policy), average on $(nbins) runs \n")
    print("number of samples: $(sum(N)/nbins) \n")
    print("realized error rate: $(1.0-sum(R)/nbins) \n")
    if SaveOrNot == "yes"
        write(io, "$(1.0-sum(R)/nbins)")
        write(io, "\n")
        append!(SAMPLE_COMP, sum(N)/nbins)
        append!(ERR_R, 1.0-sum(R)/nbins)
    end
    # print("realized allocation ratio: $(proportion) \n")
    println()

end

# posterior convergence simulator 
function sim(policy, theta, M, setting::Char, conf::Real, nbins, io=stdout)
    K = length(theta)

    randomSeeds = [200:100:(nworkers()+1)*100;]
    rngList = Array{Any}(nothing, (nworkers(), K))
    for a in 1:nworkers()
        for k in 1:K
            rngList[a, k] = MersenneTwister(randomSeeds[a]+k)
        end
    end
    
    N = zeros(nbins, K)
    N, R = @distributed ((x,y) -> (vcat(x[1],y[1]),vcat(x[2],y[2]))) for bin in 1:nbins 

        for a in 1:nworkers()
            for k in 1:K
                seed_add = abs(rand(rngList[a,k], Int32)) + bin
                rngList[a, k] = MersenneTwister(randomSeeds[a]+k+seed_add)
            end
        end
        
        n, r = policy(theta, M, setting, conf, rngList)
        n, r
    end
    
    if SaveOrNot == "yes"
        write(io, "$(theta), $(typeDistribution). Select $(M).")
        write(io, "\n")
        write(io, "$(policy)")
        write(io, "\n")
        writedlm(io, sum(N, dims=2))
    end
        
    
    proportion = zeros(K)
    for bin in 1:nbins
        if R[bin] == true 
            proportion += N[bin, :]/sum(N[bin, :])
        end
    end
    proportion = proportion/nbins
    # print allocation ratio
    print("Results for $(policy), average on $(nbins) runs \n")
    print("number of samples: $(sum(N)/nbins) \n")
    print("realized error rate: $(1.0-sum(R)/nbins) \n")
    if SaveOrNot == "yes"
        write(io, "$(1.0-sum(R)/nbins)")
        write(io, "\n")
    end
    print("realized allocation ratio: $(proportion) \n")
    println()
end

