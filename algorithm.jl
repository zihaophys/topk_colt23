using Distributions
using HCubature
include("KL_functions.jl")
include("sample_functions.jl")
include("tools.jl")

rate(t, delta) = log((log(t)+1)/delta)
optimisim = 0.5


# load the distribution-related functions
if typeDistribution == "Bernoulli"
    d = dBernoulli 
    dup = dupBernoulli
    dlow = dlowBernoulli
    sample_arm = sample_arm_Bernoulli
    post_sample_arm = post_sample_arm_Bernoulli
    etaTotheta = etaTothetaBernoulli
    thetaToeta = thetaToetaBernoulli
elseif typeDistribution == "Gaussian"
    d = dGaussian
    dup = dupGaussian
    dlow = dlowGaussian
    sample_arm = sample_arm_Gaussian
    post_sample_arm = post_sample_arm_Gaussian
    etaTotheta = etaTothetaGaussian
    thetaToeta = thetaToetaGaussian
elseif typeDistribution == "Poisson"
    d = dPoisson
end

function randmax(vector,rank=1)
   # returns an integer, not a CartesianIndex
   vector=vec(vector)
   Sorted=sort(vector,rev=true)
   m=Sorted[rank]
   Ind=findall(x->x==m,vector)
   index=Ind[floor(Int,length(Ind)*rand())+1]
   return (index)
end



# return the optimal allocation ratio for the given problem instance in Dan Russo's paper
# We do not provide a function calculating the optimal solution since it is not straightforward to control the accuracy of the optimization algorithm
# Instead, in additional file, we provide a Python script to calculate the optimal solution for users to verify the convergence.
function Oracle(ExpName)
    if ExpName == "exp1"
        return [0.4508672094687658, 0.44885257237074294, 0.06278512232073853, 0.024826750864824432, 0.012668344974928424]
    elseif ExpName == "exp2"
        return [0.008948895732085476, 0.01388379238167569, 0.024765141897753296, 0.058618983744509766, 0.3953937010818624, 0.40443494707162675, 0.05504967570287817, 0.021535553692867613, 0.011035953727224448, 0.006333354967516708]
    elseif ExpName == "exp3"
        return [0.20757678946164718, 0.05660165029090801, 0.05660165623873876, 0.05660166191012625, 0.05660166745955656, 0.05660167293856588, 0.056601678370655197, 0.056601683768511796, 0.05660168913959744, 0.05660169448959865, 0.056601620579184685, 0.05660162589563009, 0.05660163119850765, 0.056601636489364934, 0.05660164176940692]
    elseif ExpName == "exp4"
        return [0.08995260653279119, 0.00919243856753597, 0.009192442591254409, 0.009192445099770755, 0.009192446977290526, 0.009192448515361497, 0.009192449844531016, 0.009192451033724782, 0.009192452123398583, 0.00919245313913773, 0.00919236309335976, 0.009192364007391988, 0.009192364885181565, 0.009192365733155412, 0.009192366556209796, 0.009192367358153115, 0.009192368141998902, 0.009192368910168603, 0.009192369664634626, 0.009192370407022157, 0.009192371138685143, 0.009192371860761906, 0.009192372574218337, 0.009192373279880045, 0.009192373978457813, 0.009192374670567932, 0.00919237535674775, 0.009192376037468466, 0.009192376713145338, 0.009192377384146338, 0.00919237805079857, 0.009192378713394582, 0.009192379372196512, 0.009192380027440604, 0.009192380679340157, 0.009192381328088394, 0.009192381973861177, 0.00919238261681862, 0.009192383257107375, 0.009192383894861822, 0.009192384530205093, 0.009192385163251172, 0.009192385794104961, 0.009192386422863545, 0.00919238704961698, 0.00919238767444889, 0.00919238829743715, 0.009192388918654338, 0.009192389538168158, 0.009192390156042065, 0.00919239077233547, 0.009192391387104022, 0.00919239200040016, 0.009192392612273228, 0.009192393222769345, 0.009192393831932918, 0.009192394439804803, 0.009192395046424334, 0.009192395651828542, 0.00919239625605257, 0.00919239685912976, 0.009192397461091715, 0.009192398061968629, 0.009192398661789104, 0.009192399260580426, 0.009192399858368745, 0.00919240045517879, 0.009192401051034308, 0.009192401645958064, 0.009192402239971644, 0.00919240283309595, 0.009192403425350812, 0.009192404016755414, 0.009192404607327951, 0.009192405197086086, 0.009192405786046859, 0.009192406374226208, 0.009192406961640134, 0.009192407548303536, 0.009192408134230808, 0.009192408719436052, 0.009192409303932805, 0.00919240988773408, 0.00919241047085211, 0.009192411053299408, 0.009192411635087537, 0.009192412216228001, 0.009192412796731476, 0.009192413376608775, 0.009192413955870116, 0.00919241453452539, 0.009192415112584491, 0.009192415690056376, 0.009192416266950469, 0.009192416843275361, 0.009192417419039601, 0.009192417994251722, 0.00919241856891957, 0.009192419143051031, 0.009192419716653619]
    elseif ExpName == "exp5"
        return [0.4504557550670839, 0.4448894154386389, 0.06389408937446386, 0.026323079374767328, 0.014437660745046085]
    elseif ExpName == "exp6"
        return [0.16666663333436274, 0.1666666333343628, 0.0833333334710051, 0.0833333334710051, 0.0833333500825363, 0.08333335008253631, 0.0833333666320689, 0.0833333666320689, 0.0833333164800269, 0.0833333164800269]
    elseif ExpName == "exp7"
        return [0.2814313030112234, 0.19856409784763387, 0.1097484630783293, 0.09393578631735289, 0.08151923110652333, 0.06339977501907829, 0.056633681425714115, 0.05094397587991824, 0.04610609278225203, 0.017717593531974555]
    elseif ExpName == "exp8"
        return [0.2814313030112234, 0.19856409784763387, 0.1097484630783293, 0.09393578631735289, 0.08151923110652333, 0.06339977501907829, 0.056633681425714115, 0.05094397587991824, 0.04610609278225203, 0.017717593531974555]
    elseif ExpName == "exp9"
        return [0.0006855349997455692, 0.014308137586132056, 0.445826420527509, 0.441148564422825, 0.06330597743106849, 0.014303220717580839, 0.009049610026005767, 0.006245704228432742, 0.004571721021940795, 0.0005551090387596393]
    elseif ExpName == "exp10"
        return [0.0015199660151590047, 0.014197472211767936, 0.04766972536648646, 0.09535845470537507, 0.3340899610783451, 0.3331526532992767, 0.09528197817088022, 0.04765052849176751, 0.029007012941099645, 0.0020722477198424796]
    end
    
end


# algorithms 
function TSKKT_point5_ids(theta::Array, M::Integer, delta::Real, rngList)

    condition = true 
    K = length(theta)
    N = zeros(1, K)
    S = zeros(1, K)
    R = Vector{Bool}()

    order = sortperm(theta, rev=true)
    I_true = order[1:M]

    # initialization
    for a in 1:K
        N[a] = 1.0
        S[a] = sample_arm(theta[a], rngList[myid()-1, a])
    end
    t = K 

    while (condition)

        # step 0: compute true or false of the recommendation
        emp_mean = (S./N)[1,:]
        push!(R, Set(theta[sortperm(emp_mean, rev=true)[1:M]]) == Set(theta[I_true]))

        # step 1: sample theta from posterior
        theta_sample = post_sample_arm(S, N)
        A = sortperm(theta_sample, rev=true)
        I_es = A[1:M]
        J_es = A[M+1:K]


        # step 2: generate (i,j) pair as arm candidates 
        psi = (N/t)[1,:]
        C = zeros(M, K-M)
        for i in 1:M
            for j in 1:(K-M)
                id_i = I_es[i]
                id_j = J_es[j]
                theta_bar = (psi[id_i]*emp_mean[id_i] + psi[id_j]*emp_mean[id_j])/(psi[id_i] + psi[id_j])
                C[i, j] = psi[id_i] * d(emp_mean[id_i], theta_bar) + psi[id_j] * d(emp_mean[id_j], theta_bar) + optimisim * log(1/(1/N[id_i] + 1/N[id_j]))/t
            end
        end
        i, j = Tuple(argmin(C))
        it = I_es[i]
        jt = J_es[j]

        # step 3: toss a biased optimized coin 
        theta_bar = (psi[it]*emp_mean[it] + psi[jt]*emp_mean[jt])/(psi[it] + psi[jt])
        coin = psi[it] * d(emp_mean[it], theta_bar)
        Ctij = coin + psi[jt] * d(emp_mean[jt], theta_bar)
        coin = coin / Ctij 

        if (rand() < coin)
            choice = it
        else
            choice = jt 
        end

        t  = t + 1
        N[choice] += 1
        S[choice] += sample_arm(theta[choice], rngList[myid()-1, choice])

        # check the stopping rule 
        # compute the test statistics Z_t 
        C = zeros(M, K-M)
        A = sortperm(emp_mean, rev=true)
        I_es = A[1:M]
        J_es = A[M+1:K]
        for i in 1:M
            for j in 1:(K-M)
                id_i = I_es[i]
                id_j = J_es[j]
                theta_bar = (psi[id_i]*emp_mean[id_i] + psi[id_j]*emp_mean[id_j])/(psi[id_i] + psi[id_j])
                C[i, j] = psi[id_i] * d(emp_mean[id_i], theta_bar) + psi[id_j] * d(emp_mean[id_j], theta_bar)
            end
        end
        condition = (t * minimum(C) < rate(t, delta))
        if (t > 1000000)
            println("Exceed the Maximum Sample Number: 1_000_000")
            return N , false
        end
    end

    return N, last(R)

end

function TSKKT_1_ids(theta::Array, M::Integer, delta::Real, rngList)

    condition = true 
    K = length(theta)
    N = zeros(1, K)
    S = zeros(1, K)
    R = Vector{Bool}()

    order = sortperm(theta, rev=true)
    I_true = order[1:M]

    # initialization
    for a in 1:K
        N[a] = 1.0
        S[a] = sample_arm(theta[a], rngList[myid()-1, a])
    end
    t = K 

    while (condition)

        # step 0: compute true or false of the recommendation
        emp_mean = (S./N)[1,:]
        push!(R, Set(theta[sortperm(emp_mean, rev=true)[1:M]]) == Set(theta[I_true]))

        # step 1: sample theta from posterior
        theta_sample = post_sample_arm(S, N)
        A = sortperm(theta_sample, rev=true)
        I_es = A[1:M]
        J_es = A[M+1:K]


        # step 2: generate (i,j) pair as arm candidates 
        psi = (N/t)[1,:]
        C = zeros(M, K-M)
        for i in 1:M
            for j in 1:(K-M)
                id_i = I_es[i]
                id_j = J_es[j]
                theta_bar = (psi[id_i]*emp_mean[id_i] + psi[id_j]*emp_mean[id_j])/(psi[id_i] + psi[id_j])
                C[i, j] = psi[id_i] * d(emp_mean[id_i], theta_bar) + psi[id_j] * d(emp_mean[id_j], theta_bar) + 2*optimisim * log(1/(1/N[id_i] + 1/N[id_j]))/t
            end
        end
        i, j = Tuple(argmin(C))
        it = I_es[i]
        jt = J_es[j]

        # step 3: toss a biased optimized coin 
        theta_bar = (psi[it]*emp_mean[it] + psi[jt]*emp_mean[jt])/(psi[it] + psi[jt])
        coin = psi[it] * d(emp_mean[it], theta_bar)
        Ctij = coin + psi[jt] * d(emp_mean[jt], theta_bar)
        coin = coin / Ctij 

        if (rand() < coin)
            choice = it
        else
            choice = jt 
        end

        t  = t + 1
        N[choice] += 1
        S[choice] += sample_arm(theta[choice], rngList[myid()-1, choice])

        # check the stopping rule 
        # compute the test statistics Z_t 
        C = zeros(M, K-M)
        A = sortperm(emp_mean, rev=true)
        I_es = A[1:M]
        J_es = A[M+1:K]
        for i in 1:M
            for j in 1:(K-M)
                id_i = I_es[i]
                id_j = J_es[j]
                theta_bar = (psi[id_i]*emp_mean[id_i] + psi[id_j]*emp_mean[id_j])/(psi[id_i] + psi[id_j])
                C[i, j] = psi[id_i] * d(emp_mean[id_i], theta_bar) + psi[id_j] * d(emp_mean[id_j], theta_bar)
            end
        end
        condition = (t * minimum(C) < rate(t, delta))
        if (t > 1000000)
            println("Exceed the Maximum Sample Number: 1_000_000")
            return N , false
        end
    end

    return N, last(R)

end

function TSKKT_0_const(theta::Array, M::Integer, delta::Real, rngList)

    condition = true 
    K = length(theta)
    N = zeros(1, K)
    S = zeros(1, K)
    R = Vector{Bool}()

    order = sortperm(theta, rev=true)
    I_true = order[1:M]

    # initialization
    for a in 1:K
        N[a] = 1.0
        S[a] = sample_arm(theta[a], rngList[myid()-1, a])
    end
    t = K 

    while (condition)

        # step 0: compute true or false of the recommendation
        emp_mean = (S./N)[1,:]
        push!(R, Set(theta[sortperm(emp_mean, rev=true)[1:M]]) == Set(theta[I_true]))

        # step 1: sample theta from posterior
        theta_sample = post_sample_arm(S, N)
        A = sortperm(theta_sample, rev=true)
        I_es = A[1:M]
        J_es = A[M+1:K]


        # step 2: generate (i,j) pair as arm candidates 
        psi = (N/t)[1,:]
        C = zeros(M, K-M)
        for i in 1:M
            for j in 1:(K-M)
                id_i = I_es[i]
                id_j = J_es[j]
                theta_bar = (psi[id_i]*emp_mean[id_i] + psi[id_j]*emp_mean[id_j])/(psi[id_i] + psi[id_j])
                C[i, j] = psi[id_i] * d(emp_mean[id_i], theta_bar) + psi[id_j] * d(emp_mean[id_j], theta_bar)
            end
        end
        i, j = Tuple(argmin(C))
        it = I_es[i]
        jt = J_es[j]

        # step 3: toss a coin 

        if (rand() < 0.5)
            choice = it
        else
            choice = jt 
        end

        t  = t + 1
        N[choice] += 1
        S[choice] += sample_arm(theta[choice], rngList[myid()-1, choice])

        # check the stopping rule 
        # compute the test statistics Z_t 
        C = zeros(M, K-M)
        A = sortperm(emp_mean, rev=true)
        I_es = A[1:M]
        J_es = A[M+1:K]
        for i in 1:M
            for j in 1:(K-M)
                id_i = I_es[i]
                id_j = J_es[j]
                theta_bar = (psi[id_i]*emp_mean[id_i] + psi[id_j]*emp_mean[id_j])/(psi[id_i] + psi[id_j])
                C[i, j] = psi[id_i] * d(emp_mean[id_i], theta_bar) + psi[id_j] * d(emp_mean[id_j], theta_bar)
            end
        end
        condition = (t * minimum(C) < rate(t, delta))
    end
    if (t > 1000000)
        println("Exceed the Maximum Sample Number: 1_000_000")
        return N, false
    end

    return N, last(R)

end

function logGaussianTail(x)
    if x >= 8.0
        res = log(sqrt(x^2+2*log(2))-x) - x^2/2 - 0.5*log(2*pi)
    else
        res = log(1 - cdf.(Normal(0.0, 1.0), x))
    end
    return res 
end

function TSPPS_ids(theta::Array, M::Integer, delta::Real, rngList)

    condition = true 
    K = length(theta)
    N = zeros(1, K)
    S = zeros(1, K)
    R = Vector{Bool}()

    order = sortperm(theta, rev=true)
    I_true = order[1:M]

    # initialization
    for a in 1:K
        N[a] = 1.0
        S[a] = sample_arm(theta[a], rngList[myid()-1, a])
    end
    t = K 

    while (condition)

        # step 0: compute true or false of the recommendation
        emp_mean = (S./N)[1,:]
        push!(R, Set(theta[sortperm(emp_mean, rev=true)[1:M]]) == Set(theta[I_true]))

        # step 1: sample theta from posterior
        theta_sample = post_sample_arm(S, N)
        A = sortperm(theta_sample, rev=true)
        I_es = A[1:M]
        J_es = A[M+1:K]


        # step 2: generate (i,j) pair as arm candidates and 
        #         evaluate P(theta_j - theta_i > 0)
        psi = (N/t)[1,:]
        if typeDistribution == "Bernoulli"
            vals = zeros(M, K-M)
            for i in 1:M
                for j in 1:(K-M)
                    id_i = I_es[i]
                    id_j = J_es[j]
                    theta_bar = (psi[id_i]*emp_mean[id_i] + psi[id_j]*emp_mean[id_j])/(psi[id_i] + psi[id_j])
                    vals[i, j] = -t * (psi[id_i] * d(emp_mean[id_i], theta_bar) + psi[id_j] * d(emp_mean[id_j], theta_bar))
                end
            end
        elseif typeDistribution == "Gaussian"
            vals = zeros(M, K-M)
            for i in 1:M
                for j in 1:(K-M)
                    id_i = I_es[i]
                    id_j = J_es[j]
                    x = (emp_mean[id_i] - emp_mean[id_j])/sigma/sqrt(1/N[id_i] + 1/N[id_j])
                    vals[i, j] = logGaussianTail(x)
                end
            end
        end
        emp_max_exp = maximum(vals)
        vals = exp.(vals .- emp_max_exp)
        vals = vals / sum(vals)

        Entry = Vector()
        Prob = Vector{Float64}()
        for i in 1:M
            for j in 1:(K-M)
                push!(Entry, (i, j))
                push!(Prob, vals[i, j])
            end
        end

        i, j = wsample(Entry, Prob)

        it = I_es[i]
        jt = J_es[j]

        # step 3: toss a biased optimized coin 
        theta_bar = (psi[it]*emp_mean[it] + psi[jt]*emp_mean[jt])/(psi[it] + psi[jt])
        coin = psi[it] * d(emp_mean[it], theta_bar)
        Ctij = coin + psi[jt] * d(emp_mean[jt], theta_bar)
        coin = coin / Ctij 

        if (rand() < coin)
            choice = it
        else
            choice = jt 
        end

        t  = t + 1
        N[choice] += 1
        S[choice] += sample_arm(theta[choice], rngList[myid()-1, choice])

        # check the stopping rule 
        # compute the test statistics Z_t 
        C = zeros(M, K-M)
        A = sortperm(emp_mean, rev=true)
        I_es = A[1:M]
        J_es = A[M+1:K]
        for i in 1:M
            for j in 1:(K-M)
                id_i = I_es[i]
                id_j = J_es[j]
                theta_bar = (psi[id_i]*emp_mean[id_i] + psi[id_j]*emp_mean[id_j])/(psi[id_i] + psi[id_j])
                C[i, j] = psi[id_i] * d(emp_mean[id_i], theta_bar) + psi[id_j] * d(emp_mean[id_j], theta_bar)
            end
        end
        condition = (t * minimum(C) < rate(t, delta))
        if (t > 1000000)
            println("Exceed the Maximum Sample Number: 1_000_000")
            return N, false
        end
    end

    return N, last(R)

end

function TSPPS_const(theta::Array, M::Integer, delta::Real, rngList)

    condition = true 
    K = length(theta)
    N = zeros(1, K)
    S = zeros(1, K)
    R = Vector{Bool}()

    order = sortperm(theta, rev=true)
    I_true = order[1:M]

    # initialization
    for a in 1:K
        N[a] = 1.0
        S[a] = sample_arm(theta[a], rngList[myid()-1, a])
    end
    t = K 

    while (condition)

        # step 0: compute true or false of the recommendation
        emp_mean = (S./N)[1,:]
        push!(R, Set(theta[sortperm(emp_mean, rev=true)[1:M]]) == Set(theta[I_true]))

        # step 1: sample theta from posterior
        theta_sample = post_sample_arm(S, N)
        A = sortperm(theta_sample, rev=true)
        I_es = A[1:M]
        J_es = A[M+1:K]


        # step 2: generate (i,j) pair as arm candidates and 
        #         evaluate P(theta_j - theta_i > 0)
        psi = (N/t)[1,:]
        if typeDistribution == "Bernoulli"
            vals = zeros(M, K-M)
            for i in 1:M
                for j in 1:(K-M)
                    id_i = I_es[i]
                    id_j = J_es[j]
                    theta_bar = (psi[id_i]*emp_mean[id_i] + psi[id_j]*emp_mean[id_j])/(psi[id_i] + psi[id_j])
                    vals[i, j] = -t * (psi[id_i] * d(emp_mean[id_i], theta_bar) + psi[id_j] * d(emp_mean[id_j], theta_bar))
                end
            end
        elseif typeDistribution == "Gaussian"
            vals = zeros(M, K-M)
            for i in 1:M
                for j in 1:(K-M)
                    id_i = I_es[i]
                    id_j = J_es[j]
                    x = (emp_mean[id_i] - emp_mean[id_j])/sigma/sqrt(1/N[id_i] + 1/N[id_j])
                    vals[i, j] = logGaussianTail(x)
                end
            end
        end
        emp_max_exp = maximum(vals)
        vals = exp.(vals .- emp_max_exp)
        vals = vals / sum(vals)

        Entry = Vector()
        Prob = Vector{Float64}()
        for i in 1:M
            for j in 1:(K-M)
                push!(Entry, (i, j))
                push!(Prob, vals[i, j])
            end
        end

        i, j = wsample(Entry, Prob)

        it = I_es[i]
        jt = J_es[j]

        # step 3: toss a  coin 
        if (rand() < 0.5)
            choice = it
        else
            choice = jt 
        end

        t  = t + 1
        N[choice] += 1
        S[choice] += sample_arm(theta[choice], rngList[myid()-1, choice])

        # check the stopping rule 
        # compute the test statistics Z_t 
        C = zeros(M, K-M)
        A = sortperm(emp_mean, rev=true)
        I_es = A[1:M]
        J_es = A[M+1:K]
        for i in 1:M
            for j in 1:(K-M)
                id_i = I_es[i]
                id_j = J_es[j]
                theta_bar = (psi[id_i]*emp_mean[id_i] + psi[id_j]*emp_mean[id_j])/(psi[id_i] + psi[id_j])
                C[i, j] = psi[id_i] * d(emp_mean[id_i], theta_bar) + psi[id_j] * d(emp_mean[id_j], theta_bar)
            end
        end
        condition = (t * minimum(C) < rate(t, delta))
        if (t > 1000000)
            println("Exceed the Maximum Sample Number: 1_000_000")
            return N, false
        end
    end

    return N, last(R)

end

function KLLUCB(theta::Array, M::Integer, delta::Real, rngList)
    #=
    KL-LUCB in "Information Complexity in Bandit Subset Selection."
    Given confidence level delta, select top M 
    return
    N : allocation times on each arm 
    R : boolean list at each time step.
    =#

    condition = true 
    K = length(theta)
    N = zeros(1, K)
    S = zeros(1, K)
    R = Vector{Bool}()

    order = sortperm(theta, rev=true)
    I_true = order[1:M]

    # initialization 
    for a in 1:K
        N[a] = 1
        S[a] = sample_arm(theta[a], rngList[myid()-1, a])
    end
    t = K 

    while (condition)
        emp_mean = (S./N)[1,:]
        push!(R, Set(theta[sortperm(emp_mean, rev=true)[1:M]]) == Set(theta[I_true]))

        A = sortperm(emp_mean, rev=true)
        Jt = A[1:M] # set of M arms with the highest empirical mean 
        JtC = A[M+1:K]

        # compute UCB in JtC 
        UCB = zeros(K-M)
        for a in 1:K-M 
            id = JtC[a]
            UCB[a] = dup(emp_mean[id], rate(t, delta)/N[id])
        end
        ut = JtC[argmax(UCB)]

        # compute LCB in Jt 
        LCB = zeros(M)
        for a in 1:M 
            id = Jt[a]
            LCB[a] = dlow(emp_mean[id], rate(t, delta)/N[id])
        end
        lt = Jt[argmin(LCB)]
        
        # draw both arms 
        t = t + 2 
        N[ut] += 1
        S[ut] += sample_arm(theta[ut], rngList[myid()-1, ut])
        N[lt] += 1
        S[lt] += sample_arm(theta[lt], rngList[myid()-1, lt])

        # check the stopping condition 
        condition = (maximum(UCB) > minimum(LCB))
        if (t > 1000000)
            println("Exceed the Maximum Sample Number: 1_000_000")
            return N, false
        end
    end
    return N, last(R)
end

function UGapE(theta::Array, M::Integer, delta::Real, rngList)
    #=
    UGapE algorithm in "BAI: A Unified Approach to Fixed Budget and Fixed Confidence"

    Given confidence level delta, select top M 
    return 
    N: allocation times on each arm 
    R: boolean list at each time step.
    =#
    condition = true 
    K = length(theta)
    N = zeros(1, K)
    S = zeros(1, K)
    R = Vector{Bool}()

    order = sortperm(theta, rev=true)
    I_true = order[1:M]

    # initialization
    for a in 1:K
        N[a] = 1
        S[a] = sample_arm(theta[a], rngList[myid()-1, a])
    end 
    t = K 

    while (condition)

        # step 0: compute true or false of the recommendation
        emp_mean = (S./N)[1,:]
        order = sortperm(emp_mean, rev=true)
        push!(R, Set(theta[order[1:M]]) == Set(theta[I_true]))

        # step 1: compute UCB and LCB 
        UCB=zeros(K)
        LCB=zeros(K)
        for a in 1:K
           UCB[a] = dup(emp_mean[a],rate(t, delta)/N[a])
           LCB[a] = dlow(emp_mean[a],rate(t, delta)/N[a])
        end

        # step 2: compute index B 
        B = zeros(K)
        for a in 1:K 
            Index = collect(1:K)
            deleteat!(Index, a)
            B[a] = sort(UCB[Index], rev=true)[M] - LCB[a]
        end

        # step 3: identify the set of M arms J(t)
        Jt = sortperm(B)[1:M]
        JtC = setdiff(order, Jt)
        ut = JtC[argmax(UCB[JtC])]
        lt = Jt[argmin(LCB[Jt])]
        
        

        choice = (N[ut] < N[lt]) ? ut : lt 
        t = t + 1
        N[choice] += 1 
        S[choice] += sample_arm(theta[choice], rngList[myid()-1, choice])

        condition = (maximum(B[Jt]) > 0)
        if (t > 1000000)
            println("Exceed the Maximum Sample Number: 1_000_000")
            return N, false
        end
    end
    return N, last(R)
end 

function Uniform(theta::Array, M::Integer, delta::Real, rngList)
    #=
    Uniform sampling
    =#
    condition = true 
    K = length(theta)
    N = zeros(1, K)
    S = zeros(1, K)
    R = Vector{Bool}()

    order = sortperm(theta, rev=true)
    I_true = order[1:M]

    # initialization 
    for a in 1:K
        N[a] = 1
        S[a] = sample_arm(theta[a], rngList[myid()-1, a])
    end

    t = K 
    while (condition)

        emp_mean = (S./N)[1,:]
        push!(R, Set(theta[sortperm(emp_mean, rev=true)[1:M]]) == Set(theta[I_true]))

        choice = Tuple(argmin(N))[2]
        N[choice] += 1 
        S[choice] += sample_arm(theta[choice], rngList[myid()-1, choice])

        psi = (N/t)[1,:]
        C = zeros(M, K-M)
        A = sortperm(emp_mean, rev=true)
        I_es = A[1:M]
        J_es = A[M+1:K]
        for i in 1:M
            for j in 1:(K-M)
                id_i = I_es[i]
                id_j = J_es[j]
                theta_bar = (psi[id_i]*emp_mean[id_i] + psi[id_j]*emp_mean[id_j])/(psi[id_i] + psi[id_j])
                C[i, j] = psi[id_i] * d(emp_mean[id_i], theta_bar) + psi[id_j] * d(emp_mean[id_j], theta_bar)
            end
        end
        condition = (t * minimum(C) < rate(t, delta))

    end

    return N, last(R)
end

function PolicyCompilation(theta::Array, M::Integer, delta::Real, rngList)
    return zeros(1, length(theta)), true
end