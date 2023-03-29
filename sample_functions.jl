function sample_arm_Bernoulli(theta::Float64, rngStream)
    return rand(rngStream, Float64) < theta
end

function sample_arm_Gaussian(theta::Float64, rngStream)
    noise = randn(rngStream, Float64)
    # println(rngStream)
    return theta + sigma * noise
end

function post_sample_arm_Bernoulli(S, N)
    K = size(N)[2]
    theta = zeros(K)
    for a in 1:K
        theta[a] = rand(Beta(1 + S[a], 1 + N[a] - S[a]), 1)[1]
    end
    return theta
end

#%%%%%% Function to sample from the posterior of the Gaussian arm with prior N(0, sigma^2) %%%%%%%
# function post_sample_arm_Gaussian(S, N)
#     K = size(N)[2]
#     theta = zeros(K)
#     for a in 1:K
#         posmean = S[a]/(N[a] + sigma^2)
#         posvar = 1 / (1 + N[a]/(sigma^2))
#         theta[a] = rand(Normal(posmean, sqrt(posvar)), 1)[1]
#     end
#     return theta
# end

#%%%%%% Function to sample from the posterior of the Gaussian arm with improper prior %%%%%%%%%%%%%
function post_sample_arm_Gaussian(S, N)
    K = size(N)[2]
    theta = zeros(K)
    for a in 1:K
        theta[a] = rand(Normal(S[a]/N[a], sigma/sqrt(N[a])), 1)[1]
    end
    return theta
end