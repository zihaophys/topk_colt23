function m(i, j, x, theta::Array)
   return (theta[i] + x * theta[j])/(1 + x)
end

function dBernoulli(p,q)
    res=0
    if (p!=q)
       if (p<=0) p = eps() end
       if (p>=1) p = 1-eps() end
       if (q<=0) q = eps() end
       if (q>=1) q = 1-eps() end
       res=(p*log(p/q) + (1-p)*log((1-p)/(1-q))) 
    end
    return(res)
end

function dupBernoulli(p,level)
   # KL upper confidence bound:
   # return qM>p such that d(p,qM)=level 
   lM = p 
   uM = min(min(1,p+sqrt(level/2)),1) 
   for j = 1:16
      qM = (uM+lM)/2
      if dBernoulli(p,qM) > level
         uM= qM
      else
         lM=qM
      end
   end
   return(uM)
end
   
   
function dlowBernoulli(p,level)
   # KL lower confidence bound:
   # return lM<p such that d(p,lM)=level
   lM = max(min(1,p-sqrt(level/2)),0) 
   uM = p 
   for j = 1:16
      qM = (uM+lM)/2;
      if dBernoulli(p,qM) > level
         lM= qM;
      else
         uM=qM;
      end
   end
   return(lM)
end

function thetaToetaBernoulli(theta)
   return log(theta/(1-theta))
end

function etaTothetaBernoulli(eta)
   return 1.0/(1+exp(-eta))
end

  
function dPoisson(p,q)
    if (p==0)
       res=q
    else
       res=q-p + p*log(p/q)
    end
    return(res)
end

function dExpo(p,q)
    res=0
    if (p!=q)
       if (p<=0)|(q<=0)
          res=Inf
       else
          res=p/q - 1 - log(p/q)
       end
    end
    return(res)
end

function dGaussian(p,q)
    (p-q)^2/(2*sigma^2)
end

function dupGaussian(p,level)
   p+sigma*sqrt(2*level)
end

function dlowGaussian(p,level)
   p-sigma*sqrt(2*level)
end

function thetaToetaGaussian(theta)
   return theta/sigma^2
end

function etaTothetaGaussian(eta)
   return eta*sigma^2
end

