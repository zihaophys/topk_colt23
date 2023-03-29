import numpy as np 
from numpy.random import default_rng
rng = default_rng()
import matplotlib.pyplot as plt


def dGaussian(theta1, theta2, sigma=1):
    return (theta1 - theta2)**2/(2*sigma**2)

def dBernoulli(theta1, theta2):
    res = 0.0
    if theta1 != theta2:
        res = theta1 * np.log(theta1/theta2) + (1-theta1) * np.log((1-theta1)/(1-theta2))
    return res 

def print_mat(m):
    print('\n'.join(['\t'.join(['{}'.format(round(item,10)) for item in row])
                     for row in m]))

def projection_simplex(v, z=1):
    """
    projection on to a probability simplex
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


class TopM:
    def __init__(self, theta, M, dist='Gaussian'):
        theta = np.array(theta)
        self.theta = np.sort(theta)[::-1]
        self.K = theta.shape[0]
        self.I = M
        self.J = theta.shape[0] - M 
        if dist == 'Gaussian':
            self.d = dGaussian
        elif dist == 'Bernoulli':
            self.d = dBernoulli

    def Cfunc(self, i, j, psi):
        theta_bar = (psi[i] * self.theta[i] + psi[j] * self.theta[j])/(psi[i] + psi[j])
        # return (self.theta[j] - self.theta[i])**2/(2*sigma**2*(1.0/psi[i] + 1.0/psi[j]))
        return psi[i] * self.d(self.theta[i], theta_bar) + psi[j] * self.d(self.theta[j], theta_bar)
    
    def Cgrad(self, i, j, psi):
        grad = np.zeros(self.K)
        theta_bar = (psi[i] * self.theta[i] + psi[j] * self.theta[j])/(psi[i] + psi[j])
        grad[i] = self.d(self.theta[i], theta_bar)
        grad[j] = self.d(self.theta[j], theta_bar)
        return grad 
    
    def Cprint(self, psi):
        C = np.zeros((self.I, self.J))
        for i in range(self.I):
            for j in range(self.J):
                C[i,j] = self.Cfunc(i, j + self.I, psi)
        print_mat(C.T)


class KKTD:
    '''
    Equivalent KKT-Tracking.
    but deterministic algorithm.
    '''
    @staticmethod
    def solve(model:TopM, psi, T, c=1.0):
        res = []
        for t in range(1, T):

            C = np.zeros((model.I, model.J))
            for i in range(model.I):
                for j in range(model.J):
                    C[i,j] = model.Cfunc(i, j + model.I, psi)
            res.append(C.min())

            it = np.where(C == C.min())[0][0]
            jt = np.where(C == C.min())[1][0]
            jt += model.I 

            thetabar = (psi[it] * model.theta[it] + psi[jt] * model.theta[jt])/(psi[it] + psi[jt])
            hij = psi[it] * model.d(model.theta[it], thetabar)
            hji = psi[jt] * model.d(model.theta[jt], thetabar)
            hij = hij/(hij+hji)


            z = np.zeros(model.K)
            z[it] = hij 
            z[jt] = 1-hij

            psi = psi + c/(t+1)*(z-psi)
            # print(psi)

        return psi, res


class GD:
    '''
    Gradient ascent, choose the gradient of the minimum of C
    learning rate c, mannually tuned
    '''
    @staticmethod
    def solve(model:TopM, psi, T, c=1e-3):
        res = []
        for t in range(1, T):

            C = np.zeros((model.I, model.J))
            for i in range(model.I):
                for j in range(model.J):
                    C[i,j] = model.Cfunc(i, j + model.I, psi)
            res.append(C.min())

            it = np.where(C == C.min())[0][0]
            jt = np.where(C == C.min())[1][0]
            
            jt += model.I 

            thetabar = (psi[it] * model.theta[it] + psi[jt] * model.theta[jt])/(psi[it] + psi[jt])

            grad = np.zeros(model.K)
            grad[it] = model.d(model.theta[it], thetabar)
            grad[jt] = model.d(model.theta[jt], thetabar)
            
            psi += c*grad
            psi = projection_simplex(psi)
            # print(psi)
            
        return psi, res
    

if __name__ == '__main__':
    theta = [0.1, 0.2, 0.3, 0.4]
    M = 2
    model = TopM(theta, M, dist='Bernoulli')
    psi = np.ones(len(theta))/len(theta)
    T = 1000
    psi, res = KKTD.solve(model, psi, T, c=1.0)
    # psi, res = GD.solve(model, psi, T, c=1e-3)
    print(psi)
    plt.plot(res)
    plt.xlabel('step')
    plt.ylabel('min C')
    plt.tight_layout()
    plt.show()