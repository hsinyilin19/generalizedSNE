import numpy as np
import matplotlib.pyplot as pyplot
import argparse
import torch
from scipy.io import loadmat
import os

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=int, default=0, help="if use cuda accelarate")
parser.add_argument('--N', type=int, default=6000, help="number of samples")
parser.add_argument('--max_iter', type=int, default=20000)
parser.add_argument('--stop_lying_iter', type=int, default=1000)
parser.add_argument('--mom_switch_iter', type=int, default=50)
parser.add_argument('--epsilon', type=int, default=500)
parser.add_argument('--initial_momentum', type=float, default=0.3)
parser.add_argument('--final_momentum', type=float, default=0.8)
parser.add_argument('--min_gain', type=float, default=0.01)
parser.add_argument('--dgain', type=float, default=0.2)
parser.add_argument('--r_crit', type=float, default=0.1)
parser.add_argument('--r0', type=float, default=1.0, help="initial random radius")

parser.add_argument('--no_dims', type=int, default=2, help="target dim for data representation")
parser.add_argument('--initial_dims', type=int, default=50, help='initial dim for SNE(after PCA reduction)')
parser.add_argument('--perplexity', type=int, default=30, help='detetermine  size  of data neighborhood')

parser.add_argument('--alpha', type=float, default=-0.5) # α - divergence[α = -1(KL), α -> 0(Hellinger)]
parser.add_argument('--beta', type=float, default=3.0, help='power-law exponent')
parser.add_argument('--eta', type=float, default=0.1, help='eta')

args = parser.parse_args()
print("get choice from args", args)
if args.cuda:
    print("set use cuda")
    torch.set_default_dtype(torch.float64)
else:
    torch.set_default_dtype(torch.float64)


def Hbeta_torch(D, beta=1.0):
    beta=beta.to(device)
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P


def x2p_torch(X, tol=1e-5, perplexity=30):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n).to(device)
    beta = torch.ones(n, 1).to(device)
    logU = torch.log(torch.tensor([perplexity])).to(device)
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P


def pca_torch(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    X = torch.from_numpy(X).type(torch.DoubleTensor)
    X = X - torch.mean(X, 0)

    (l, M) = torch.linalg.eig(torch.mm(X.t(), X))

    l = l.real
    M = M.real

    Y = torch.mm(X, M[:, 0:no_dims])
    return Y


def gsne(X, no_dims=2, initial_dims=50, perplexity=30, alpha=-1.0, beta=2, eta=0.075, N=3000,
         max_iter=10000, stop_lying_iter=100, initial_momentum=0.5, final_momentum=0.8, mom_switch_iter=50,
         epsilon=100, min_gain=0.01, dgain=0.2, r0=0.3):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    X = pca_torch(X, initial_dims)
    (n, d) = X.shape
    X=X.to(device)

    Y = r0 * torch.randn(n, no_dims).to(device)

    dY = torch.zeros(n, no_dims).to(device)
    iY = torch.zeros(n, no_dims).to(device)
    gains = torch.ones(n, no_dims).to(device)

    # Compute P-values
    P = x2p_torch(X, 1e-5, perplexity)
    P.fill_diagonal_(0)
    P = 2 * (P + P.t())

    if torch.sum(P)==0:
        P = torch.tensor(torch.finfo(torch.float32).tiny)
    else:
        P = P / torch.sum(P)

    print("get P shape", P.shape)

    P = P * 4.  # early exaggeration

    def f(t):
        return (4 / (1 - alpha ** 2)) * ((1 - alpha) / 2 + (1 + alpha) / 2 * t - t ** ((alpha + 1) / 2))

    def df(t):
        return (2 / (1 - alpha)) * (1 - t ** ((alpha - 1) / 2))

    def Q(x):
        result = 1 / ((x + eta) ** beta)
        return result

    def dQ(x):
        result = -beta * (x ** (beta-1)) / ((eta + x ** beta ) ** 2)
        return result

    # Run iterations
    for iter in range(max_iter):
# -----------------------------------------------
#         # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)

        sqnum = sum_Y[:, np.newaxis] + sum_Y - 2 * torch.matmul(Y, Y.T)

        r = torch.sqrt(torch.max(sqnum,torch.tensor(torch.finfo(torch.float32).tiny))).real
        Qr = Q(r)
        Qr.fill_diagonal_(0)
        dQr = dQ(r)
        dQr.fill_diagonal_(0)

        Z = torch.sum(Qr)
        q = torch.max(Qr / Z, torch.tensor(torch.finfo(torch.float32).tiny))

        q_P = q / P
        q_P.fill_diagonal_(0)  # Set the diagonal elements to 0
        dfq_P = df(q_P)
        dfq_P.fill_diagonal_(0)
        L = dQr * (dfq_P - (1 / Z) * torch.sum(torch.sum(dfq_P * Qr, dim=1), dim=0))
        L.fill_diagonal_(0)


        L_r = L/r
        L_r.fill_diagonal_(0)
        L_sum = torch.sum(L_r, 1)
        diag_matrix = torch.diag(L_sum)
        dY = (2 / Z) * torch.matmul((diag_matrix - L_r), Y)

        # Perform the update
        if iter < mom_switch_iter:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        #update the solution
        gains = (gains + dgain) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - epsilon * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = (P * f(q_P)).fill_diagonal_(0)
            C = C.sum()

            # C = torch.sum(torch.sum(P.view(-1) * torch.log(P.view(-1))) - torch.sum(P.view(-1) * torch.log(Q.view(-1))))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == stop_lying_iter:
            P = P / 4.

        if (iter + 1) % 5000 == 0:
            Y_np = Y.cpu()
            path = './Y_files/N{}/'.format(N)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            np.save(path+'Y_np.npy', Y_np)

            pyplot.scatter(Y_np[:, 0], Y_np[:, 1], 10, labels)
            pyplot.title('gSNE, alpha={}, beta={}, eta={}'.format(alpha, beta, eta))

            path = './plots/N{}/'.format(N)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            pyplot.savefig(path+'plot.png')
            pyplot.close()
            print('iteration {} plot saved!'.format(iter + 1))

    # Return solution
    return Y


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")

    data = loadmat('mnist.mat')

    trainX = data['trainX']
    testX = data['testX']
    trainY = data['trainY']
    testY = data['testY']

    N = args.N

    X = (trainX[:N,:]+1)/255
    labels = trainY[:, :N].reshape(-1)

    # confirm that x file get same number point than label file
    # otherwise may cause error in scatter
    assert(len(X[:, 0]) == len(X[:, 1]))
    assert(len(X) == len(labels))

    with (((((((torch.no_grad()))))))):
        no_dims = args.no_dims   #(target)Dimensionfor data representation
        initial_dims = args.initial_dims   #initial dim for SNE(after PCA reduction)
        perplexity = args.perplexity   #detetermine  size  of data neighborhood
        alpha = args.alpha  #alpha - divergence[alpha = -1(KL), alpha -> 0(Hellinger)]
        beta = args.beta   #power - law exponent( if choosing 'pSNE' in previous method)
        eta = args.eta
        max_iter = args.max_iter
        stop_lying_iter = args.stop_lying_iter
        initial_momentum = args.initial_momentum
        final_momentum = args.final_momentum
        mom_switch_iter = args.mom_switch_iter
        epsilon = args.epsilon
        min_gain = args.min_gain
        dgain = args.dgain
        r0 = args.r0

        use_cude = torch.cuda.is_available()

        device = torch.device("cuda:0" if use_cude else "cpu")


        Y = gsne(X, no_dims=no_dims, initial_dims=initial_dims, perplexity=perplexity, alpha=alpha, beta=beta, eta=eta, N=N,
                 max_iter=max_iter, stop_lying_iter=stop_lying_iter, initial_momentum=initial_momentum,
                 final_momentum=final_momentum, mom_switch_iter=mom_switch_iter, epsilon=epsilon,
                 min_gain=min_gain, dgain=dgain, r0=r0)

