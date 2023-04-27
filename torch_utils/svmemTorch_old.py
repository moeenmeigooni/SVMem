import torch
import numpy as np
import matplotlib.pyplot as plt
!pip install gpytorch
import gpytorch as gtorch
import sklearn.svm 
import warnings
warnings.simplefilter("ignore")

def sampling(N: int, amp=2, shift=3, differentiable=True):
    np.random.seed(42)
    torch.manual_seed(42)
    x = torch.linspace(-10, 10, N)
    upper = shift + amp*x.cos() + torch.randn_like(x) * 0.1
    lower = -shift + amp*x.cos() + torch.randn_like(x) * 0.1
    return torch.stack([x,upper,lower], dim=-1).requires_grad_() if differentiable else torch.stack([x,upper,lower], dim=-1)
samples = sampling(1000, differentiable=True)

def plot_samples(samples: torch.Tensor, show=False, alpha=0.3):
    UP = samples[:,:2]
    DOWN = samples[:,[0,2]]
    plt.scatter(*UP.detach().numpy().T, alpha=alpha, marker="x", c="r")
    plt.scatter(*DOWN.detach().numpy().T, alpha=alpha, marker="*", c="r")
    if show: plt.show()
    else: pass
plot_samples(samples)

def implicitDecisionBoundary(samples1: torch.Tensor, samples2: torch.Tensor, surface=False, weights=None):
    #samples #(L,2)
    covar = gtorch.kernels.ScaleKernel(
        gtorch.kernels.MaternKernel(nu=0.5, ard_num_dims=2), 
        outputscale_constraint=gtorch.constraints.Positive(),
        lengthscale_prior=gtorch.priors.GammaPrior(3,1)
        ) + gtorch.kernels.ScaleKernel(
            gtorch.kernels.RBFKernel(ard_num_dims=2, 
                                     lengthscale_constraint=gtorch.constraints.Positive(), 
                                     lengthscale_prior=gtorch.priors.GammaPrior(3,1)), 
                                     outputscale_constraint=gtorch.constraints.Positive()
            ) + gtorch.kernels.ScaleKernel(
                gtorch.kernels.RQKernel(ard_num_dims=2),
                lengthscale_prior=gtorch.priors.GammaPrior(3,1) 
                ) 
    # covar = gtorch.kernels.RBFKernel(ard_num_dims=2, 
    #                                  lengthscale_constraint=gtorch.constraints.Positive(), 
    #                                  lengthscale_prior=gtorch.priors.GammaPrior(1,1))
    # covar = gtorch.kernels.RBFKernel(ard_num_dims=2, 
    #                                  lengthscale_constraint=gtorch.constraints.Positive(), 
    #                                  lengthscale_prior=gtorch.priors.GammaPrior(1,1))  
                                       
    kernel = covar(samples1, samples2) #(L,L) instance of LazyTensor
    kernel = kernel.evaluate() #(L,L)
    if weights == None:
        #Unspecified weights for ea data point before marginalization
        weights = torch.ones_like(kernel) #L,L
    else:
        assert weights.ndim == 1, "weight dimension is wrong..."
        weights = weights.unsqueeze(dim=0).expand_as(kernel)
    if not surface:
        return kernel * weights #(L,L)
    else:
        return (kernel * weights).sum(dim=-1, keepdim=True) #L,1
# k = implicitDecisionBoundary(samples, samples, surface=True)

def differentiate_surface(samples: torch.Tensor, reference: torch.Tensor, direct: bool = False, svm: sklearn.svm.SVC = None, sample_len = None):
    samples = torch.from_numpy(samples).requires_grad_() if isinstance(samples, np.ndarray) else samples
    if svm == None:
        weights = None
    else:
        indices = svm.support_
        sample_len = sample_len if sample_len else (2 - direct) * reference.shape[0]  #if direct, use the same length; if not, multiply 2
        weights = torch.zeros(sample_len) #(l,)
        weights.data[indices] = torch.from_numpy(svm.dual_coef_.reshape(-1,)).float() #(l,) ##WIP!!
        weights = weights.view(-1,)
        # print(weights.shape, weights.count_nonzero())
        # print(svm.support_.shape, weights.nonzero(), svm.dual_coef_.shape)

    if direct:
        # Not concatenated tensor! (l,2)
        UP = reference[:,:2] #(L,2)
        DOWN = reference[:,[0,2]] #(L,2)
        reference_samples = torch.cat([UP, DOWN], dim=0) #To 2L,2 for cat;
        # reference_samples.requires_grad_()
        train_samples = torch.from_numpy(samples).to(reference_samples) if isinstance(samples, np.ndarray) else samples.to(reference_samples) #(l,2)
        train_samples.requires_grad_()
        # print(train_samples.shape, reference_samples.shape)

        kernel = implicitDecisionBoundary(train_samples, reference_samples, surface=True, weights=weights) # then get kernel (l,2L) -> weighted distance (l,)
        grad = torch.autograd.grad(kernel, train_samples, grad_outputs=torch.ones_like(kernel))[0] #(l,2) gradient
    else:
        UP = samples[:,:2] #(L,2)
        DOWN = samples[:,[0,2]] #(L,2)
        train_samples = torch.cat([UP, DOWN], dim=0) #To 2L,2 for cat;
        UP = reference[:,:2] #(L,2)
        DOWN = reference[:,[0,2]] #(L,2)
        reference_samples = torch.cat([UP, DOWN], dim=0) #To 2L,2 for cat;

        kernel = implicitDecisionBoundary(train_samples, reference_samples, surface=True, weights=weights) # then get kernel (2L,2L) -> weighted distance (L,)
        grad = torch.autograd.grad(kernel, train_samples, grad_outputs=torch.ones_like(kernel))[0] #(L,2) gradient
    return grad, kernel #gradient is DelF(x,y,z) and kernel is F(x,y,z)=k value! computed by weighted sum...

def svm_and_dual(samples: torch.Tensor):
    UP = samples[:,:2] #(L,2)
    DOWN = samples[:,[0,2]] #(L,2)
    # print(UP)
    pos = torch.ones(UP.shape[0]).view(-1,1) #y (L,1)
    neg = -torch.ones(UP.shape[0]).view(-1,1) #y (L,1)
    pos_sam = torch.cat([UP, pos], dim=-1) #L,3
    neg_sam = torch.cat([DOWN, neg], dim=-1) #L,3
    sams = torch.cat([pos_sam, neg_sam], dim=0) #(2L,3)
    train_samples = torch.cat([UP, DOWN], dim=0) #To 2L,2 for cat;

    kernel = implicitDecisionBoundary(train_samples, train_samples, surface=False) # then get kernel (2L,2L)
    svc = sklearn.svm.SVC(kernel="precomputed")
    kernel.detach_().numpy() #2L,2L
    sams_npy = sams.detach().numpy() #2L,3
    svc.fit(kernel, sams_npy[:,[-1]])
    return svc
svc = svm_and_dual(samples)

def create_svm_decision_boundary(svm: sklearn.svm.SVC, N: int, samples: torch.Tensor):
    xranges = torch.linspace(-10,10,N)
    yranges = torch.linspace(-10,10,N)
    X, Y = torch.meshgrid(xranges, yranges)
    shapes = X.detach().numpy().shape
    test_samples = torch.stack([X.ravel(), Y.ravel()], dim=-1) #Batch, 2; Tensor
    test_samples_npy = test_samples.detach().numpy() #Batch, 2; numpy
    UP = samples[:,:2] #(L,2)
    DOWN = samples[:,[0,2]] #(L,2)
    train_samples = torch.cat([UP, DOWN], dim=0) #(2L,2)
    kernel = implicitDecisionBoundary(test_samples, train_samples, surface=False) #To (Batch, 2L)
    kernel.detach_()
    kernel = kernel.numpy()

    Z = svm.decision_function(kernel) #Batch, ncls; numpy; must use precomputed format; F(x,y,z) = 0/+-1 etc.
    Z_npy = Z.reshape(*shapes)
    X_npy, Y_npy = X.detach().numpy(), Y.detach().numpy()

    plt.contourf(X_npy, Y_npy, Z_npy, alpha=0.8, cmap=plt.cm.get_cmap("gnuplot2"))
    plt.colorbar()
    CS = plt.contour(X_npy, Y_npy, Z_npy, levels=[-1, 0, 1], alpha=1, linestyles=["--", "-", "--"], colors=["k", "k", "k"], )
    # plot_samples(samples, show=False, alpha=.8) #scatter plot
    # plt.scatter(*train_samples[svm.support_].detach().numpy().T, s=100, linewidth=1, facecolors="none", edgecolors="k",)
    level1 = CS.allsegs[1][0] #Fetch the coordinates of boundary condition! (WITHOUT F(x,y,z)=0 implicit function!!)
    plt.scatter(*level1.T, c="g") #Make sure boundary is drawn
    # level1 = CS.allsegs[1][1] #Fetch the coordinates of boundary condition! (WITHOUT F(x,y,z)=0 implicit function!!)
    # plt.scatter(*level1.T, c="g") #Make sure boundary is drawn

    #INFERENCE
    what_samples = train_samples #train_samples, test_samples, level1
    # grad, surf = differentiate_surface(samples, reference=samples, svm=svm) #(2L,2) ; weighted sum
    grad, surf = differentiate_surface(what_samples, reference=samples, direct=True, svm=svm, sample_len=samples.shape[0]*2) #(2L,2) ; weighted sum
    # signed_grad = - grad.data.div(grad.data.norm(dim=-1, keepdim=True)) * surf.sign() #SIGN should be considered for gradient info!
    signed_grad = - grad.data * surf.sign() #SIGN should be considered for gradient info!
    if isinstance(what_samples, torch.Tensor): plt.quiver(*what_samples.detach().numpy().T, *signed_grad.detach().numpy().T, width=.005)
    else: plt.quiver(*what_samples.T, *signed_grad.detach().numpy().T, width=.005)
    plt.title("Decision Boundary")
    plt.show()

    plt.hist(surf.detach().numpy().reshape(-1,), bins=50)
    plt.axvline(surf.detach().numpy().reshape(-1,).mean(), c="r", linewidth=4)
    plt.title("Distribution of Implicit Function Values")
    plt.show()

    try: print(np.allclose(Z, surf.detach().numpy().reshape(-1,), atol=5e-4)) #True: LIBSVM decision bounary (i.e. F(x,y,z)=k and manual F(x,y,z)=k are the same!)
    except Exception as e: print(e)
    print("Done!")

    return Z_npy, CS
db, CS = create_svm_decision_boundary(svm=svc, N=100, samples=samples)



