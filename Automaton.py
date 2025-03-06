import torch,torch.nn,torch.nn.functional as F
import numpy as np
from torchenhanced import DevModule
from utils.noise_gen import perlin,perlin_fractal
from utils.main_utils import gen_batch_params,gen_params
import matplotlib.pyplot as plt
import matplotlib.animation
import pickle
from utils.voronoi_utils import load_pattern, orbium
import os
from utils.hash_params_to_names import get_params_name
import time



class BatchLeniaMC(DevModule):
    """
        Batched Multi-channel lenia, to run batch_size worlds in parallel !
        Does not support live drawing in pygame, maybe will later.
    """
    def __init__(self, size, dt, num_channels=1, params=None, state_init = None, device='cpu', multi_step_size = 1,):
        """
            Initializes automaton.  

            Args :
                size : (B,H,W) of ints, size of the automaton and number of batches
                dt : time-step used when computing the evolution of the automaton
                num_channels : int, number of channels (C) in the automaton
                params : dict of tensors containing the parameters. If none, generates randomly
                    keys-values : 
                    'k_size' : odd int, size of kernel used for computations
                    'mu' : (B,C,C) tensor, mean of growth functions
                    'sigma' : (B,C,C) tensor, standard deviation of the growth functions
                    'beta' :  (B,C,C, # of rings) float, max of the kernel rings 
                    'mu_k' : (B,C,C, # of rings) [0,1.], location of the kernel rings
                    'sigma_k' : (B,C,C, # of rings) float, standard deviation of the kernel rings
                    'weights' : (B,C,C) float, weights for the growth weighted sum
                device : str, device 
        """
        super().__init__()
        self.to(device)

        self.batch= size[0]
        self.h, self.w  = size[1:]
        self.C = num_channels

        self.multi_step_size = multi_step_size
        

        if(params is None):
            # Generates random parameters
            params = gen_batch_params(self.batch,device,num_channels=self.C)
           
        self.params = params
        self.name = get_params_name(params)

        self.k_size = params['k_size'] # kernel sizes (same for all) MUST BE ODD !!!
        self.radius = (self.k_size-1)//2

        self.register_buffer('state',torch.rand((self.batch,self.C,self.h,self.w)))

        if(state_init is None):
            self.set_init_fractal() # Fractal perlin init
        else:
            self.state = state_init.to(self.device) # Specific init

        self.dt = dt

        try:
            with open(f'utils/polygons{self.h}.pickle', 'rb') as handle:
                polygons = pickle.load(handle)

            self.polygons = polygons
        except:
            print("polygons for this array size not generated yet")

        array_size = self.w
        ii, jj = torch.meshgrid(torch.arange(0, array_size), torch.arange(0, array_size), indexing='ij')
        coords = torch.stack([torch.reshape(ii, (-1,)), torch.reshape(jj, (-1,))], axis=-1).float().to(device).view(array_size**2, 2, 1).to(device)

        self.coords = coords


        # Buffer for all parameters since we do not require_grad for them :
        self.register_buffer('mu', params['mu']) # mean of the growth functions (B,C,C)
        self.register_buffer('sigma', params['sigma']) # standard deviation of the growths functions (B,C,C)
        self.register_buffer('beta',params['beta']) # max of the kernel rings (B,C,C, # of rings)
        self.register_buffer('mu_k',params['mu_k'])# mean of the kernel gaussians (B,C,C, # of rings)
        self.register_buffer('sigma_k',params['sigma_k'])# standard deviation of the kernel gaussians (B,C,C, # of rings)
        self.register_buffer('weights',params['weights']) # raw weigths for the growth weighted sum (B,C,C)

        self.norm_weights()
        self.register_buffer('kernel',torch.zeros((self.k_size,self.k_size)))
        self.kernel = self.compute_kernel() # (B,C,C,h, w)
        self.kernel_eff = self.kernel.reshape([self.batch*self.C*self.C,1,self.k_size,self.k_size]) #(B*C^2,1,k,k)
        #self.plot_kernel()

        self.seed = None

        mu = self.mu[0][0]
        sigma = self.sigma[0][0]
        f = lambda u : 2*torch.exp(-((u-mu)**2/(sigma)**2)/2)*(-2)*(u-mu)/(sigma)**2/2
        self.dg = f

        f = lambda u : 2*torch.exp(-((u-mu)**2/(sigma)**2)/2)-1
        self.g = f


    def plot_kernel(self):
        if self.C == 1:
            kernel = self.kernel[0, 0, :].cpu().numpy()
            print(kernel.shape)
            knl = load_pattern(kernel, [self.k_size+1, self.k_size+1])
            plt.imshow(1-(knl)[0,:,:], cmap="binary")
            plt.grid(axis='x', color='0.95')
            plt.axis('off')
            plt.savefig("cont_kernel.png", bbox_inches='tight')
            plt.close()
        else:
            print("not yet implemented yet")





    def update_params(self, params):
        """
            Updates some or all parameters of the automaton. 
            Changes batch size to match the one of provided params (take mu as reference)
        """
        self.mu = params.get('mu',self.mu)
        self.sigma = params.get('sigma',self.sigma)
        self.beta = params.get('beta',self.beta)
        self.mu_k = params.get('mu_k',self.mu_k)
        self.sigma_k = params.get('sigma_k',self.sigma_k)
        self.weights = params.get('weights',self.weights)
        self.k_size = params.get('k_size',self.k_size) # kernel sizes (same for all)

        self.norm_weights()

        self.batch = self.mu.shape[0] # update batch size
        self.kernel = self.compute_kernel() # (B,C,C,h,w)

    def hash_config(self):
        pickled = pickle.dumps(self.state.cpu().numpy())
        return pickled

    
    def norm_weights(self):
        """
            Normalizes the relative weight sum of the growth functions
            (A_j(t+dt) = A_j(t) + dt G_{ij}w_ij), here we enforce sum_i w_ij = 1
        """
        # Normalizing the weights
        N = self.weights.sum(dim=1, keepdim = True) # (B,1,C)
        self.weights = torch.where(N > 1.e-6, self.weights/N, 0)

    def get_params(self):
        """
            Get the parameter dictionary which defines the automaton
        """
        params = dict(k_size = self.k_size,mu = self.mu, sigma = self.sigma, beta = self.beta,
                       mu_k = self.mu_k, sigma_k = self.sigma_k, weights = self.weights)
        
        return params

    def set_init_fractal(self, wave=1.5):
        """
            Sets the initial state of the automaton using fractal perlin noise.
            Max wavelength is k_size*1.5, chosen a bit randomly
        """
        self.state = perlin_fractal((self.batch,self.h,self.w),int(self.k_size*wave),
                                    device=self.device,black_prop=0.25,num_channels=self.C,persistence=0.4) 
    
    def set_init_perlin(self,wavelength=None, seed=None):
        """
            Sets initial state using one-wavelength perlin noise.
            Default wavelength is 2*K_size
        """
        if(not wavelength):
            wavelength = self.k_size
        self.seed, self.state = perlin((self.batch,self.h,self.w),[wavelength]*2,
                            device=self.device,num_channels=self.C,black_prop=0.25, seed=seed)
        
    
    def set_perturbed(self,configs, rand_coef=0.05, seed=None):
        if not seed:
            seed = np.random.randint(2**32)

        if not self.seed:
            self.seed = seed

        g = torch.Generator().manual_seed(self.seed)

        noise = rand_coef*torch.rand(*configs.shape).to(self.device)

        self.state = configs+noise
        
    def set_init_voronoi(self, polygon_size=60, sample=0, seed=None):
        """
            Sets initial state using one-wavelength perlin noise.
            Default wavelength is 2*K_size
        """
   
        mask = self.polygons[polygon_size][sample%1024]
        mask = load_pattern(mask.reshape(1, *mask.shape), [self.h, self.w]).reshape(self.h, self.w)

        if not seed:
            seed = np.random.randint(2**32)
        g = torch.Generator().manual_seed(seed)

        self.seed = seed

        rand = torch.rand(self.batch, self.C, self.h, self.w, generator=g)
        rand_np = rand.cpu().numpy()

        pattern = np.asarray(rand_np * mask)
        self.state = torch.tensor(np.asarray(pattern)).view(self.batch,self.C,self.h,self.w).float().to(self.device)

    def set_init_voronoi_batch(self, polygon_size=60, batch_index=0, seeds=None):
        if not seeds:
            seeds = [np.random.randint(2**32) for _ in range(self.batch)]
        elif not len(seeds)==self.batch:
            print("number of seeds does not match batch size, reinitializing seeds")
            seeds = [np.random.randint(2**32) for _ in range(self.batch)]

        self.seeds = seeds

        states = torch.empty((self.batch, self.C, self.h, self.w)).to(self.device) # (B,C,H,W)
        for i, seed in enumerate(seeds):
            polygon_index = batch_index+i
            mask = self.polygons[polygon_size][polygon_index%1024]
            mask = load_pattern(mask.reshape(1, *mask.shape), [self.h, self.w]).reshape(self.h, self.w)

            g = torch.Generator().manual_seed(seed)

            rand = torch.rand(1, self.C, self.h, self.w, generator=g)
            rand_np = rand.cpu().numpy()

            pattern = np.asarray(rand_np * mask)
            state = torch.tensor(np.asarray(pattern)).view(1,self.C,self.h,self.w).float().to(self.device)

            states[i, :, :, :] = state
        
        self.state = states  # (B,C,H,W)

    def set_init_voronoi_batch_range(self, polygon_size_range, batch_index, mini_batch_size, seeds=None):
        assert self.batch == len(polygon_size_range*mini_batch_size)

        if not seeds:
            seeds = [np.random.randint(2**32) for _ in range(self.batch)]
        elif not len(seeds)==self.batch:
            print("number of seeds does not match batch size, reinitializing seeds")
            seeds = [np.random.randint(2**32) for _ in range(self.batch)]

        self.seeds = seeds

        states = torch.empty((self.batch, self.C, self.h, self.w)).to(self.device) # (B,C,H,W)
        for i, seed in enumerate(seeds):
            polygon_size = polygon_size_range[i//mini_batch_size]

            polygon_index = batch_index+i
            mask = self.polygons[polygon_size][polygon_index%1024]
            mask = load_pattern(mask.reshape(1, *mask.shape), [self.h, self.w]).reshape(self.h, self.w)

            g = torch.Generator().manual_seed(seed)

            rand = torch.rand(1, self.C, self.h, self.w, generator=g)
            rand_np = rand.cpu().numpy()

            pattern = np.asarray(rand_np * mask)
            state = torch.tensor(np.asarray(pattern)).view(1,self.C,self.h,self.w).float().to(self.device)

            states[i, :, :, :] = state
        
        self.state = states  # (B,C,H,W)


    def plot_init_config(self, channel=0):
        fig_w = int(np.ceil(np.sqrt(self.batch)))
        fig_h = int(self.batch//fig_w)

        fig, axs = plt.subplots(fig_h, fig_w)

        for i in range(fig_w):
            for j in range(fig_h):
                config = self.state[i*fig_h+j, channel, :, :].cpu().numpy()
                axs[j, i].imshow(config, cmap="gray")
                axs[j, i].set_axis_off()
        
        plt.tight_layout()
        plt.savefig(f"{self.name}_init.png")



    
    def get_init_voronoi(self, polygon_size, sample=0, seed=None):
        mask = self.polygons[polygon_size][sample%1024]
        mask = load_pattern(mask.reshape(1, *mask.shape), [self.h, self.w]).reshape(self.h, self.w)

        if not seed:
            seed = np.random.randint(2**32)
        g = torch.Generator().manual_seed(seed)

        self.seed = seed

        rand = torch.rand(self.h, self.w, generator=g)
        rand_np = rand.cpu().numpy()

        pattern = np.asarray(rand_np * mask)
        self.state = torch.tensor(np.asarray(pattern)).view(self.h*self.w).float().to(self.device)

        return self.state

        

    def set_init_voronoi_wave(self, polygon_size=60, wave=None, black_prop=0.5, sample=0, seed=None):
        """
            Sets initial state using one-wavelength perlin noise.
            Default wavelength is 2*K_size
        """
        if(not wave):
            wave = self.k_size
   
        mask = self.polygons[polygon_size][sample%1024]
        mask = load_pattern(mask.reshape(1, *mask.shape), [self.h, self.w]).reshape(self.h, self.w)

        if not seed:
            seed = np.random.randint(2**32)
        g = torch.Generator().manual_seed(seed)

        self.seed = seed


        seed, rand = perlin((self.batch,self.h,self.w),[wave]*2,
                            device=self.device,num_channels=self.C,black_prop=black_prop, seed=seed)
        
        rand_np = rand.cpu().numpy()

        pattern = np.asarray(rand_np * mask)
        self.state = torch.tensor(np.asarray(pattern)).view(self.batch,self.C,self.h,self.w).float().to(self.device)
        
        
    def kernel_slice(self, r):
        """
            Given a distance matrix r, computes the kernel of the automaton.
            In other words, compute the kernel 'cross-section' since we always assume
            rotationally symmetric kernel

            Args :
            r : (k_size,k_size), value of the radius for each pixel of the kernel
        """
        # Expand radius to match expected kernel shape
        r = r[None, None, None,None] #(1,1, 1, 1, k_size, k_size)
        r = r.expand(self.batch,self.C,self.C,self.mu_k.shape[3],-1,-1) #(B,C,C,#of rings,k_size,k_size)

        mu_k = self.mu_k[..., None, None] # (B,C,C,#of rings,1,1)
        sigma_k = self.sigma_k[..., None, None]# (B,C,C,#of rings,1,1)

        K = torch.exp(-((r-mu_k)/sigma_k)**2/2) #(B,C,C,#of rings,k_size,k_size)

        beta = self.beta[..., None, None] # (B,C,C,#of rings,1,1)
        K = torch.sum(beta*K, dim = 3) #

        
        return K #(B,C,C,k_size, k_size)
    
    def compute_kernel(self):
        """
            Computes the kernel given the current parameters.
        """
        xyrange = torch.arange(-1, 1+0.00001, 2/(self.k_size-1)).to(self.device)
        X,Y = torch.meshgrid(xyrange, xyrange,indexing='ij')
        r = torch.sqrt(X**2+Y**2)

        K = self.kernel_slice(r) #(B,C,C,k_size,k_size)

        # Normalize the kernel, s.t. integral(K) = 1
        summed = torch.sum(K, dim = (-1,-2), keepdim=True) #(B,C,C,1,1)

        # Avoid divisions by 0
        summed = torch.where(summed<1e-6,1,summed)
        K /= summed

        return K #(B,C,C,k_size,k_size)
    
    def growth(self, u): # u:(B,C,C,H,W)
        """
            Computes the growth of the automaton given the concentration u.

            Args :
            u : (B,C,C,H,W) tensor of concentrations.
        """

        # Possibly in the future add other growth function using bump instead of guassian
        mu = self.mu[..., None, None] # (B,C,C,1,1)
        sigma = self.sigma[...,None,None] # (B,C,C,1,1)
        mu = mu.expand(-1,-1,-1, self.h, self.w) # (B,C,C,H,W)
        sigma = sigma.expand(-1,-1,-1, self.h, self.w) # (B,C,C,H,W)

        return 2*torch.exp(-((u-mu)**2/(sigma)**2)/2)-1 #(B,C,C,H,W)


    def step(self):
        """
            Steps the automaton state by one iteration.
        """
   
        
        U = self.state.reshape(1,self.batch*self.C,self.h,self.w) # (1,B*C,H,W)
        U = F.pad(U, [(self.k_size-1)//2]*4, mode = 'circular') # (1,B*C,H+pad,W+pad)
        
        U = F.conv2d(U, self.kernel_eff, groups=self.C*self.batch).squeeze(1) #(B*C^2,1,H,W) squeeze to (B*C^2,H,W)
        U = U.reshape(self.batch,self.C,self.C,self.h,self.w) # (B,C,C,H,W)

        assert (self.h,self.w) == (self.state.shape[2], self.state.shape[3])

        weights = self.weights[...,None, None] # (B,C,C,1,1)
        weights = weights.expand(-1,-1, -1, self.h,self.w) # (B,C,C,H,W)

        # Weight normalized growth :
        dx = (self.growth(U)*weights).sum(dim=1) #(B,C,H,W)

        # Apply growth and clamp
        self.state = torch.clamp(self.state + self.dt*dx, 0, 1)     

    
    def grad_step(self, A):
        """
            Steps the automaton state by one iteration.
        """
        B = A.view(self.batch,self.C,self.h,self.w) #(B,C,H,W)

        U = A.reshape(1,self.batch*self.C,self.h,self.w) # (1,B*C,H,W)
        U = F.pad(U, [(self.k_size-1)//2]*4, mode = 'circular') # (1,B*C,H+pad,W+pad)
        
        U = F.conv2d(U, self.kernel_eff, groups=self.C*self.batch).squeeze(1) #(B*C^2,1,H,W) squeeze to (B*C^2,H,W)
        U = U.reshape(self.batch,self.C,self.C,self.h,self.w) # (B,C,C,H,W)


        weights = self.weights[...,None, None] # (B,C,C,1,1)
        weights = weights.expand(-1,-1, -1, self.h,self.w) # (B,C,C,H,W)

        # Weight normalized growth :
        dx = (self.growth(U)*weights).sum(dim=1) #(B,C,H,W)

        # Apply growth and clamp
        A = torch.clamp(B + self.dt*dx, 0, 1)    
     
        return A.view(self.w*self.h)

    def grad_multi_step(self, A):
        """
            Steps the automaton state by one iteration.
        """

        for _ in range(self.multi_step_size):
            B = A.view(self.batch,self.C,self.h,self.w) #(B,C,H,W)

            U = A.reshape(1,self.batch*self.C,self.h,self.w) # (1,B*C,H,W)
            U = F.pad(U, [(self.k_size-1)//2]*4, mode = 'circular') # (1,B*C,H+pad,W+pad)
            
            U = F.conv2d(U, self.kernel_eff, groups=self.C*self.batch).squeeze(1) #(B*C^2,1,H,W) squeeze to (B*C^2,H,W)
            U = U.reshape(self.batch,self.C,self.C,self.h,self.w) # (B,C,C,H,W)


            weights = self.weights[...,None, None] # (B,C,C,1,1)
            weights = weights.expand(-1,-1, -1, self.h,self.w) # (B,C,C,H,W)

            # Weight normalized growth :
            dx = (self.growth(U)*weights).sum(dim=1) #(B,C,H,W)

            # Apply growth and clamp
            A = torch.clamp(B + self.dt*dx, 0, 1)    
     

        return A.view(self.w*self.h)
    
    def jacob(self, A):
        J = torch.autograd.functional.jacobian(self.grad_multi_step, A)
        return J

    
    def get_eig(self, J):
        J = J.to("cpu")
        Jt = torch.transpose(J, 0, 1)
        M = torch.matmul(Jt, J)
        largest = torch.lobpcg(M, k=100, largest=True)[0]
        return largest
    

    def get_lyapunov_seq(self, max_it, input, pretime, step):
        for _ in range(pretime):
            input = self.grad_step(input)

        A = input.clone().detach() 
        largest_eigs = []
        M = torch.eye(self.h*self.w).float().to(self.device)

        for t in range(max_it):
            print("computing iterate", t)
            J = self.jacob(A)
            A = self.grad_multi_step(A)
            M = J @ M
            print(M.shape)
            if not (t+1)%step:
                largest = self.get_eig(M)
                largest_eigs.append(largest)

        return largest_eigs




    def mass(self):
        """
            Computes average 'mass' of the automaton for each channel

            returns :
            mass : (B,C) tensor, mass of each channel
        """

        return self.state.mean(dim=(-1,-2)) # (B,C) mean mass for each color

    def draw(self):
        """
            Draws the RGB worldmap from state.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"
        toshow= self.state[0].permute((2,1,0)) # (W,H,C) for pygame

        if(self.C==1):
            toshow = toshow.expand(-1,-1,3)
        elif(self.C==2):
            toshow = torch.cat([toshow,torch.zeros_like(toshow)],dim=-1)
        else :
            toshow = toshow[:,:,:3]

        return toshow.cpu().numpy()
    
        self._worldmap= toshow.cpu().numpy()   
    
        
    @property
    def worldmap(self):
        return (255*self._worldmap).astype(dtype=np.uint8)
    
    def make_video(self, time_steps = 100, start_steps = 0, step=1, seed = None, wave = None, config = None, video_path="video.gif"):
        if config is not None:
            print("loading config for video")
            self.state = config
        elif seed:
            self.set_init_perlin(wave, seed)
        else:
            self.set_init_perlin(wave)
        
        
        A = np.zeros((time_steps, self.w, self.h, 3), dtype=float)
        for t in range(start_steps):
            self.step()

        A[0, :, :, :] = self.draw()

        for t in range(time_steps-1):
            self.step()
            A[t, :, :, :] =self.draw()
    

        if time_steps > 800:
            step = 10
        elif time_steps > 8000:
            step = 100

        
        fig = plt.figure(figsize=(4,4), dpi=75, frameon=False)
        img = plt.imshow(A[0])
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.close()
        def animate(i):
            # title.set_text(str(i))
            img.set_data(A[i*step])
        anim = matplotlib.animation.FuncAnimation(fig, animate, frames=A.shape[0]//step, interval=20)
        anim.save(video_path, writer=matplotlib.animation.PillowWriter(fps=20))


    def make_grad_video(self, time_steps = 100, step=1, seed = None, polygon_size = 60, config = None, video_path="video.gif"):
        if config is not None:
            A = config
        elif seed:
            A = self.get_init_voronoi(self, polygon_size, sample=0, seed=seed)
        else:
            A = self.get_init_voronoi(self, polygon_size, sample=0, seed=None)

        A = A.view(self.w, self.h)
        
        
        M = np.zeros((time_steps, self.w, self.h), dtype=float)
        M[0, :, :] = A.cpu().numpy()

        for t in range(time_steps-1):
            t += 1
            A = self.grad_multi_step(A).view(self.w, self.h)
            M[t, :, :] =A.cpu().numpy()
    

        if time_steps//self.multi_step_size > 500:
            step = 10
        

        
        fig = plt.figure(figsize=(4,4), dpi=75, frameon=False)
        img = plt.imshow(M[0])
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.close()
        def animate(i):
            # title.set_text(str(i))
            img.set_data(M[i*step])
        anim = matplotlib.animation.FuncAnimation(fig, animate, frames=M.shape[0]//step, interval=20)
        anim.save(video_path, writer=matplotlib.animation.PillowWriter(fps=30))


    
    def get_mass_center(self, array):
        C, H, W = array.shape # array shape: (C,H,W)
        print(array.shape)
        A = torch.permute(array, (1, 2, 0)).view(H*W, 1, C)   # A shape: (H*W, 1, C)

        total_mass = torch.sum(A, axis = 0)[-1]   # (C)
        print("total mass")
        print(total_mass.shape)

        prod = A*self.coords
        print("prod: ", prod.shape)  # (H*W, 2, 1)
        sum_mass = torch.sum(prod, axis=0) # (2, 1)
        print("sum mass:", sum_mass, sum_mass.shape)

        
        mask = (total_mass != 0)
        sum_mass[:, mask] = sum_mass[:, mask] / total_mass[mask]

        return sum_mass, total_mass
    
    def get_batch_mass_center(self, array):
        B, C, H, W = array.shape # array shape: (B,C,H,W)
        A = torch.permute(array, (2, 3, 0, 1)).view(H*W, 1, B*C)   # A shape: (H*W, 1, B*C)

        total_mass = torch.sum(A, axis = 0)[-1]   # (B*C)

        prod = A*self.coords
        sum_mass = torch.sum(prod, axis=0) # (2, B*C)
        
        mask = (total_mass != 0) 
        sum_mass[:, mask] = sum_mass[:, mask] / total_mass[mask]

        return sum_mass, total_mass
        


class DiscreteLenia(DevModule):
    """
        Discrete Batched Single-channel Lenia, to run batch_size worlds in parallel !
    """
    def __init__(self, size, discretization:int, params=None, state_init = None, device='cpu' ):
        """
            Initializes automaton.  

            Args :
                size : (B,H,W) of ints, size of the automaton and number of batches
                discretization : int, number of discrete values for the automaton
                params : dict of tensors containing the parameters. If none, generates randomly
                    keys-values : 
                    'k_size' : odd int, size of kernel used for computations
                    'mu' : (B,) tensor, mean of growth functions
                    'sigma' : (B,) tensor, standard deviation of the growth functions
                    'mu_k' : (B,) [0,1.], location of the kernel rings (NOT USED FOR NOW)
                    'sigma_k' : (B,) float, standard deviation of the kernel rings (NOT USED FOR NOW)
                    'device' : str, device 
        """
        super().__init__()
        self.to(device)

        self.batch= size[0]
        self.h, self.w  = size[1:]
        # 0,1,2,3 of the first dimension are the N,W,S,E directions
        if(params is None):
            params = self.gen_batch_params(device)
    
        self.k_size = params['k_size'] # kernel sizes (same for all)
        self.discri = discretization

        self.register_buffer('state',torch.randint(low=0,high=discretization+1,size=(self.batch,1,self.h,self.w)))

        if(state_init is None):
            self.set_init_fractal()
        else:
            self.state = state_init.to(self.device,dtype=torch.int)

        # Buffer for all parameters since we do not require_grad for them :
        self.register_buffer('mu', params['mu']) # mean of the growth functions (3,3)
        self.register_buffer('sigma', params['sigma']) # standard deviation of the growths functions (3,3)
        self.register_buffer('mu_k',params['mu_k'])# mean of the kernel gaussians (3,3, # of rings)
        self.register_buffer('sigma_k',params['sigma_k'])# standard deviation of the kernel gaussians (3,3, # of rings)

        self.register_buffer('kernel',torch.zeros((self.k_size,self.k_size)))
        self.update_params(params)

    def gen_batch_params(self,device):
        """ Generates batch parameters."""
        mu = 0.15*(torch.ones((self.batch,), device=device))
        sigma = 0.015*(torch.ones_like(mu))
            

        params = {
            'k_size' : 27, 
            'mu':  mu ,
            'sigma' : sigma,
            'mu_k' : torch.full((self.batch,), fill_value=0.5, device=device),
            'sigma_k' : torch.full((self.batch,),fill_value=0.15, device=device),
        }
        
        return params

    def update_params(self, params):
        """
            Updates some or all parameters of the automaton. Changes batch size to match one of provided params (take mu as reference)
        """
        self.mu = params.get('mu',self.mu) # mean of the growth functions (C,C)
        self.sigma = params.get('sigma',self.sigma) # standard deviation of the growths functions (C,C)
        self.mu_k = params.get('mu_k',self.mu_k)
        self.sigma_k = params.get('sigma_k',self.sigma_k)
        self.k_size = params.get('k_size',self.k_size) # kernel sizes (same for all)


        self.batch = self.mu.shape[0] # update batch size
        self.update_kern_growth()
    
    def update_kern_growth(self):
        self.kernel, max_kern_activ = self.compute_kernel() # (B,1,1,h, w)
        self.growths = self.compute_growth(max_kern_activ)[:,:,None,None,None,None] # (B,2,1,1,1,1) of min and max growths (comparable direclty with state)

    def compute_growth(self,max_kernel_activation):
        """
            Create growth range given parameters
        """

        growth_x_axes =  [torch.linspace(0, 1, max_kern.item(),device=self.device) for max_kern in max_kernel_activation] # mapping from activation to [0,1], shape (max_kern,)*B
        g_tensor = [] # (B,2) tensor of min and max growth values
        for i,growth_axis in enumerate(growth_x_axes):
            growth_lookup = self.gaussian_func(growth_axis, m=self.mu[i], s=self.sigma[i]) # growth value per activation (max_kern,)

            growth_lookup = self.discretize(growth_lookup, div=1, mult=True) # discretize the growth values to [0, discri]

            growth_support = growth_lookup.nonzero()[:,0] # true if growth is nonzero

            if growth_support.any():
                arg_min, arg_max = torch.min(growth_support).item(), torch.max(growth_support).item()
                g = torch.tensor([max(0,arg_min-1), min(arg_max+1,growth_lookup.shape[0]-1)], dtype=torch.int, device=self.device)
            else:
                g = torch.tensor([0, 0], dtype=torch.int, device=self.device)

            g_tensor.append(g)
        # print('DA G TENSOR : ', torch.stack(g_tensor,dim=0))
        
        return torch.stack(g_tensor,dim=0) # (B,2)
    
    def get_params(self):
        """
            Get the parameter dictionary which defines the automaton
        """
        params = dict(k_size = self.k_size,mu = self.mu, sigma = self.sigma, beta = self.beta,
                       mu_k = self.mu_k, sigma_k = self.sigma_k, weights = self.weights)
        
        return params

    def set_init_fractal(self):
        """
            Sets the initial state of the automaton using perlin noise
        """
        perlin = perlin_fractal((self.batch,self.h,self.w),int(self.k_size*1.5),
                                    device=self.device,black_prop=0.25,num_channels=1,persistence=0.4) 

        self.state = (perlin*self.discri).round().clamp(0,self.discri)


    def set_init_perlin(self,wavelength=None, square_size=None):
        if(not wavelength):
            wavelength = self.k_size
        perlino = perlin((self.batch,self.h,self.w),[wavelength]*2,
                            device=self.device,num_channels=1,black_prop=0.25)

        if(square_size):
            masku = torch.zeros_like(perlino)
            masku[:,:,self.h//2-square_size//2:self.h//2+square_size//2,self.w//2-square_size//2:self.w//2+square_size//2] = 1
            perlino = perlino*masku
        self.state = (perlino*self.discri).round().clamp(0,self.discri)
    
    @staticmethod
    def gaussian_func(x, m, s, h=1):
        def safe_divide(x, y, eps=1e-10):
            return x / (y + eps)
        return torch.exp(-safe_divide((x - m), s)**2 / 2) * h
    
    @staticmethod
    def discretize(tensor, div=1, mult=True):
        if mult:
            return torch.round(tensor * div).to(dtype=torch.int)
        else:
            return tensor.to(dtype=torch.int)

    def compute_kernel(self):
        """
            Get the kernel in the case k=1
        """
        # calculate distance from origin
        kernel_sizes = [self.k_size, self.k_size] # (k_y, k_x)
        kernel_radius = (self.k_size-1)//2

        kernel_mids = [size // 2 for size in kernel_sizes] # (mid_y, mid_x)

        ranges = [slice(0 - mid, size - mid) for size, mid in zip(kernel_sizes, kernel_mids)] # y range, x range

        space = np.asarray(np.mgrid[ranges], dtype=float)  # [2, k_x,k_y]. space[:,x,y] = [x,y]
        distance = np.linalg.norm(space, axis=0)  # [k_x,k_y]. distance[x,y] = sqrt(x^2 + y^2)

        # calculate kernel K
        distance_scaled = distance / kernel_radius  # [xyz]
        
        distance_scaled = torch.tensor(distance_scaled).to(self.device) # (k_x,k_y) tensor of r_distances
        distance_scaled = distance_scaled[None].expand(self.batch,-1,-1) # (B,k_x,k_y) tensor of r_distances

        kernel = self.gaussian_func(distance_scaled,m=self.mu_k[:,None,None], s=self.sigma_k[:,None,None])  # (B,k_x,k_y) tensor of kernel values
        kernel = self.discretize(kernel, self.discri, mult=True) # (B,k_x,k_y) tensor of discretized kernel values

        kernel_sum = torch.sum(kernel,dim=(-1,-2))  # [ B, ] 

        kernel_max_activation = self.discri * kernel_sum # [ B, ] max activations

        kernel = kernel.reshape(self.batch,1, 1, self.k_size, self.k_size)  # (B,1,1,k_x,k_y) tensor of kernel values

        # showTens(kernel.float())
        return kernel.float() , kernel_max_activation

    def step(self):
        """
            Steps the automaton state by one iteration.

            Args :
            discrete_g : 2-uple of floats, min and max values, when using 'discrete' growth.
            If None, will use the normal growth function.
        """
        # Shenanigans to make all the convolutions at once.
        kernel_eff = self.kernel.reshape([self.batch,1,self.k_size,self.k_size])#(B,1,k,k)

        U = self.state.reshape(1,self.batch,self.h,self.w) # (1,B,H,W)
        U = F.pad(U, [(self.k_size-1)//2]*4, mode = 'circular') # (1,B,H+pad,W+pad)
        
        U = F.conv2d(U, kernel_eff, groups=self.batch).squeeze(1) #(B*1^2,1,H,W) squeeze to (B*1,H,W)
        U = U.reshape(self.batch,1,1,self.h,self.w) # (B,1,1,H,W)

        assert (self.h,self.w) == (self.state.shape[2], self.state.shape[3])
        dx = ((U > self.growths[:,0]) & (U < self.growths[:,1])).to(dtype=torch.int)
        dx = (dx * 2 - 1).sum(dim=1) # -1 if not grown, 1 if grown (B,1,1,H,W) -> (B,1,H,W)

        self.state = torch.clamp(torch.round(self.state + dx), 0, self.discri)     

    def mass(self):
        """
            Computes average 'mass' of the automaton for each channel

            returns :
            mass : (B,C) tensor, mass of each channel
        """

        return (self.state).mean(dim=(-1,-2))/self.discri # (B,1) normalized mean mass for each color

    def draw(self):
        """
            Draws the worldmap from state.
            Separate from step so that we can freeze time,
            but still 'paint' the state and get feedback.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"
        toshow= self.state[0].permute((2,1,0)) #(W,H,C)

        toshow = toshow.expand(-1,-1,3).to(torch.float)
        toshow = toshow/self.discri # normalize to [0,1]

    
        self._worldmap= toshow.cpu().numpy()   
    
        
    @property
    def worldmap(self):
        return (255*self._worldmap).astype(dtype=np.uint8)

class Automaton(DevModule) :
    """
        USE DEPRECATED FOR NOW
        Class that internalizes the rules and evolution of 
        the cellular automaton at hand. It has a step function
        that makes one timestep of the evolution. By convention,
        and to keep in sync with pygame, the world tensor has shape
        (W,H,3). It contains float values between 0 and 1, which
        are (automatically) mapped to 0 255 when returning output, 
        and describes how the world is 'seen' by an observer.

        Parameters :
        size : 2-uple (W,H)
            Shape of the CA world
        device : str
    """

    def __init__(self,size, device='cpu'):
        super().__init__()
        self.w, self.h  = size
        self.size= size
        # This self._worldmap should be changed in the step function.
        # It should contains floats from 0 to 1 of RGB values.
        self._worldmap = np.random.uniform(( self.w,self.h,3))
        self.to(device)
        

    

    def step(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')
    
    @property
    def worldmap(self):
        return (255*self._worldmap).astype(dtype=np.uint8)



            

#============================== PARAMETERS ================================
device = 'cpu' # Device on which to run the automaton
W,H = 30,30 # Size of the automaton
array_size = W
dt = 0.1 # Time step size
num_channels= 1
#==========================================================================


params = {'k_size': 27, 
          'mu': torch.tensor([[[0.15]]], device=device), 
          'sigma': torch.tensor([[[0.015]]], device=device), 
          'beta': torch.tensor([[[[1]]]], device=device), 
          'mu_k': torch.tensor([[[[0.5]]]], device=device), 
          'sigma_k': torch.tensor([[[[0.15]]]], device=device), 
          'weights': torch.tensor([[[1]]], device=device)}





#================================ EXAMPLE =================================


