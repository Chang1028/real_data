import numpy as np
import jax.numpy as jnp
from jax import random
from jax import grad
import math
import matplotlib.pyplot as plt
import time
from poprogress import simple_progress


class bilinear_lasso:
    def __init__(self, input1, input2, y, lam1, lam2):
        self.X1 = input1
        self.X2 = input2
        self.y = y
        self.lam1 = lam1
        self.lam2 = lam2

        self.p = input1.shape[0]
        self.n = input1.shape[2]
        self.beta1 = None
        self.beta2 = None
        self.l = None

        self.trajectory = {}
        self.loss_history = {}
        self.sel_idx = None

    def initialize_beta(self, beta_1, beta_2):
        self.beta1 = beta_1
        self.beta2 = beta_2

    def predict(self, beta_1, beta_2):
        output = jnp.einsum("ij,ijk,ji->k", beta_1, self.X1, beta_1) + jnp.einsum("ij,ijk,ji->k", beta_2, self.X2, beta_2)  
        return output

    def l1_loss(self, beta_1, beta_2):
        return jnp.sum(jnp.abs(beta_1)) * self.lam1 + jnp.sum(jnp.abs(beta_2)) * self.lam2
    
    def loss(self, beta_1, beta_2):
        preds = self.predict(beta_1, beta_2)  
        return jnp.mean((preds - self.y)**2)

    def loss_with_penalty(self, beta_1, beta_2):
        return self.loss(beta_1, beta_2) + self.l1_loss(beta_1, beta_2)
    
    def prox(self, x, a, L):
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - a / L, 0.0)
    

    def fit(self, num_candidates = 1, max_iter = 8000, step_size = 0.1, tol = 1e-5, disturbance = 0.1):
        
        res = {}
        
        for rs in range(num_candidates):

            res[rs] = {}
            self.loss_history[rs] = []
            self.trajectory[rs] = {}
            self.trajectory[rs]["beta1"] = []
            self.trajectory[rs]["beta2"] = []

            beta_1 = self.beta1 + disturbance * random.normal(random.PRNGKey(np.random.randint(0, 2025)), (self.p, 1))
            beta_2 = self.beta2 + disturbance * random.normal(random.PRNGKey(np.random.randint(0, 2025)), (self.p, 1))

            b1 = beta_1.copy()
            b2 = beta_2.copy()

            t = 1.0

            l0 = self.loss(beta_1, beta_2)
            print("="*30)
            print(f"{rs} th try")
            print(f"initialized loss: {l0}")
            flag = 0

            starttime = time.time()

            self.trajectory[rs]['beta1'].append(np.array(beta_1))
            self.trajectory[rs]['beta2'].append(np.array(beta_2))


            for idx in simple_progress(range(max_iter)):
                # Differentiate `loss` with respect to the specific positional argument:

                grads_1, grads_2 = grad(self.loss, (0, 1))(beta_1, beta_2)

                beta_1_new = self.prox(b1 - step_size * grads_1, self.lam1, 1/step_size)
                beta_2_new = self.prox(b2 - step_size * grads_2, self.lam2, 1/step_size)

                # FISTA momentum update
                t_new = (1 + jnp.sqrt(1 + 4 * t**2)) / 2
                momentum = (t - 1) / t_new

                b1 = beta_1_new + momentum * (beta_1_new - beta_1)
                b2 = beta_2_new + momentum * (beta_2_new - beta_2)

                if max(jnp.linalg.norm(beta_1 - beta_1_new), jnp.linalg.norm(beta_2 - beta_2_new)) < tol:
                    flag = True

                beta_1 = beta_1_new.copy()
                beta_2 = beta_2_new.copy()
                t = t_new

                # if idx % int(max_iter / 5) == 0 or flag:
                #     l = self.loss_with_penalty(beta_1, beta_2)
                #     print(f"iteration {idx}: {l}")
                #     if flag:
                #         break
                
                if flag:
                    break
                
                if idx % int(max_iter / 100) == 0:
                    self.trajectory[rs]['beta1'].append(np.array(beta_1))
                    self.trajectory[rs]['beta2'].append(np.array(beta_2))
                
                self.loss_history[rs].append(float(self.loss_with_penalty(beta_1, beta_2)))

            self.trajectory[rs]['beta1'] = jnp.stack(self.trajectory[rs]['beta1'])
            self.trajectory[rs]['beta2'] = jnp.stack(self.trajectory[rs]['beta2'])

            endtime = time.time()
            print('Job took: ', endtime-starttime)

            res[rs]["loss"] = self.loss_with_penalty(beta_1, beta_2)
            res[rs]["beta1"] = beta_1.copy()
            res[rs]["beta2"] = beta_2.copy()
            print(f"Final loss: {res[rs]["loss"]}")
        
        loss_list = []
        for key in res.keys():
            loss_list.append(res[key]["loss"])
        sel_idx = np.array(loss_list).argmin()
        self.beta1 = res[sel_idx]["beta1"]
        self.beta2 = res[sel_idx]["beta2"]
        self.l = self.loss_with_penalty(self.beta1, self.beta2)
        print(f"Final loss: {res[sel_idx]['loss']}")

        # Trajectories for selected candidate
        self.sel_idx = sel_idx
        
