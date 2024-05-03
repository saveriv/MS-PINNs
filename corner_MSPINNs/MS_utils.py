import scipy.io
import deepxde as dde
from deepxde.backend import tf
import numpy as np
import os, sys, subprocess, re, struct, errno
import torch

class system_dynamics():
    
    def __init__(self):
        
        ## PDE Parameters
        self.V_gate = 0.13
        self.a_crit = 0.0  # comment in openCARP: 0.13
        self.tau_in = 0.001 # 0.3 in openCARP #0.05 in Belhamadia et al
        self.tau_out = 0.1 # 5.0 in opencarp #1 in Belhamadia et al
        self.tau_open = 95 # 120.0 in openCARP #95 in Belhamadia et al
        self.tau_close = 162 # 150 in openCARP # 162 in Belhamadia et al
        self.D = 0.001

        ## Geometry Parameters
        self.min_x = 0
        self.max_x = 10            
        self.min_y = 0 
        self.max_y = 10
        self.min_t = 1
        self.max_t = 70
        self.spacing = 0.1 #spacing size = distance between two consecutive points

    # Read igb file
    def read_array_igb(self, igbfile):
        data = []
        file = open(igbfile, mode="rb")
        header = file.read(1024)
        words = header.split()
        word = []
        for i in range(4):
            word.append(int([re.split(r"(\d+)", s.decode("utf-8")) for s in [words[i]]][0][1]))

        nnode = word[0] * word[1] * word[2]

        for _ in range(os.path.getsize(igbfile) // 4 // nnode):
            data.append(struct.unpack("f" * nnode, file.read(4 * nnode)))

        file.close()
        return data

    # Read pts/vertex file
    def read_pts(self, modname, n=3, vtx=False, item_type=float):  
        with open(modname + (".vtx" if vtx else ".pts")) as file:
            count = int(file.readline().split()[0])
            if vtx:
                file.readline()

            pts = np.empty((count, n), item_type)
            for i in range(count):
                pts[i] = [item_type(val) for val in file.readline().split()[0:n]]

        return pts if n > 1 else pts.flat

    def generate_data(self, v_file_name, h_file_name, pt_file_name):
        
        data_V = np.array(self.read_array_igb(v_file_name))  # parser for Vm.igb
        data_h = np.array(self.read_array_igb(h_file_name))  # parser for h.igb
        coordinates = np.array(self.read_pts(pt_file_name))  # new parser for .pt file

        t = np.arange(0, data_V.shape[0]/100, 0.01).reshape(-1, 1)

        coordinates = (coordinates - np.min(coordinates))/1000
        coordinates = coordinates[:, 0:2]
        x = np.unique(coordinates[:, 0]).reshape((1, -1))
        y = np.unique(coordinates[:, 1]).reshape((1, -1))
        len_x = x.shape[1]
        len_y = y.shape[1]
        len_t = t.shape[0]

        no_of_nodes = coordinates.shape[0]
        repeated_array = np.repeat(coordinates, len_t, axis=0)
        xy_concatenate = np.vstack(repeated_array)
        t_concatenate = np.concatenate([t] * no_of_nodes, axis=0)
        grid = np.concatenate([xy_concatenate, t_concatenate], axis=1)

        data_V = data_V.T
        data_h = data_h.T

        shape = [len_x, len_y, len_t]
        V = data_V.reshape(-1, 1)
        h = data_h.reshape(-1, 1)

        shape = [len_x, len_y, len_t]
        Vsav = V.reshape(len_x, len_y, len_t)

        return grid, V, h, Vsav, len_t

    def geometry_time(self):
            
            geom = dde.geometry.Rectangle([self.min_x,self.min_y], [self.max_x,self.max_y])
            timedomain = dde.geometry.TimeDomain(self.min_t, self.max_t)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)
            return geomtime
    
    def params_to_inverse(self,args_param):
        
        params = []
        if not args_param:
           return self.V_gate, self.a_crit, params
        # If inverse:
        # The tf.variables are initialized with a positive scalar, relatively close to their ground truth values
        if 'V_gate' in args_param:
            self.V_gate = tf.math.exp(tf.Variable(0.13))
            params.append(self.V_gate)
        if 'a_crit' in args_param:
            self.a_crit = tf.math.exp(tf.Variable(0))
            params.append(self.a_crit)
        return params #correct values or not?

    def state_equations(self, x, y):

        V, h = y[:, 0:1], y[:, 1:2]

        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)

        # Define the conditions based on V and V_gate
        condition = V[:, 0] < self.V_gate

        # Extract V and h based on the condition
        V_p_ext, h_p_ext = V[condition], h[condition]
        V_q_ext, h_q_ext = V[~condition], h[~condition]
        Vp = V_p_ext.reshape(-1, 1)
        hp = h_p_ext.reshape(-1, 1)
        Vq = V_q_ext.reshape(-1, 1)
        hq = h_q_ext.reshape(-1, 1)

        # compute derivatives
        Vp, hp = y[:, 0:1], y[:, 1:2]
        Vq, hq = y[:, 0:1], y[:, 1:2]
        dVp_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dhp_dt = dde.grad.jacobian(y, x, i=1, j=2)
        dVq_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dhq_dt = dde.grad.jacobian(y, x, i=1, j=2)

        # if V < V_gate
        eq_a_p = dVp_dt - (hp * Vp * (Vp  - self.a_crit) * (1 - Vp) / self.tau_in + (-Vp / self.tau_out)) - self.D*(dv_dxx + dv_dyy)
        eq_b_p = dhp_dt - (1. - hp) / self.tau_open

        # if V > V_gate
        eq_a_q = dVq_dt - (hq * Vq * (Vq  - self.a_crit) * (1 - Vq) / self.tau_in + (-Vq / self.tau_out)) - self.D*(dv_dxx + dv_dyy)
        eq_b_q = dhq_dt + hq / self.tau_close

        # Combine the equations based on the conditions
        eq_a = torch.cat((eq_a_p, eq_a_q))
        eq_b = torch.cat((eq_b_p, eq_b_q))
        
        return [eq_a, eq_b]

    # def pde_2D_heter(self, x, y): 
    
    #     V, W, var = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    #     dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
    #     dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    #     dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    #     dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
    #     dv_dx = dde.grad.jacobian(y, x, i=0, j=0)
    #     dv_dy = dde.grad.jacobian(y, x, i=0, j=1)
        
    #     ## Heterogeneity
    #     D_heter = tf.math.sigmoid(var)*0.08+0.02;
    #     dD_dx = dde.grad.jacobian(D_heter, x, i=0, j=0)
    #     dD_dy = dde.grad.jacobian(D_heter, x, i=0, j=1)
        
    #     ## Coupled PDE+ODE Equations
    #     eq_a = dv_dt -  D_heter*(dv_dxx + dv_dyy) -dD_dx*dv_dx -dD_dy*dv_dy + self.k*V*(V-self.a)*(V-1) +W*V 
    #     eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
    #     return [eq_a, eq_b]
 
    # def pde_2D_heter_forward(self, x, y): #for solving eqns numerically
                
    #     V, W, D = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    #     dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
    #     dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    #     dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    #     dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
    #     dv_dx = dde.grad.jacobian(y, x, i=0, j=0)
    #     dv_dy = dde.grad.jacobian(y, x, i=0, j=1)
        
    #     ## Heterogeneity
    #     dD_dx = dde.grad.jacobian(D, x, i=0, j=0)
    #     dD_dy = dde.grad.jacobian(D, x, i=0, j=1)
        
    #     ## Coupled PDE+ODE Equations
    #     eq_a = dv_dt -  D*(dv_dxx + dv_dyy) -dD_dx*dv_dx -dD_dy*dv_dy + self.k*V*(V-self.a)*(V-1) +W*V 
    #     eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
    #     return [eq_a, eq_b]   
 
    def IC_func(self,observe_train, v_train):
        
        T_ic = observe_train[:,-1].reshape(-1,1)
        idx_init = np.where(np.isclose(T_ic,1))[0]
        v_init = v_train[idx_init]
        observe_init = observe_train[idx_init]
        return dde.PointSetBC(observe_init,v_init,component=0)
    
    def BC_func(self, geomtime):
        bc = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), self.boundary_func_2d, component=0)
        return bc
    
    def boundary_func_2d(self,x, on_boundary):
            return on_boundary and ~(x[0:2]==[self.min_x,self.min_y]).all() and  ~(x[0:2]==[self.min_x,self.max_y]).all() and ~(x[0:2]==[self.max_x,self.min_y]).all()  and  ~(x[0:2]==[self.max_x,self.max_y]).all() 
   
    def modify_inv_heter(self, x, y):                
        domain_space = x[:,0:2]
        D = tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(domain_space, 60,
                            tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 1, activation=None)        
        return tf.concat((y[:,0:2],D), axis=1)    
    
    def modify_heter(self, x, y):
        
        x_space, y_space = x[:, 0:1], x[:, 1:2]
        
        x_upper = tf.less_equal(x_space, 54*0.1)
        x_lower = tf.greater(x_space,32*0.1)
        cond_1 = tf.logical_and(x_upper, x_lower)
        
        y_upper = tf.less_equal(y_space, 54*0.1)
        y_lower = tf.greater(y_space,32*0.1)
        cond_2 = tf.logical_and(y_upper, y_lower)
        
        D0 = tf.ones_like(x_space)*0.02 
        D1 = tf.ones_like(x_space)*0.1
        D = tf.where(tf.logical_and(cond_1, cond_2),D0,D1)
        return tf.concat((y[:,0:2],D), axis=1)
