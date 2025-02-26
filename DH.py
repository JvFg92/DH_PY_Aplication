import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Mechanism:
    def __init__(self, param):
        self.param = param
        self.n_joints = len(param)
        print("Joint Numbers:", self.n_joints)

        #Defining symbols for DH:
        self.theta = []
        self.d = []
        self.a = [sp.Symbol(f'a_{i}') for i in range(self.n_joints)]
        self.alpha = [sp.Symbol(f'alpha_{i}') for i in range(self.n_joints)]
        self.phi = []
        self.epsilon = []
        self.sigma = [sp.Symbol(f'sigma_{i}') for i in range(self.n_joints)]
        self.beta = [sp.Symbol(f'beta_{i}') for i in range(self.n_joints)]

        for i, params in enumerate(param):
            if params['type'] in ['revolute', 'cylindrical']:
                self.theta.append(sp.Symbol(f'theta_{i}'))
                self.phi.append(sp.Symbol(f'phi_{i}'))
            elif params['type'] == 'prismatic':
                self.theta.append(0)  #theta normally is fixed (0) for prismatic joints
                self.phi.append(sp.Symbol(f'phi_{i}'))
            self.d.append(sp.Symbol(f'd_{i}'))
            self.epsilon.append(sp.Symbol(f'epsilon_{i}'))

    def dh_matrix(self,i,apply_errors=False):
      """Return the symbolic homogeneous matrix."""
      params = self.param[i]
      #Nominal or error values
      a = self.a[i] + (self.sigma[i] if apply_errors else 0)
      alpha = self.alpha[i] + (self.beta[i] if apply_errors else 0)
      d = self.d[i] + (self.epsilon[i] if apply_errors else 0)
      
      if params['type'] in ['revolute', 'cylindrical']:
          theta = self.theta[i] + (self.phi[i] if apply_errors else 0)
      else:  #prismatic
          theta = params.get('theta_offset', 0) + (self.phi[i] if apply_errors else 0)

      cos_theta = sp.cos(theta)
      sin_theta = sp.sin(theta)
      cos_alpha = sp.cos(alpha)
      sin_alpha = sp.sin(alpha)
      
      return sp.Matrix([
          [cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, a * cos_theta],
          [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
          [         0,             sin_alpha,              cos_alpha,             d],
          [         0,                      0,                      0,            1]
      ])
    
    def forward_kinematics(self, apply_errors=False):
        T = sp.eye(4)  #Matrix 4x4 identity
        for i in range(self.n_joints):
            Ti = self.dh_matrix(i, apply_errors)
            T = T * Ti
        position = T[:3, 3]
        return T, position
    
    def evaluate_param(self, T, variable_values=None, apply_errors=False):
        """Evaluate the position of the effector using params from self.param and optional variable values."""
        #Build a dictionary with the parameters
        subs_dict = {}
        for i, params in enumerate(self.param):
            subs_dict[self.a[i]] = params['a']
            subs_dict[self.alpha[i]] = params['alpha']
            subs_dict[self.d[i]] = params['d']
            #For prismatic joints, theta is fixed
            if params['type'] == 'prismatic':
                subs_dict[self.theta[i]] = params.get('theta_offset', 0)
            #Add errors if apply_errors is True
            if apply_errors:
                subs_dict[self.phi[i]] = params['errors']['phi']
                subs_dict[self.epsilon[i]] = params['errors']['epsilon']
                subs_dict[self.sigma[i]] = params['errors']['sigma']
                subs_dict[self.beta[i]] = params['errors']['beta']
            else:
                subs_dict[self.phi[i]] = 0
                subs_dict[self.epsilon[i]] = 0
                subs_dict[self.sigma[i]] = 0
                subs_dict[self.beta[i]] = 0
        
        #Add variable values
        if variable_values:
            subs_dict.update(variable_values)

        T_numerica = T.subs(subs_dict)
        return T_numerica, T_numerica[:3, 3]
    
    def get_joint_positions(self, variable_values, apply_errors=False):
        """Return the joint positions for the mechanism in 3D."""
        positions = [[0, 0, 0]]  # Origem
        T = sp.eye(4)
        for i in range(self.n_joints):
            T = T * self.dh_matrix(i, apply_errors)
            T_eval, pos_eval = self.evaluate_param(T, variable_values, apply_errors)
            pos_num = [float(pos_eval[0]), float(pos_eval[1]), float(pos_eval[2])]
            positions.append(pos_num)
        return np.array(positions)
    
    def plot_mechanism(self, variable_values=None):
        """Plot the mechanism in 3D."""
        #Evaluate the joint positions
        joints_no_error = self.get_joint_positions(variable_values, apply_errors=False)
        joints_with_error = self.get_joint_positions(variable_values, apply_errors=True)

        #Create the figure
        fig = plt.figure(figsize=(12, 5))
        
        #Mechanism plot
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(joints_no_error[:, 0], joints_no_error[:, 1], joints_no_error[:, 2], 'b-o', label='Sem erros')
        ax.plot(joints_with_error[:, 0], joints_with_error[:, 1], joints_with_error[:, 2], 'r--o', label='Com erros')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Mechanism kinematics with and without errors')
        ax.legend()
        plt.show()