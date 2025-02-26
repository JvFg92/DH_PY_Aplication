import numpy as np
import DH 

if __name__ == "__main__":
    
    #Define the DH parameters:(exempli gratia)
    eg = [
        {'type': 'revolute', 'a': 1.0, 'alpha': 0, 'd': 0.0, 
         'errors': {'phi': 0.05, 'epsilon': 0.02, 'sigma': 0.01, 'beta': 0.03}},
        {'type': 'revolute', 'a': 1.0, 'alpha': np.pi/2, 'd': 0.0, 
         'errors': {'phi': 0.03, 'epsilon': 0.01, 'sigma': 0.02, 'beta': 0.02}},
        {'type': 'prismatic', 'a': 0.0, 'alpha': 0, 'd': 0.5, 
         'errors': {'phi': 0.0, 'epsilon': 0.05, 'sigma': 0.0, 'beta': 0.0}}
    ]

    #Create the robot:
    robot = DH.Mechanism(eg)

    #Calculate the forward kinematics without errors:
    matrix,position=robot.forward_kinematics(False)
    print("\nMatrix and position algebrical without errors:\n")
    print("Matrix:\n", matrix)
    print("\nPosition:\n", position)

    #Calculate the forward kinematics with errors:
    matrix_e,position_e=robot.forward_kinematics(True)
    print("\nMatrix and position algebrical with errors:\n")
    print("Matrix:\n", matrix_e)
    print("\nPosition:\n", position_e)

    #Evaluate the position of the effector using optional variable values:
    variable_values = {
    robot.theta[0]: 0.5,    #theta_0 for the first revolute joint
    robot.theta[1]: 1.0,    #theta_1 for the second cylindrical joint   
    robot.d[2]:     1.0     #d_2 for the third prismatic joint
    }

    #Calculate the forward kinematics without errors:
    matrix_numerical, position_numerical = robot.evaluate_param(matrix, variable_values)
    print("\nMatrix and position numerical with errors:\n")
    print("Matrix:\n", matrix_numerical)
    print("\nPosition:\n", position_numerical)

    #Calculate the forward kinematics with errors:
    matrix_numerical_e, position_numerical_e = robot.evaluate_param(matrix_e, variable_values, True)
    print("\nMatrix and position numerical with errors:\n")
    print("Matrix:\n", matrix_numerical_e)
    print("\nPosition:\n", position_numerical_e)

    #Calculate error values:
    print("\n [X, Y, Z] position values:")
    print("Position without erros: \n", [p for p in position_numerical])
    print("Position with errors: \n", [p for p in position_numerical_e])
    error = [0, 0, 0]
    for i in range(3):
        error[i] = position_numerical_e[i] - position_numerical[i]
    print("\n [X, Y, Z] error values:")
    print("\n", error)
        
    #Plot the robot:
    robot.plot_mechanism(variable_values)