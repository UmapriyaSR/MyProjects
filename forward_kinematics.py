import numpy as np
from math import pi, cos, sin #appropriate library for math functions
import modern_robotics as mr
def forward_kinematics(joints):
    # input: joint angles [joint1, joint2, joint3
    
    # Given dimensions of the link and declaration of the output variables
    link1z = 0.065
    link2z = 0.039
    link3x = 0.050
    link3z = 0.150
    link4x = 0.150
    joint1 = joints[0]
    joint2 = joints[1]
    joint3 = joints[2]
    
    #Transformation matrix based on the rotation matrix and position vector from Joint 0 to 1 :
    T01=np.matrix([[cos(joint1),-sin(joint1),0,0],[sin(joint1),cos(joint1),0,0],[0,0,1,link1z],[0,0,0,1]])

    #Transformation matrix based on the rotation matrix and position vector from Joint 1 to 2 :
    T12=np.matrix([[cos(joint2),0,sin(joint2),0],[0,1,0,0],[-sin(joint2),0,cos(joint2),link2z],[0,0,0,1]])
    
    #Transformation matrix based on the rotation matrix and position vector from Joint 2 to 3 :
    T23=np.matrix([[1,0,0,link3x],[0,-1,0,0],[0,0,-1,link3z],[0,0,0,1]])
  
    #Transformation matrix based on the rotation matrix and position vector from Joint 3 to 4 :
    T34=np.matrix([[1,0,0,link4x*cos(joint3)],[0,1,0,0],[0,0,1,-link4x*sin(joint3)],[0,0,0,1]])

    #Final Transformation matrix using product of the above:
    T04=T01*T12*T23*T34
    print(T04)

    #Extraction of x,y and z values from the final matrix
    x=T04[0,3]
    y=T04[1,3]
    z=T04[2,3]
    # output: the position of end effector [x, y, z]
    return [x, y, z]


#if __name__ == '__main__':
    #forward_kinematics()
