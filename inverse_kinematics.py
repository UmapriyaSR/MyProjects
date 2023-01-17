import numpy as np
from math import pi, cos, sin, atan, atan2, sqrt, acos

def inverse_kinematics(position):
    # input: the position of end effector [x, y, z]

#Given dimension of the links and x,y,z position
	link1z = 0.065
	link2z = 0.039
	link3x = 0.050
	link3z = 0.150
	link4x = 0.150
	x = position[0]
	y = position[1]
	z = position[2]

#link from joint 2 to 3 by pythagoras theorem
	j2_3=sqrt(link3x**2 + link3z**2)
#link from joint 2 to 4 by pythagoras theorem
	j2_4=sqrt(x**2+y**2)

#angle between link 3z and hypotenuse connecting joint 2 to joint 3
	alpha = atan2(link3x,link3z)

#length of remaining portion	
	B = z-link1z-link2z
#angle between horizontal reference from joint 2 and line connecting joint 2 to 4	
	gamma=atan2(B,j2_4)
	a = sqrt(j2_4**2 + B**2)

#From figure , computed using cosine formulas & tan inverse formulas	
	beta1 = acos((a**2+j2_3**2 - link4x**2)/(2*a*j2_3))
	theta2=pi/2-gamma-alpha-beta1	
	beta2=acos((link4x**2+j2_3**2-a**2)/(2*link4x*j2_3))
	theta3=beta2-alpha-pi/2
	theta1=atan2(y,x)

    # output: joint angles [joint1, joint2, joint3]
	return [theta1, theta2, theta3]


#if __name__== '__main__':
#	inverse_kinematics()
