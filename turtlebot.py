#!/usr/bin/env python

from math import pi, sqrt, atan2, cos, sin
import numpy as np

import rospy
import tf
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D
from motion_planning import get_path_from_A_star

class Turtlebot():
	def __init__(self):
		rospy.init_node("turtlebot_move")
		rospy.loginfo("Press Ctrl + C to terminate")
		self.vel_pub = rospy.Publisher("cmd_vel_mux/input/navi", Twist, queue_size=10) #publish velocity multiplexer
		self.rate = rospy.Rate(10) #define frequency

        # reset odometry to zero
		self.reset_pub = rospy.Publisher("mobile_base/commands/reset_odometry", Empty, queue_size=10)
		for i in range(10):
			self.reset_pub.publish(Empty())
			self.rate.sleep()

		# initializing conditions
		self.previous_point=[0,0] #initial position
		self.previous_velocity=[0,0] #initial velocity
		self.vel_steering = 0.1
		self.vel=Twist()

        # subscribe to odometry
		self.pose = Pose2D()
		self.logging_counter = 0
		self.trajectory = list()
		self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_callback)

		try:
			self.run()
		except rospy.ROSInterruptException:
			rospy.loginfo("Action terminated.")
		finally:
			# save trajectory into csv file
			np.savetxt('trajectory.csv', np.array(self.trajectory), fmt='%f', delimiter=',')

	
	def run(self):
		start = (0, 0)  # this is a tuple data structure in Python initialized with 2 integers
		goal = (4.5,0)
		obstacles = [(0.5, 0.5), (0.5,1), (1.5, -0.5), (1.5,0), (3,0.5), (3,1)]
		path = get_path_from_A_star(start, goal, obstacles)
		print(path)
		path.append(goal)
		waypoints = path
		#waypoints = [[0.5, 0], [0.5, -0.5], [1, -0.5], [1, 0], [1, 0.5],\
		#				[1.5, 0.5], [1.5, 0], [1.5, -0.5], [1, -0.5], [1, 0],\
		#			[1, 0.5], [0.5, 0.5], [0.5, 0], [0, 0], [0, 0]]
		
		for i in range(len(waypoints)-1):
			self.move_to_point(waypoints[i], waypoints[i+1])


	def polynomial_time_scaling_3rd_order(self, p_start, v_start, p_end, v_end, T):
		#input position and velocity of start and end points
		#output coefficients of the polynomial
		X=np.array([[0,0,0,1],[T**3, T**2, T, 1],[0,0,1,0],[3*T**2, 2*T,1,0]])
		B = np.array([[p_start],[p_end],[v_start],[v_end]])
		coefficient=np.dot(np.linalg.inv(X),B)   #compute the coefficients using matrix multiplication and inverse
		return coefficient


	def move_to_point(self, current_waypoint, next_waypoint):
        # generate polynomial trajectory and move to current_waypoint
        # next_waypoint is to help determine the velocity to pass current_waypoint

		#position boundary conditions
		x_start = self.previous_point[0]
		x_end = current_waypoint[0]
		y_start = self.previous_point[1]
		y_end = current_waypoint[1]

		#velocity boundary conditions
		xdot = self.previous_velocity[0]
		ydot = self.previous_velocity[1]

		#Calculating angle
		dx = next_waypoint[0]-current_waypoint[0]
		dy = next_waypoint[1]-current_waypoint[1]
		theta = atan2(dy,dx)

		#Decomposing velocity
		vx_end = self.vel_steering*cos(theta)
		vy_end = self.vel_steering*sin(theta)

		T=1

		coeff_x = self.polynomial_time_scaling_3rd_order(x_start,xdot,x_end,vx_end,T)  
		coeff_y = self.polynomial_time_scaling_3rd_order(y_start,ydot,y_end,vy_end,T)
		c=10
		for i in range(c*T):
			t=i*0.1
			V = np.array([3*t**2, 2*t,1,0])
			Vx=np.dot(V,coeff_x)
			Vy=np.dot(V,coeff_y)
			#Linear velocity
			P=sqrt(Vx**2+Vy**2)
			#calculating angle
			theta1=atan2(Vy,Vx)
			#calculating error
			error=theta1-self.pose.theta
			if error<-pi:
				error = error + 2*pi
			elif error>pi:
				error = error - 2*pi
			else:
				error = error
			Kp=5
			Q=Kp*error #Angular velocity

			#Publish
			self.vel.linear.x = P
			self.vel.angular.z = Q
			self.vel_pub.publish(self.vel)
			self.rate.sleep()

			#Update previous position and velocity values
			self.previous_point = current_waypoint
			self.previous_velocity = [vx_end, vy_end]


	def odom_callback(self, msg):
		# get pose = (x, y, theta) from odometry topic
		quarternion = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,\
				msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
		(roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quarternion)
		self.pose.theta = yaw
		self.pose.x = msg.pose.pose.position.x
		self.pose.y = msg.pose.pose.position.y

		# logging once every 100 times (Gazebo runs at 1000Hz; we save it at 10Hz)
		self.logging_counter += 1
		if self.logging_counter == 100:
			self.logging_counter = 0
			self.trajectory.append([self.pose.x, self.pose.y])  # save trajectory
			rospy.loginfo("odom: x=" + str(self.pose.x) +\
				";  y=" + str(self.pose.y) + ";  theta=" + str(yaw))

def neighbors(current):
    # define the list of 4 neighbors
    # For any point, 4 neighbors are the point to the above, below, left and right
    # Therefore, they are +/- on the x and y-axis
    neighbors = [[0, 0.5], [0, -0.5], [0.5, 0], [-0.5, 0]]
    return [(current[0] + nbr[0], current[1] + nbr[1]) for nbr in neighbors]

#Distance between the candidate and goal
def heuristic_distance(candidate, goal):
    dist_x, dist_y = [abs(goal[i] - candidate[i]) for i in range(len(goal))]
    dist = (dist_x**2 + dist_y**2) ** 0.5
    return dist


def get_path_from_A_star(start, goal, obstacles):
    # input  start: integer 2-tuple of the current grid, e.g., (0, 0)
    #        goal: integer 2-tuple  of the goal grid, e.g., (5, 1)
    #        obstacles: a list of grids marked as obstacles, e.g., [(2, -1), (2, 0), ...]
    # output path: a list of grids connecting start to goal, e.g., [(1, 0), (1, 1), ...]
    #   note that the path should contain the goal but not the start
    #   e.g., the path from (0, 0) to (2, 2) should be [(1, 0), (1, 1), (2, 1), (2, 2)]
    #Contains nodes that have not been visited yet
    open_list = []
    open_list.append((0, start))
    #Contains list of nodes that have been visited
    closed_list = []
    past_cost = {}
    past_cost[start] = 0
    #parent of the node previously visited
    parent = {}

    while len(open_list) > 0:
        open_list.sort()
        current = open_list.pop(0)[1]
        closed_list.append(current)

        if current == goal:
            break

        for nbr in neighbors(current):
            if nbr in obstacles:
                continue
            if nbr not in closed_list:
                new_cost = past_cost[current] + 0.5

                if nbr not in past_cost or new_cost < past_cost[nbr]:
                    past_cost[nbr] = new_cost
                    parent[nbr] = current

                    final = heuristic_distance(nbr, goal) + past_cost[nbr]
                    open_list.append((final, nbr))
    path = [] #List to store the path
    while current != start:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path


if __name__ == "__main__":
    start = (0, 0)  # this is a tuple data structure in Python initialized with 2 integers
    goal = (4.5,0)
    obstacles = [(0.5, 0.5), (0.5,1), (1.5, -0.5), (1.5,0), (3,0.5), (3,1)]
    path = get_path_from_A_star(start, goal, obstacles)
    print(path)
    whatever = Turtlebot()
