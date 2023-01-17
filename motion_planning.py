def neighbors(current):
    # define the list of 4 neighbors
    # For any point, 4 neighbors are the point to the above, below, left and right
    # Therefore, they are +/- on the x and y-axis
    neighbors = [[0, 1], [0, -1], [1, 0], [-1, 0]]
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
                new_cost = past_cost[current] + 1

                if nbr not in past_cost or new_cost < past_cost[nbr]:
                    past_cost[nbr] = new_cost
                    parent[nbr] = current

                    final = heuristic_distance(nbr, goal) + past_cost[nbr]
                    open_list.append((final, nbr))
    path = [] #List to store the path
    while current != start:
        path.append(current)
        current = parent[current]

    path = path[::-1]
    return path


if __name__ == "__main__":
    start = (0, 0)  # this is a tuple data structure in Python initialized with 2 integers
    goal = (-5, -2)
    obstacles = [(-2, 1), (-2, 0), (-2, -1), (-2, -2), (-4, -2), (-4, -3)]
    path = get_path_from_A_star(start, goal, obstacles)
    print(path)
