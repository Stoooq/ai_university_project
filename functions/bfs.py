from copy import deepcopy
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

class node:
    def __init__(self, state):
        self.state = state
        self.parent = None
        self.action = None

def succ(x, y, direction):
    if direction == NORTH:
        return ([[[x - 1, y, direction], 'go'], [[x, y, WEST], 'left'], [[x, y, EAST], 'right']])
    elif direction == EAST:
        return ([[[x, y + 1, direction], 'go'], [[x, y, NORTH], 'left'], [[x, y, SOUTH], 'right']])
    elif direction == SOUTH:
        return ([[[x + 1, y, direction], 'go'], [[x, y, EAST], 'left'], [[x, y, WEST], 'right']])
    elif direction == WEST:
        return ([[[x, y - 1, direction], 'go'], [[x, y, SOUTH], 'left'], [[x, y, NORTH], 'right']])
    

def heuristic(start_x,start_y,chest_x, chest_y):
    return (abs(chest_x-start_x)+abs(chest_y-start_y)) // 32

def find_chest(chest_locations, start_x, start_y):
    shortest= heuristic(start_x,start_y,chest_locations[0][0],chest_locations[0][1])
    index=0
    x=0
    for loc in chest_locations:
        if heuristic(start_x,start_y,loc[0],loc[1])<shortest:
            shortest=heuristic(start_x,start_y,loc[0],loc[1])
            index=x
        x=x+1
    return chest_locations[index]



def cost(tile): #można by dać zmienen np chesttile lub carpettile
    if tile==0:
        cost=1
    elif tile==5:
        cost=5
    elif tile==3 or tile==4:
        cost=50
    elif tile==1:
        cost=100
    elif tile==2:
        cost=100
    else:
        cost=100

    return cost


def shortestPath(omap, start_x, start_y, direction,chest_locations):
    # for i in omap:
    #     print(i)
    chest=find_chest(chest_locations, start_x, start_y)
    grid = deepcopy(omap)
    fringe = []
    explored = []

    start_node = node([start_x, start_y, direction])
    fringe.append([start_node,0])
    explored.append((start_x, start_y, direction))

    while len(fringe) > 0:
        #print(fringe)
        elem,r = fringe.pop(0)
        ex, ey, ed = elem.state

        if chest[0] == ey and chest[1] == ex:
            path = []
            while elem.parent != None:
                path.append(elem.action)
                elem = elem.parent
            path.reverse()
            return (path[:-1], chest)
        explored.append((ex, ey, ed))
        for (state, action) in succ(ex, ey, ed):
            nx, ny, nd = state
            if not (0 <= nx < len(grid) and 0 <= ny < len(grid[0]) ):
                continue
            child = node([nx, ny, nd])
            child.parent = elem
            child.action = action
            chx,chy,chd= child.state
            p=heuristic(chx,chy,chest[0], chest[1])+cost(grid[chx][chy])
            if (nx, ny, nd) not in explored and not any((chx, chy, chd) == (n.state[0], n.state[1], n.state[2]) for (n, cost) in fringe):
                t=0
                for i in range(len(fringe)):
                    if fringe[i][1]>p+r:
                        fringe.insert(i,[child,p+r])
                        t=1
                        break
                if t==0:
                    fringe.append([child,p])
            elif any((chx, chy, chd) == (n.state[0], n.state[1], n.state[2]) for (n, cost) in fringe):
                for i in range(len(fringe)): 
                    if (chx, chy, chd)== fringe[i][0]:
                        if fringe[i][1] >p+r:
                            fringe[i]= [child,p+r]
    return [[],[]]


