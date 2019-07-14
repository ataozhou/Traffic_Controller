from IntersectionRoads import Intersection
from GUI import GUI


length = 5
prob = 0.5
keepalive = -1
wait_weight = 0.5

state_size = 2*(length + 2)
action_size = 3
        
if __name__ == '__main__':
    game = Intersection(numRoads = 2, length = 3, prob = 1, speed = 1, wait_weight = 1)
    window = GUI(50)


    while not game.gameEnd():

    	text = raw_input("Input (0 is for nothing, 1 is for NS, 2 is for EW: ")
    	stepReward = 0
    	if(int(text) == 0):
    		stepReward = game.step(0)
    	elif(int(text) == 1):
    		stepReward = game.step(1)
    	elif(int(text) == 2):
    		stepReward = game.step(2)
        else:
            stepReward = game.step(3)

        print(game.binaryRepresentation())
        print(game.getIntersection())

    	print("Step Reward: ", stepReward)
    	print("Total Score: ", game.getReward())
        print("Total Wait: ", game.getWait())
    	window.update(game.binaryRepresentation())
