from Intersection import Intersection
from GUI import GUI


length = 5
prob = 0.5
keepalive = -1
wait_weight = 0.5

state_size = 2*(length + 2)
action_size = 3
        
if __name__ == '__main__':
    game = Intersection(length=length, prob = prob, keepalive = keepalive, wait_weight=wait_weight)
    window = GUI(50)

    window.update(game.getState())

    while not game.gameEnd():
    	text = raw_input("Input (0 is for nothing, 1 is for NS, 2 is for EW: ")
    	stepReward = 0
    	if(int(text) == 0):
    		stepReward = game.step(0,0)
    	elif(int(text) == 1):
    		stepReward = game.step(1,0)
    	else:
    		stepReward = game.step(0,1)

    	print("Step Reward: ", stepReward)
    	print("Total Score: ", game.getReward())
        game.gameWait()
    	window.update(game.getState())
