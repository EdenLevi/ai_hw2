from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random

# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, taxi_id: int):
    # heuristic=(manhatten(agent,package)+ manhatten(package,destination) + manhatten(destination,charger)  > battery)  ? manhatten(agent,charger) ∶ manhatten(agent,package)
    robo = env.get_robot(taxi_id)

    #corner = (0, 0)
    #agent_to_corner = manhattan_distance(robo.position, corner)
    #return agent_to_corner



    min_package_position = env.packages[0].position
    if manhattan_distance(robo.position, env.packages[0].position) > manhattan_distance(robo.position, env.packages[1].position): min_package_distance = manhattan_distance(robo.position, env.packages[1].position)
    package_position = [min(x) for manhattan_distance(x.position, robo.position) in env.packages]

    agent_to_package = manhattan_distance(robo.position, env.packages[0].position)
    package_to_destination = manhattan_distance(env.packages[0].position, env.packages[0].destination)
    destination_to_charger = min(manhattan_distance(env.packages[0].destination, env.charge_stations[0].position), manhattan_distance(env.packages[0].destination, env.charge_stations[1].position))
    agent_to_charger = min(manhattan_distance(robo.position, env.charge_stations[0].position), manhattan_distance(robo.position, env.charge_stations[1].position))
    agent_to_destination = manhattan_distance(robo.position, env.packages[0].destination)

    # all variables are valid, now need to use them ^
    if robo.package:
        return 10 - agent_to_destination  # deliver package
    else:
        if (agent_to_package + package_to_destination) > robo.battery:
            return 10 - agent_to_charger  # go charge
        else:
            return 10 - agent_to_package  # get package

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)