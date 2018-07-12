import sys
from collections import OrderedDict
import copy
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_cyber_attack.envs.cyber_attack_spaces import AddressSpace
from gym_cyber_attack.envs.cyber_attack_spaces import CAActionSpace
from gym_cyber_attack.envs.cyber_attack_spaces import CAObservationSpace


# Service states
UNKNOWN = 0
PRESENT = 1
ABSENT = 2

# action type
SCAN = 0

# Value of different machines
R_SENSITIVE = 9000.0
R_USER = 5000.0

# Action costs
COST_EXPLOIT = 10.0
COST_SCAN = 10.0


class CyberAttackEnv(gym.Env):
    """
    A cyber attack simulator environment.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, nM=5, nS=3):
        self.nM = nM
        self.nS = nS
        self.action_space = CAActionSpace(nM, nS)
        self.observation_space = CAObservationSpace(nM, nS)
        self._construct_network(nM, nS)

        self.reset()

    def _construct_network(self, nM, nS):
        """
        Generate the network for the environment.

        Arguments:
            int nM : total number of machines in network
            int nS : number of possible services running on a machine

        Generates:
            dict network : dictionary of machine address as keys and their
                service configuration (array of bools) and value as values
            list sensitive_machines : list of addresses of machines that have
                sensitive documents (i.e. have rewards)
        """
        # set seed so its consistent across runs
        np.random.seed(1)
        self.network = OrderedDict()
        self.sensitive_machines = []
        # set of possible service configurations
        configs = np.asarray(permutations(nS)[:-1])
        for m in AddressSpace.generate_address_space(nM):
            m_cfg = configs[np.random.choice(configs.shape[0])]
            m_value = 0
            if m[1] % 10 == 0:
                if m[0] == 1:
                    # 10th machine in subnet 1
                    m_value = R_SENSITIVE
                    self.sensitive_machines.append(m)
                elif m[0] == 2:
                    # 10th machine in subnet 2
                    m_value = R_USER
                    self.sensitive_machines.append(m)
            self.network[m] = {"services": m_cfg, "value": m_value}

    def reset(self):
        """
        Reset the state of the environment and returns the initial observation.

        Returns:
            dict obs : the intial observation of the network environment
        """
        self.current_state = self.observation_space.get_initial_state()
        return self.current_state

    def step(self, action):
        """
        Run one step of the environment using action.

        Arguments:
            Action action : Action object from action_space

        Returns:
            dict observation : agent's observation of the network
            float reward : reward from performing action
            bool done : whether the episode has ended or not
            dict info : contains extra info useful for debugging or
                visualization
        """
        target = action["target"]
        action_cost = COST_SCAN if action["action"] == SCAN else COST_EXPLOIT

        if (not self.current_state[target]["reachable"] or
                self.current_state[target]["compromised"]):
            # machine not reachable or already compromised so no state change
            return self.current_state, 0 - action_cost, False, {}

        success, value, services = self._take_action(action)
        self._update_state(action, success, services)
        done = self.is_goal()
        reward = value - action_cost
        return copy.deepcopy(self.current_state), reward, done, {}

    def _take_action(self, action):
        """
        Attempt to perform given action against this machine

        Arguments:
            Action action : the exploit Action

        Returns:
            bool success : True if exploit/scan was successful, False otherwise
            float value : value gained from action. Is the value of the machine
                if successfuly exploited, otherwise 0 if unsuccessful or scan.
            list services : the list of services identified by action. This is
                services if exploit was successful or scan, otherwise an empty
                list
        """
        target = action["target"]
        target_services = self.network[target]["services"]
        if action["action"] == SCAN:
            return True, 0, target_services
        elif target_services[action["action"] - 1]:
            # target service is present, so exploit is successful
            return True, self.network[target]["value"], target_services
        else:
            # target service absent, so exploit fails
            return False, 0, np.asarray([])

    def _update_state(self, action, success, services):
        """
        Updates the current observation of network state based on if action was
        successful and the gained service info

        Arguments:
            Action action : the action performed
            bool success : whether action was successful
            list services : service info gained from action
        """
        target = action["target"]
        # current knowledge of target machines state
        target_state = self.current_state[target]["services"]
        if not success:
            # exploit failed so we know target service is absent
            target_state[action["action"] - 1] = ABSENT
        else:
            # exploit or scan was successful so all target service info gained
            for i, s in enumerate(services):
                target_state[i] = PRESENT if s else ABSENT
            if action["action"] != SCAN:
                self.current_state[target]["compromised"] = 1
                self._update_reachable(target)

    def _update_reachable(self, target):
        """
        Updates the reachable status of machines on network, based on current
        observation and newly exploited machine

        Arguments:
            Address target : compromised machine address
        """
        # if machine on subnet 0 is compromised then subnets 1 and 2 are
        # reachable
        if target[0] == 0:
            for m, v in self.current_state.items():
                v["reachable"] = True
        # otherwise all machines all already reachable
        # TODO update for when more than 3 subnets

    def is_goal(self):
        """
        Check if the current state is the goal state. Where the goal
        state is defined as when all sensitive documents have been collected
        (i.e. all machines containing sensitive documents have been
        compromised)

        Returns:
            bool goal : True if goal state, otherwise False
        """
        for m in self.sensitive_machines:
            if not self.current_state[m]["compromised"]:
                return False
        return True

    def render(self, mode='human', close=False):
        """
        Render to stdout. Machines displayed in columns, with one column for
        each subnet and machines displayed in order of id within subnet

        Key, for each machine:
        C   sensitive & compromised
        R   sensitive & reachable
        S   sensitive
        c   compromised
        r   reachable
        o   non-of-above
        """
        outfile = sys.stdout

        subnets = [[], [], []]
        for m, v in self.current_state.items():
            subnets[m[0]].append(self._get_machine_symbol(v))

        max_row_size = max([len(x) for x in subnets])
        min_row_size = min([len(x) for x in subnets])

        output = "\n-----------------------------"
        for i, row in enumerate(subnets):
            output += "\nsubnet {0}: ".format(i)
            output += " " * ((max_row_size - len(row)) // 2)
            for col in row:
                output += col
            output += "\n"
            if i < len(subnets) - 1:
                n_spaces = (max_row_size - min_row_size) // 2
                output += " " * (len("subnet X: ") + n_spaces) + "|"
        output += "-----------------------------\n"

        outfile.write(output)

    def _get_machine_symbol(self, m_state):
        if m_state["sensitive"]:
            if m_state["compromised"]:
                symbol = "C"
            elif m_state["reachable"]:
                symbol = "R"
            else:
                symbol = "S"
        elif m_state["compromised"]:
            symbol = "c"
        elif m_state["reachable"]:
            symbol = "r"
        else:
            symbol = "o"
        return symbol


def permutations(n):
    """
    Generate list of all possible permutations of n bools
    """
    # base cases
    if n <= 0:
        return []
    if n == 1:
        return [[True], [False]]

    perms = []
    for p in permutations(n - 1):
        perms.append([True] + p)
        perms.append([False] + p)
    return perms
