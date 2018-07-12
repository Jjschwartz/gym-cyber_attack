from bisect import bisect_left
import copy
from collections import OrderedDict
import gym
import gym.spaces as spaces
import numpy as np

# Service states
UNKNOWN = 0
PRESENT = 1
ABSENT = 2


class AddressSpace(gym.Space):
    """
    Defines the possible addresses for machines on network

    An address is a (subnet, id) tuple, where:
    - subnet - is the subnet the machine is located on
    - id - is the id of the machine within the subnet

    Address space is dictated based on network construction rules detailed in
    environment description.
    """

    def __init__(self, nM):
        """
        Construct the space

        Arguments:
            int nM : total number of machines on network
        """
        self.nM = nM
        self.space = AddressSpace.generate_address_space(nM)
        gym.Space.__init__(self, None, None)

    @staticmethod
    def generate_address_space(nM):
        # network must have a minimum of 3 machines
        assert 2 < nM
        addresses = []
        s_id = 0
        u_id = 0
        for m in range(nM):
            if m == 0:
                # first machine is in subnet 0 (exposed)
                addresses.append((0, 0))
            elif m % 10 == 1:
                # every 10th machine (from 1) is in subnet 1 (sensitive)
                addresses.append((1, s_id))
                s_id += 1
            else:
                # all other machines are in subnet 2 (user)
                addresses.append((2, u_id))
                u_id += 1
        addresses.sort()
        return addresses

    def sample(self):
        """
        Uniformly randomly sample a random address from address space
        """
        return self.space[spaces.np_random.randint(self.nM)]

    def contains(self, x):
        """
        Return boolean specifying if x is a valid address from this space
        """
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        return (isinstance(x, tuple) and len(x) == 2 and
                self.space[bisect_left(self.space, x)] == x)

    def __repr__(self):
        return "AddressSpace: " + str(self.space)


class CAActionSpace(spaces.Dict):
    """
    Defines the action space for Cyber Attack environment.

    A dictionary of action attributes:
    'target' : AddressSpace(nM)
        - the address of target machine
    'action' : spaces.Discrete(nS)
        - the number of action to perform, where scan = 0, and
          exploit service n = (n - 1)

    Has following properties:
    - nM : total number of machines in network
    - nS : possible number of services running on each machine
    """
    def __init__(self, nM, nS):
        assert nM > 2
        assert nS > 0
        spaces.Dict.__init__(self, {"target": AddressSpace(nM),
                                    "action": spaces.Discrete(nS + 1)})


class CAObservationSpace(gym.Spaces):
    """
    Defines the observation space for Cyber Attack environment.

    A dictionary of observed state of each machine in network:
    - keys = (subnet, id) tuples (i.e. AddressSpace)
    - values = spaces.Dict()
        'services' : spaces.MultiDiscrete([ nS, 3])
            - whether service is unknown, present, absent
        'compromised' : spaces.Discrete(2)
            - whether machine is compromised or not
        'reachable' : spaces.Discrete(2)
            - whether machine is reachable or not
        'sensitive' : spaces.Discrete(2)
            - whether machine contains sensitive documents or not

    Has following properties:
    - nM : total number of machines in network
    - nS : possible number of services running on each machine
    """
    def __init__(self, nM, nS):
        assert nM > 2
        assert nS > 0
        self.nM = nM
        self.nS = nS

        self.address_space = AddressSpace(nM)
        self.space = OrderedDict()
        for m in self.address_space.space:
            self.space[m] = spaces.Dict({
                "services": spaces.MultiDiscrete([3] * nS),
                "compromised": spaces.Discrete(2),
                "reachable": spaces.Discrete(2),
                "sensitive": spaces.Discrete(2)})

        gym.Space.__init__(self, None, None)

        self.initial_observation = self.gen_initial_state(nM, nS)

    def gen_initial_state(self, nM, nS):
        init_state = OrderedDict()
        for m in self.address_space.space:
            services = np.full(nS, UNKNOWN, dtype=np.int8)
            compromised = False
            if m[0] == 0:
                # machine on subnet 0, has no sensitive info
                sensitive = False
                # only machines on exposed subnet are reachable at start
                reachable = True
            else:
                # every 10th machine has sensitive info
                sensitive = m[1] % 10 == 0
                reachable = False
            init_state[m] = OrderedDict({"services": services,
                                         "compromised": compromised,
                                         "sensitive": sensitive,
                                         "reachable": reachable})
        return init_state

    def get_initial_state(self):
        return copy.deepcopy(self.initial_observation)

    def sample(self):
        """
        Uniformly randomly sample a random observation from space
        """
        sample = OrderedDict()
        for m in self.address_space.space:
            sample[m] = self.space[m].sample()
        return sample

    def contains(self, x):
        """
        Return boolean specifying if x is a valid observation from this space
        """
        if not isinstance(x, dict) or len(x) != len(self.space):
            return False
        for m in x.keys():
            if not self.address_space.contains(m):
                return False
            if not self.space[m].contains(x[m]):
                return False
        return True

    def __repr__(self):
        return "ObservationSpace: " + str(self.space)
