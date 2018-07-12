import unittest
from gym_cyber_attack.envs.cyber_attack_spaces import AddressSpace
from gym_cyber_attack.envs.cyber_attack_spaces import CAActionSpace
from gym_cyber_attack.envs.cyber_attack_spaces import CAObservationSpace


class SpacesTestCase(unittest.TestCase):

    def setUp(self):
        self.nM = 3
        self.nS = 4
        self.address_space = AddressSpace(self.nM)

    def roundtrip(self, space):
        sample_1 = space.sample()
        sample_2 = space.sample()
        self.assertTrue(space.contains(sample_1))
        self.assertTrue(space.contains(sample_2))

    def test_address_space(self):
        self.roundtrip(self.address_space)

    def test_action_space(self):
        action_space = CAActionSpace(self.nM, self.nS)
        self.roundtrip(action_space)

    def test_observation_space(self):
        observation_space = CAObservationSpace(self.nM, self.nS)
        self.roundtrip(observation_space)

        init_obs = observation_space.get_initial_state()
        self.assertTrue(observation_space.contains(init_obs))


if __name__ == "__main__":
    unittest.main()
