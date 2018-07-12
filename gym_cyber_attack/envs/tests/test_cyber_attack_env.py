import unittest
import gym_cyber_attack.envs.cyber_attack_env as cyatk
from gym_cyber_attack.envs.cyber_attack_env import CyberAttackEnv
from gym_cyber_attack.envs.cyber_attack_spaces import CAObservationSpace


class EnvTestCase(unittest.TestCase):

    def setUp(self):
        self.nM = 4
        self.nS = 1
        self.env = CyberAttackEnv(self.nM, self.nS)
        self.obs_space = CAObservationSpace(self.nM, self.nS)
        self.targets = [(0, 0), (1, 0), (2, 0), (2, 1)]
        self.exploits = []
        self.scans = []
        for t in self.targets:
            self.exploits.append({"target": t, "action": 1})
            self.scans.append({"target": t, "action": 0})

    def test_reset(self):
        # test following no steps taken
        expected_obs = self.obs_space.get_initial_state()
        actual_obs = self.env.reset()
        self.assertDictEqual(expected_obs, actual_obs)
        # test following taking steps
        self.env.step(self.scans[0])
        actual_obs = self.env.reset()
        self.assertDictEqual(expected_obs, actual_obs)

    def test_step_not_reachable(self):
        action = self.scans[1]  # scan (1, 0)
        expected_obs = self.env.reset()
        o, r, d, _ = self.env.step(action)
        self.assertEqual(r, -cyatk.COST_SCAN)
        self.assertFalse(d)
        self.assertDictEqual(o, expected_obs)

    def test_step_scan_reachable(self):
        action = self.scans[0]  # scan (0, 0)
        expected_obs = self.env.reset()
        o, r, d, _ = self.env.step(action)
        expected_obs = self.update_obs(action, o, True)
        self.assertEqual(r, -cyatk.COST_SCAN)
        self.assertFalse(d)
        self.assertDictEqual(o, expected_obs)

    def test_step_exploit_reachable(self):
        action = self.exploits[0]  # exploit (0, 0), 0
        expected_obs = self.env.reset()
        o, r, d, _ = self.env.step(action)
        expected_obs = self.update_obs(action, o, True)
        self.assertEqual(r, -cyatk.COST_EXPLOIT)
        self.assertFalse(d)
        self.assertDictEqual(o, expected_obs)

    def test_step_exploit_sensitive(self):
        action1 = self.exploits[0]  # exploit (0, 0), 0
        action2 = self.exploits[1]  # exploit (1, 0), 0
        expected_obs = self.env.reset()
        o, r, d, _ = self.env.step(action1)
        expected_obs = self.update_obs(action1, o, True)
        o, r, d, _ = self.env.step(action2)
        expected_obs = self.update_obs(action2, o, True)

        self.assertEqual(r, cyatk.R_SENSITIVE - cyatk.COST_EXPLOIT)
        self.assertFalse(d)
        self.assertDictEqual(o, expected_obs)

    def test_step_exploit_user(self):
        action1 = self.exploits[0]  # exploit (0, 0), 0
        action2 = self.exploits[2]  # exploit (2, 0), 0
        expected_obs = self.env.reset()
        o, r, d, _ = self.env.step(action1)
        expected_obs = self.update_obs(action1, o, True)
        o, r, d, _ = self.env.step(action2)
        expected_obs = self.update_obs(action2, o, True)

        self.assertEqual(r, cyatk.R_USER - cyatk.COST_EXPLOIT)
        self.assertFalse(d)
        self.assertDictEqual(o, expected_obs)

    def test_step_done(self):
        action1 = self.exploits[0]  # exploit (0, 0), 0
        action2 = self.exploits[1]  # exploit (1, 0), 0
        action3 = self.exploits[2]  # exploit (2, 0), 0
        expected_obs = self.env.reset()
        o, r, d, _ = self.env.step(action1)
        expected_obs = self.update_obs(action1, o, True)
        o, r, d, _ = self.env.step(action2)
        expected_obs = self.update_obs(action2, o, True)
        o, r, d, _ = self.env.step(action3)
        expected_obs = self.update_obs(action3, o, True)

        self.assertTrue(d)
        self.assertDictEqual(o, expected_obs)

    def test_step_already_rewarded(self):
        action1 = self.exploits[0]  # exploit (0, 0), 0
        action2 = self.exploits[1]  # exploit (1, 0), 0
        expected_obs = self.env.reset()
        o, r, d, _ = self.env.step(action1)
        expected_obs = self.update_obs(action1, o, True)
        o, r, d, _ = self.env.step(action2)
        expected_obs = self.update_obs(action2, o, True)
        o, r, d, _ = self.env.step(action2)
        expected_obs = self.update_obs(action2, o, True)

        self.assertEqual(r, -cyatk.COST_EXPLOIT)
        self.assertFalse(d)
        self.assertDictEqual(o, expected_obs)

    def test_render(self):
        nM = 5
        nS = 1
        env = CyberAttackEnv(nM, nS)
        env.render()
        env.render()

    def update_obs(self, action, obs, success):
        """ Valid for test where E = 1 """
        m = action["target"]
        if success:
            # successfully scanned or exploited
            for s in range(len(obs[m]["services"])):
                obs[m]["services"][s] = cyatk.PRESENT
        if action["action"] != cyatk.SCAN and success:
            # successful exploit
            obs[m]["compromised"] = True
            if m[0] == 0:
                # exploited subnet 0 so all other subnets now reachable
                for o in obs.keys():
                    obs[o]["reachable"] = True
        elif action["action"] != cyatk.SCAN and not success:
            obs[m["services"][action["action"] - 1]] = cyatk.ABSENT
        return obs


if __name__ == "__main__":
    unittest.main()
