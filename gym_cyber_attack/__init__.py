from gym.envs.registration import register

register(
    id='cyber_attack-v0',
    entry_point='gym_cyber_attack.envs:CyberAttackEnv',
)
