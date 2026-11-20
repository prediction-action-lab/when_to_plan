"""

Implementation of heuristic policies and function "get" each one for baselines

"""

import random


class HeuristicPolicy:
    def __init__(self, policy_type):
        self.policy_type = policy_type

    def get_action(self):
        if self.policy_type == "always_model_free":
            return self.always_model_free_policy()
        elif self.policy_type == "always_short_plan":
            return self.always_short_plan_policy()
        elif self.policy_type == "always_medium_plan":
            return self.always_medium_plan_policy()
        elif self.policy_type == "always_long_plan":
            return self.always_long_plan_policy()
        elif self.policy_type == "random_policy":
            return self.random_policy()
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")

    def random_policy(self):
        return random.choice([0, 1, 2, 3])

    def always_model_free_policy(self):
        return 0

    def always_short_plan_policy(self):
        return 1

    def always_medium_plan_policy(self):
        return 2

    def always_long_plan_policy(self):
        return 3
