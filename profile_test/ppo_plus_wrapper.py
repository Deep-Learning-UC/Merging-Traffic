from ppo_plus.agent import run_training, RunConfig, PPOPlusConfig
from profile_test_environment import make_env

class Metrics:
    def log(self, data, step=None, commit=True):
        print("METRICS", data)
    def finish(self):
        print("FINISH")

class PPOPlusWrapper:
    def __init__(self, ppo_config, run_config, env):
        self.ppo_config = ppo_config
        self.run_config = run_config
        pass

    def learn():
        pass

    def save():
        pass

    def predict():
        pass

    def load():
        pass




if __name__ == "__main__":

    # smoke test 
    # i think it's running
    run_training(
        learner_config=PPOPlusConfig(),
        # run_config=RunConfig(total_steps=300_000, buffer_size=50_000, policy_steps=5_000,
        #   num_epochs=10, on_policy_data=False, log_interval=10**9),
        run_config=RunConfig(total_steps=10_000, buffer_size=50_000, policy_steps=1_000,
          num_epochs=10, on_policy_data=False, log_interval=10**9),
        env=make_env(),
        eval_env=None,
        metrics=Metrics()
    )
    