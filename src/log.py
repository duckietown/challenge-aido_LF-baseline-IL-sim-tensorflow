import cv2
from gym_duckietown.envs import DuckietownEnv
from pure_pursuit_policy import PurePursuitExpert
from _loggers import Logger

# Log configuration, you can pick your own values here
EPISODES = 10
STEPS = 512


env = DuckietownEnv(
    map_name='udem1',  # check the gym documentation, there are many maps of different complexity
    max_steps=EPISODES * STEPS
)
expert = PurePursuitExpert(env=env)

logger = Logger(env, log_file='train.log')
# let's collect our samples

for episode in range(0, EPISODES):
    for steps in range(0, STEPS):
        action = expert.predict(None)
        observation, reward, done, info = env.step(action)
        # we can resize the image here
        observation = cv2.resize(observation, (60, 80))
        logger.log(observation, action, reward, done, info)
        # [optional] env.render() to watch the expert interaction with the environment
        # we log here
    logger.on_episode_done() # speed logging by flushing the file
    env.reset()

# we flush everything and close the file, it should be ~ 120mb
logger.close()

env.close()
