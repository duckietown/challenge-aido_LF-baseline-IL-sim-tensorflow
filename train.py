import numpy as np
from tqdm import tqdm


from _loggers import Reader
from model import TensorflowModel

BATCH_SIZE = 32
EPOCHS = 10

reader = Reader('train.log')

observations, actions = reader.read()
actions = np.array(actions)
observations = np.array(observations)

model = TensorflowModel(
    observation_shape=(None, 60, 80, 3),  # from the logs we've got
    action_shape=(None, 2),  # same
    graph_location="trained_models/behavioral_cloning",  # where do we want to store our trained models
    seed=1234  # to seed all random operations in the model (e.g., dropout)
)


for i in tqdm(range(EPOCHS)):
    for batch in range(0, len(observations), BATCH_SIZE):
        loss = model.train(
            observations=observations[batch:batch + BATCH_SIZE],
            actions=actions[batch:batch + BATCH_SIZE]
        )
    print(loss)

    if i % 10 == 0:
        model.commit()

