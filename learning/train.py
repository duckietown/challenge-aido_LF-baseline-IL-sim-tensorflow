import numpy as np
from tqdm import tqdm

from _loggers import Reader
from model import TensorflowModel
from eval import eval
# configuration zone
BATCH_SIZE = 32
EPOCHS = 1000
# here we assume the observations have been resized to 60x80
OBSERVATIONS_SHAPE = (None, 32, 32, 3)
ACTIONS_SHAPE = (None, 2)
SEED = 1234
STORAGE_LOCATION = "trained_models/behavioral_cloning"

np.random.seed(SEED)


def shuffle(data, labels):
    p = np.random.permutation(len(data))
    return data[p], labels[p]


reader = Reader('train.log')

observations, actions = reader.read()
observations, actions = shuffle(np.array(observations), np.array(actions))

model = TensorflowModel(
    observation_shape=OBSERVATIONS_SHAPE,  # from the logs we've got
    action_shape=ACTIONS_SHAPE,  # same
    graph_location=STORAGE_LOCATION,  # where do we want to store our trained models
    seed=SEED  # to seed all random operations in the model (e.g., dropout)
)

# we trained for EPOCHS epochs
epochs_bar = tqdm(range(EPOCHS))
best_reward = float('-inf')
for i in epochs_bar:
    # we defined the batch size, this can be adjusted according to your computing resources...
    loss = 0.0
    for batch in range(0, len(observations), BATCH_SIZE):
        loss += model.train(
            observations=observations[batch:batch + BATCH_SIZE],
            actions=actions[batch:batch + BATCH_SIZE]
        )

        epochs_bar.set_postfix({'loss': loss / BATCH_SIZE})

    # every 5 epochs, we store the model we have

    if i % 10 == 0:
        avg_reward = eval(model, display=True)
        if avg_reward > best_reward:
            best_reward = avg_reward
            model.commit()
            epochs_bar.set_description('New model saved')
        epochs_bar.set_postfix({'loss': loss / BATCH_SIZE, 'avg_rew': avg_reward})
    else:
        epochs_bar.set_description('')

# the loss at this point should be on the order of 2e-2, which is far for great, right?

# we release the resources...
model.close()
reader.close()

