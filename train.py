import argparse
import os
import pickle
import shutil
from random import sample
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tensorboardX import SummaryWriter
import time
from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_decay_epochs", type=float, default=5000)
    parser.add_argument("--num_epochs", type=int, default=7000)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--replay_memory_size", type=int, default=100000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args

def train(opt):
    key = jax.random.PRNGKey(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    key, subkey = jax.random.split(key)
    model = DeepQNetwork(subkey)
    optim = optax.adam(opt.lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    state = env.reset()
    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    t1 = time.time()
    total_time = 0
    best_score = 1000

    @eqx.filter_jit
    def loss_fn(model, state_batch, reward_batch, next_state_batch, done_batch, gamma):
        q_values = jax.vmap(model)(state_batch).squeeze()
        next_predictions = jax.vmap(model)(next_state_batch).squeeze()
        targets = jnp.where(done_batch, reward_batch, reward_batch + gamma * next_predictions)
        return jnp.mean((q_values - targets) ** 2)

    @eqx.filter_jit
    def train_step(model, opt_state, batch, gamma):
        state_batch, reward_batch, next_state_batch, done_batch = batch
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, state_batch, reward_batch, next_state_batch, done_batch, gamma)
        updates, new_opt_state = optim.update(grads, opt_state)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    best = 0
    while epoch < opt.num_epochs:
        start_time = time.time()
        next_steps = env.get_next_states()
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey)
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = jnp.array([state for state in next_states])
        predictions = jax.vmap(model)(next_states).squeeze()
        if random_action:
            key, subkey = jax.random.split(key)
            index = jax.random.randint(subkey, (), 0, len(next_steps))
        else:
            index = jnp.argmax(predictions)
        action = next_actions[index]
        next_state = next_states[index]
        reward, done = env.step(action, render=True)
        replay_memory.append([state, reward, next_state, done])
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = jnp.array(state_batch)
        next_state_batch = jnp.array(next_state_batch)
        reward_batch = jnp.array(reward_batch)
        done_batch = jnp.array(done_batch, dtype=jnp.float32)
        model, opt_state, loss = train_step(model, opt_state, (state_batch, reward_batch, next_state_batch, done_batch), opt.gamma)
        end_time = time.time()
        use_time = end_time - t1 - total_time
        total_time = end_time - t1
        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}, Used time: {}, total used time: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines,
            use_time,
            total_time))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)
        if epoch > 0 and epoch % opt.save_interval == 0:
            print("save interval model: {}".format(epoch))
            with open("{}/tetris_{}.pkl".format(opt.saved_path, epoch), "wb") as f:
                pickle.dump(model, f)
        # elif final_score > best_score:
        #     best_score = final_score
        #     print("save best model: {}".format(best_score))
        #     with open("{}/tetris_{}.pkl".format(opt.saved_path, best_score), "wb") as f:
        #         pickle.dump(model, f)
        if final_score > best:
            best = final_score
            print("save best model: {}".format(epoch))
            with open("{}/tetris_best_{}.pkl".format(opt.saved_path, epoch), "wb") as f:
                pickle.dump(model, f)


if __name__ == "__main__":
    opt = get_args()
    train(opt)

