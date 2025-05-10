import argparse
import jax
import jax.numpy as jnp
import equinox as eqx
import pickle
import cv2
from src.tetris import Tetris

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=30, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--result", type=str, default="result.mp4")
    args = parser.parse_args()
    return args

def test(opt):
    with open("{}/tetris_5000.pkl".format(opt.saved_path), "rb") as f:
        model = pickle.load(f)
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()
    out = cv2.VideoWriter(opt.result, cv2.VideoWriter_fourcc(*"mp4v"), opt.fps,
                          (int(1.5 * opt.width * opt.block_size), opt.height * opt.block_size))
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = jnp.array(next_states)  # Shape: (num_next_states, 4)
        predictions = jax.vmap(model)(next_states).squeeze()  # Shape: (num_next_states,)
        index = jnp.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True, video=out)
        if done:
            out.release()
            break

if __name__ == "__main__":
    opt = get_args()
    test(opt)