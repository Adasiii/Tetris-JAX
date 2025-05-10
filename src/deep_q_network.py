"""
Traditional DQN and our Double DQN implementations
"""

# import jax
# import jax.numpy as jnp
# import equinox as eqx
# import jax.random as random

# class DeepQNetwork(eqx.Module):
#     conv1: eqx.nn.Linear
#     conv2: eqx.nn.Linear
#     conv3: eqx.nn.Linear

#     def __init__(self, key):
#         key1, key2, key3 = random.split(key, 3)
        
#         # Initialize weights and biases for conv1
#         in_features1, out_features1 = 4, 64
#         bound1 = 1.0 / jnp.sqrt(in_features1)
#         weight1 = random.uniform(key1, (out_features1, in_features1), minval=-bound1, maxval=bound1)
#         bias1 = jnp.zeros(out_features1)
#         self.conv1 = eqx.nn.Linear(in_features1, out_features1, use_bias=True, key=key1)
#         self.conv1 = eqx.tree_at(lambda m: m.weight, self.conv1, weight1)
#         self.conv1 = eqx.tree_at(lambda m: m.bias, self.conv1, bias1)
        
#         # Initialize weights and biases for conv2
#         in_features2, out_features2 = 64, 64
#         bound2 = 1.0 / jnp.sqrt(in_features2)
#         weight2 = random.uniform(key2, (out_features2, in_features2), minval=-bound2, maxval=bound2)
#         bias2 = jnp.zeros(out_features2)
#         self.conv2 = eqx.nn.Linear(in_features2, out_features2, use_bias=True, key=key2)
#         self.conv2 = eqx.tree_at(lambda m: m.weight, self.conv2, weight2)
#         self.conv2 = eqx.tree_at(lambda m: m.bias, self.conv2, bias2)
        
#         # Initialize weights and biases for conv3
#         in_features3, out_features3 = 64, 1
#         bound3 = 1.0 / jnp.sqrt(in_features3)
#         weight3 = random.uniform(key3, (out_features3, in_features3), minval=-bound3, maxval=bound3)
#         bias3 = jnp.zeros(out_features3)
#         self.conv3 = eqx.nn.Linear(in_features3, out_features3, use_bias=True, key=key3)
#         self.conv3 = eqx.tree_at(lambda m: m.weight, self.conv3, weight3)
#         self.conv3 = eqx.tree_at(lambda m: m.bias, self.conv3, bias3)

#     def __call__(self, x):
#         x = jax.nn.relu(self.conv1(x))
#         x = jax.nn.relu(self.conv2(x))
#         x = self.conv3(x)
#         return x


import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as random

class DeepQNetwork(eqx.Module):
    conv1: eqx.nn.Linear
    conv2: eqx.nn.Linear
    conv3: eqx.nn.Linear
    target_conv1: eqx.nn.Linear
    target_conv2: eqx.nn.Linear
    target_conv3: eqx.nn.Linear

    def __init__(self, key):
        key1, key2, key3, key4, key5, key6 = random.split(key, 6)
        
        # Initialize weights and biases for conv1
        in_features1, out_features1 = 4, 64
        bound1 = 1.0 / jnp.sqrt(in_features1)
        weight1 = random.uniform(key1, (out_features1, in_features1), minval=-bound1, maxval=bound1)
        bias1 = jnp.zeros(out_features1)
        self.conv1 = eqx.nn.Linear(in_features1, out_features1, use_bias=True, key=key1)
        self.conv1 = eqx.tree_at(lambda m: m.weight, self.conv1, weight1)
        self.conv1 = eqx.tree_at(lambda m: m.bias, self.conv1, bias1)
        
        # Initialize weights and biases for conv2
        in_features2, out_features2 = 64, 64
        bound2 = 1.0 / jnp.sqrt(in_features2)
        weight2 = random.uniform(key2, (out_features2, in_features2), minval=-bound2, maxval=bound2)
        bias2 = jnp.zeros(out_features2)
        self.conv2 = eqx.nn.Linear(in_features2, out_features2, use_bias=True, key=key2)
        self.conv2 = eqx.tree_at(lambda m: m.weight, self.conv2, weight2)
        self.conv2 = eqx.tree_at(lambda m: m.bias, self.conv2, bias2)
        
        # Initialize weights and biases for conv3
        in_features3, out_features3 = 64, 1
        bound3 = 1.0 / jnp.sqrt(in_features3)
        weight3 = random.uniform(key3, (out_features3, in_features3), minval=-bound3, maxval=bound3)
        bias3 = jnp.zeros(out_features3)
        self.conv3 = eqx.nn.Linear(in_features3, out_features3, use_bias=True, key=key3)
        self.conv3 = eqx.tree_at(lambda m: m.weight, self.conv3, weight3)
        self.conv3 = eqx.tree_at(lambda m: m.bias, self.conv3, bias3)

        # Initialize target network weights and biases
        self.target_conv1 = eqx.nn.Linear(in_features1, out_features1, use_bias=True, key=key4)
        self.target_conv1 = eqx.tree_at(lambda m: m.weight, self.target_conv1, weight1)
        self.target_conv1 = eqx.tree_at(lambda m: m.bias, self.target_conv1, bias1)
        
        self.target_conv2 = eqx.nn.Linear(in_features2, out_features2, use_bias=True, key=key5)
        self.target_conv2 = eqx.tree_at(lambda m: m.weight, self.target_conv2, weight2)
        self.target_conv2 = eqx.tree_at(lambda m: m.bias, self.target_conv2, bias2)
        
        self.target_conv3 = eqx.nn.Linear(in_features3, out_features3, use_bias=True, key=key6)
        self.target_conv3 = eqx.tree_at(lambda m: m.weight, self.target_conv3, weight3)
        self.target_conv3 = eqx.tree_at(lambda m: m.bias, self.target_conv3, bias3)

    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def target_call(self, x):
        x = jax.nn.relu(self.target_conv1(x))
        x = jax.nn.relu(self.target_conv2(x))
        x = self.target_conv3(x)
        return x

    def update_target_network(self):
        self.target_conv1 = eqx.tree_at(lambda m: m.weight, self.target_conv1, self.conv1.weight)
        self.target_conv1 = eqx.tree_at(lambda m: m.bias, self.target_conv1, self.conv1.bias)
        self.target_conv2 = eqx.tree_at(lambda m: m.weight, self.target_conv2, self.conv2.weight)
        self.target_conv2 = eqx.tree_at(lambda m: m.bias, self.target_conv2, self.conv2.bias)
        self.target_conv3 = eqx.tree_at(lambda m: m.weight, self.target_conv3, self.conv3.weight)
        self.target_conv3 = eqx.tree_at(lambda m: m.bias, self.target_conv3, self.conv3.bias)
