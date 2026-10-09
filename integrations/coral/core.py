import copy
from dataclasses import asdict, dataclass
import math
import random
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    dimension: int
    hidden: int
    train_steps: int
    eval_steps: int
    learning_starts: int
    early_stop_tolerance: float
    max_updates: int
    actor_lr: float
    critic_lr: float
    gamma: float
    tau: float
    ou_theta: float
    ou_sigma: float
    ou_dt: float
    replay_capacity: int
    batch_size: int
    updates_per_step: int
    epochs: int
    episodes_per_epoch: int
    seed: int
    rating_threshold: float
    expected_candidates: int
    state_transition_training: str
    embedding_policy: str
    yes_token_text: str

    def __post_init__(self):
        for key in ('dimension', 'hidden', 'train_steps', 'eval_steps', 'max_updates', 'replay_capacity', 'batch_size', 'updates_per_step', 'epochs', 'episodes_per_epoch', 'expected_candidates'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key + ' must be a positive integer')
        if self.batch_size > self.replay_capacity or max(self.train_steps, self.eval_steps) > 1000 or type(self.learning_starts) is not int or self.learning_starts < 0:
            raise ValueError('invalid replay batch or retrieval step bound')
        for key in ('actor_lr', 'critic_lr', 'gamma', 'tau', 'ou_theta', 'ou_sigma', 'ou_dt', 'rating_threshold', 'early_stop_tolerance'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(key + ' must be finite')
        if min(self.actor_lr, self.critic_lr, self.ou_dt) <= 0 or min(self.ou_theta, self.ou_sigma) < 0:
            raise ValueError('learning rates, OU scale or time step are invalid')
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1 or type(self.seed) is not int or not 0 <= self.seed < 2 ** 32:
            raise ValueError('gamma, tau or seed is invalid')
        if not 0 <= self.early_stop_tolerance <= 1:
            raise ValueError('early_stop_tolerance must be in [0,1]')
        if self.state_transition_training != 'critic_td_recomputed_history_v1' or self.embedding_policy != 'external_train_only_frozen':
            raise ValueError('explicit supported reconstruction policies are required')
        if not isinstance(self.yes_token_text, str) or not self.yes_token_text:
            raise ValueError('yes_token_text must be explicitly provided')

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError('CoRAL config must specify exactly every configuration field')
        return cls(**value)


class Ratings:
    def __init__(self, items, records, threshold):
        self.items, self.threshold, self.records = items, threshold, []
        seen = set()
        if not isinstance(items, dict) or not items:
            raise ValueError('catalog must be a nonempty object')
        for key, item in items.items():
            if not isinstance(key, str) or not key or not isinstance(item, dict) or not isinstance(item.get('title'), str) or not item['title'].strip():
                raise ValueError('catalog requires nonempty item IDs and titles')
        for row in records:
            if row.get('split') != 'train' or not isinstance(row.get('user_id'), str) or not row['user_id'] or row.get('item_id') not in items:
                raise ValueError('ratings must be explicit training user/item records')
            if type(row.get('timestamp')) is not int or row['timestamp'] < 0:
                raise ValueError('rating timestamp must be integer milliseconds')
            value = row.get('rating')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('explicit finite observed ratings are required; missing interactions are not dislikes')
            identity = (row['user_id'], row['item_id'], row['timestamp'])
            if identity in seen:
                raise ValueError('duplicate timestamped user/item rating')
            seen.add(identity)
            self.records.append({**row, 'label': int(value >= threshold)})
        self.records.sort(key=lambda row: (row['timestamp'], row['user_id'], row['item_id']))
        if not self.records or {row['label'] for row in self.records} != {0, 1}:
            raise ValueError('CoRAL requires observed positive and negative training ratings')

    def before(self, user, item, moment):
        matrix = {}
        for row in self.records:
            if row['timestamp'] >= moment:
                break
            pair = (row['user_id'], row['item_id'])
            if pair != (user, item):
                matrix[pair] = row['label']
        return matrix

    def prompt(self, user, item, matrix, users, items):
        lines = ['As a recommender system please solve the following problem.']
        for candidate in dict.fromkeys(items):
            positive = [value for value in dict.fromkeys(users) if matrix.get((value, candidate)) == 1]
            negative = [value for value in dict.fromkeys(users) if matrix.get((value, candidate)) == 0]
            title = self.items[candidate]['title']
            lines += [f'The item {candidate}: {title} is liked by the users {positive}.',
                      f'The item {candidate}: {title} is disliked by the users {negative}.',
                      f'Try to understand what kinds of users typically like {title} based on the above information.']
        for label, name in ((1, 'likes'), (0, 'does not like')):
            observed = [f'{value}: {self.items[value]["title"]}' for (owner, value), rating in sorted(matrix.items()) if owner == user and rating == label]
            lines.append(f'Items the user {user} {name}: {observed}.')
        lines.append(f'For the item {item}: {self.items[item]["title"]}, would you recommend it to user {user}? Answer Yes or No without additional text.')
        return [{'role': 'user', 'content': '\n'.join(lines)}]


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


class CoRAL:
    def __init__(self, config, ratings, user_vectors, item_vectors):
        self.config, self.ratings = config, ratings
        self.users, self.items = sorted(user_vectors), sorted(item_vectors)
        if not self.users or set(self.items) != set(ratings.items) or not {row['user_id'] for row in ratings.records} <= set(self.users):
            raise ValueError('external embeddings do not cover required users and exact catalog')
        self.user_index = {value: index for index, value in enumerate(self.users)}
        self.item_index = {value: index for index, value in enumerate(self.items)}
        self.user_vectors = torch.tensor([user_vectors[user] for user in self.users], dtype=torch.float32)
        self.item_vectors = torch.tensor([item_vectors[item] for item in self.items], dtype=torch.float32)
        for value in (self.user_vectors, self.item_vectors):
            if value.ndim != 2 or value.shape[1] != config.dimension or not torch.isfinite(value).all():
                raise ValueError('external embeddings must be finite with the declared dimension')
        torch.manual_seed(config.seed)
        width = 2 * config.dimension
        self.actor = mlp(width, config.hidden, width)
        self.critic = mlp(2 * width, config.hidden, 1)
        self.transition = mlp(2 * width, config.hidden, width)
        self.actor_target, self.critic_target, self.transition_target = (copy.deepcopy(model).requires_grad_(False) for model in (self.actor, self.critic, self.transition))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.critic.parameters()) + list(self.transition.parameters()), lr=config.critic_lr)
        self.random = random.Random(config.seed)
        self.noise_random = np.random.RandomState(config.seed)
        self.replay, self.updates, self.episodes, self.environment_steps = [], 0, 0, 0

    def state(self, user, item, history, target=False):
        state = torch.cat((self.user_vectors[self.user_index[user]], self.item_vectors[self.item_index[item]]))
        transition = self.transition_target if target else self.transition
        for found_user, found_item in history:
            pair = torch.cat((self.user_vectors[self.user_index[found_user]], self.item_vectors[self.item_index[found_item]]))
            state = transition(torch.cat((state, pair)))
        return state

    def nearest(self, action, matrix, query_user):
        users = sorted({user for user, _ in matrix if user != query_user})
        items = sorted({item for _, item in matrix})
        if not users or not items:
            raise ValueError('no strictly earlier collaborative rating evidence is available')
        dimension = self.config.dimension
        left = self.user_vectors[[self.user_index[user] for user in users]]
        right = self.item_vectors[[self.item_index[item] for item in items]]
        return (users[int(((left - action[:dimension]) ** 2).sum(-1).argmin())],
                items[int(((right - action[dimension:]) ** 2).sum(-1).argmin())])

    def probability(self, score, user, item, matrix, history):
        value = score(self.ratings.prompt(user, item, matrix, [pair[0] for pair in history], [pair[1] for pair in history]))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Yes probability must be finite and in [0,1]')
        return float(value)

    def update(self):
        cfg = self.config
        if len(self.replay) < cfg.batch_size or self.environment_steps < cfg.learning_starts or self.updates >= cfg.max_updates:
            return None
        batch = self.random.sample(self.replay, cfg.batch_size)
        states = torch.stack([self.state(row['user'], row['item'], row['history']) for row in batch])
        actions = torch.tensor([row['action'] for row in batch], dtype=torch.float32)
        with torch.no_grad():
            next_states = torch.stack([self.state(row['user'], row['item'], row['next_history'], target=True) for row in batch])
            target_actions = self.actor_target(next_states)
            bootstrap = self.critic_target(torch.cat((next_states, target_actions), -1)).squeeze(-1)
            labels = torch.tensor([row['reward'] for row in batch]) + cfg.gamma * torch.tensor([0.0 if row['done'] else 1.0 for row in batch]) * bootstrap
        predicted = self.critic(torch.cat((states, actions), -1)).squeeze(-1)
        critic_loss = (predicted - labels).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.check_finite(critic_loss, [self.critic, self.transition])
        self.critic_optimizer.step()
        self.critic.requires_grad_(False)
        try:
            states = torch.stack([self.state(row['user'], row['item'], row['history']).detach() for row in batch])
            actor_loss = -self.critic(torch.cat((states, self.actor(states)), -1)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.check_finite(actor_loss, [self.actor])
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)
        self.check_tensors_finite(self.actor_optimizer.state_dict())
        self.check_tensors_finite(self.critic_optimizer.state_dict())
        with torch.no_grad():
            for live, target in ((self.actor, self.actor_target), (self.critic, self.critic_target), (self.transition, self.transition_target)):
                for parameter, target_parameter in zip(live.parameters(), target.parameters()):
                    if not torch.isfinite(parameter).all():
                        raise ValueError('nonfinite updated CoRAL parameters')
                    target_parameter.mul_(1 - cfg.tau).add_(parameter, alpha=cfg.tau)
        self.updates += 1
        return {'critic_loss': float(critic_loss.detach()), 'actor_loss': float(actor_loss.detach())}

    @staticmethod
    def check_finite(loss, modules):
        if not torch.isfinite(loss):
            raise ValueError('nonfinite CoRAL optimization objective')
        for module in modules:
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in module.parameters()):
                raise ValueError('nonfinite CoRAL optimization gradient')

    def episode(self, user, item, moment, score, label=None, train=False):
        if user not in self.user_index or item not in self.item_index:
            raise ValueError('query user/item missing from external embeddings')
        if type(moment) is not int or moment < 0 or train and label not in (0, 1):
            raise ValueError('episode timestamp or explicit binary training label is invalid')
        matrix = self.ratings.before(user, item, moment)
        history, trace = [], []
        noise = np.zeros(2 * self.config.dimension, dtype=np.float32)
        previous = self.probability(score, user, item, matrix, history)
        steps = self.config.train_steps if train else self.config.eval_steps
        if train and abs(previous - label) < self.config.early_stop_tolerance:
            self.episodes += 1
            return previous, trace
        for step in range(steps):
            with torch.no_grad():
                state = self.state(user, item, history, target=train)
                action = (self.actor_target if train else self.actor)(state)
                if train:
                    cfg = self.config
                    noise += -cfg.ou_theta * noise * cfg.ou_dt + cfg.ou_sigma * math.sqrt(cfg.ou_dt) * self.noise_random.normal(size=noise.shape)
                    action = action + torch.tensor(noise, dtype=torch.float32)
                if not torch.isfinite(action).all():
                    raise ValueError('nonfinite retrieval action')
                pair = self.nearest(action, matrix, user)
            next_history = history + [pair]
            probability = self.probability(score, user, item, matrix, next_history)
            reward = abs(previous - label) - abs(probability - label) if train else None
            done = step + 1 == steps or train and abs(probability - label) < self.config.early_stop_tolerance
            updates = []
            if train:
                self.replay.append({'user': user, 'item': item, 'history': history, 'action': action.tolist(),
                                    'reward': reward, 'next_history': next_history, 'done': bool(done)})
                self.replay = self.replay[-self.config.replay_capacity:]
                self.environment_steps += 1
                for _ in range(self.config.updates_per_step):
                    value = self.update()
                    if value is not None:
                        updates.append(value)
            trace.append({'pair': pair, 'probability': probability, 'reward': reward, 'updates': updates})
            history, previous = next_history, probability
            if done:
                break
        if train:
            self.episodes += 1
        return previous, trace

    def snapshot(self):
        names = ('actor', 'critic', 'transition', 'actor_target', 'critic_target', 'transition_target', 'actor_optimizer', 'critic_optimizer')
        state = {'config': asdict(self.config), 'models': {name: getattr(self, name).state_dict() for name in names},
                              'replay': self.replay, 'updates': self.updates, 'episodes': self.episodes, 'environment_steps': self.environment_steps,
                              'python_rng': self.random.getstate(), 'noise_rng': self.noise_random.get_state(),
                              'torch_rng': torch.get_rng_state()}
        self.check_tensors_finite(state['models'])
        return copy.deepcopy(state)

    @staticmethod
    def check_tensors_finite(value):
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError('nonfinite CoRAL checkpoint or optimizer tensor')
        if isinstance(value, dict):
            for child in value.values():
                CoRAL.check_tensors_finite(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                CoRAL.check_tensors_finite(child)

    def restore(self, state):
        if state.get('config') != asdict(self.config):
            raise ValueError('CoRAL checkpoint configuration differs')
        expected = {'actor', 'critic', 'transition', 'actor_target', 'critic_target', 'transition_target', 'actor_optimizer', 'critic_optimizer'}
        if set(state.get('models', {})) != expected:
            raise ValueError('checkpoint model components are incomplete')
        self.check_tensors_finite(state['models'])
        for name, values in state['models'].items():
            if name not in {'actor', 'critic', 'transition', 'actor_target', 'critic_target', 'transition_target', 'actor_optimizer', 'critic_optimizer'}:
                raise ValueError('unknown checkpoint model component')
            getattr(self, name).load_state_dict(values)
        self.replay = copy.deepcopy(state['replay'])
        if len(self.replay) > self.config.replay_capacity:
            raise ValueError('checkpoint replay capacity exceeded')
        for row in self.replay:
            if not math.isfinite(row['reward']) or len(row['action']) != 2 * self.config.dimension or any(not math.isfinite(value) for value in row['action']) or type(row['done']) is not bool:
                raise ValueError('checkpoint replay contains invalid scalars')
            if row['user'] not in self.user_index or row['item'] not in self.item_index or any(user not in self.user_index or item not in self.item_index for user, item in row['history'] + row['next_history']):
                raise ValueError('checkpoint replay identities are invalid')
        if any(type(state[key]) is not int or state[key] < 0 for key in ('updates', 'episodes', 'environment_steps')):
            raise ValueError('checkpoint counters are invalid')
        self.updates, self.episodes = state['updates'], state['episodes']
        self.environment_steps = state['environment_steps']
        self.random.setstate(state['python_rng'])
        self.noise_random.set_state(state['noise_rng'])
        torch.set_rng_state(state['torch_rng'])
