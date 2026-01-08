"""Train the core policy on application-prepared, temporally aligned NPZ windows."""
import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from mop.policy import MOPPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', default='configs/mop.yaml')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='checkpoints/mop.pt')
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError('Training requires a positive epoch count.')
    with open(args.config, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    with np.load(args.data, allow_pickle=False) as data:
        arrays = {key: torch.from_numpy(data[key].copy()) for key in data.files}
    required = {'rgb', 'point_cloud', 'agent_pos', 'action', 'object_position'}
    if not required.issubset(arrays):
        raise ValueError(f'Missing fields: {required - arrays.keys()}')
    count = len(arrays['action'])
    if count == 0 or any(len(value) != count for value in arrays.values()):
        raise ValueError('All fields must contain aligned, nonempty training windows.')
    for key in required - {'rgb'}:
        arrays[key] = arrays[key].float()
    normalizer = LinearNormalizer()
    statistics = {key: arrays[key] for key in ('point_cloud', 'agent_pos', 'action')}
    if 'action_valid' in arrays:
        statistics['action'] = arrays['action'][arrays['action_valid'].bool()]
    if any(value.numel() == 0 or not torch.isfinite(value).all() for value in statistics.values()):
        raise ValueError('Normalization requires finite training observations and valid actions.')
    normalizer.fit(statistics)
    dimensions = {'state_dim': arrays['agent_pos'].shape[-1], 'action_dim': arrays['action'].shape[-1]}
    policy = MOPPolicy(**dimensions, **config['policy']).to(args.device)
    policy.set_normalizer(normalizer)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config['training']['learning_rate'])
    fields = list(arrays)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(arrays[k] for k in fields)),
                                         batch_size=config['training']['batch_size'], shuffle=True)
    policy.train()
    for epoch in range(args.epochs):
        for values in loader:
            data = {key: value.to(args.device) for key, value in zip(fields, values)}
            batch = {'obs': {key: data[key] for key in ('rgb', 'point_cloud', 'agent_pos')},
                     'action': data['action'], 'object_position': data['object_position']}
            batch.update({key: data[key] for key in ('object_valid', 'action_valid') if key in data})
            loss, metrics = policy.compute_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f'epoch={epoch + 1} loss={float(loss.detach()):.6f} {metrics}')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': policy.state_dict(), 'policy_config': config['policy'], **dimensions}, output)


if __name__ == '__main__':
    main()
