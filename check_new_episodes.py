#!/usr/bin/env python3
"""Script to check which episodes are new in the dataset."""

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

# Load dataset metadata
repo_id = "sourenp/so101_nod_act_v2"
meta = LeRobotDatasetMetadata(repo_id)

print(f"Current total episodes: {meta.total_episodes}")
print(f"Episodes are indexed from 0 to {meta.total_episodes - 1}")
print("\nTo train only on new episodes, you need to know:")
print("1. How many episodes were in the dataset BEFORE you recorded new ones")
print("2. The new episodes will be from that number onwards")
print("\nFor example, if you had 10 episodes before and now have 15:")
print("New episodes would be: [10, 11, 12, 13, 14]")
print("\nUse --dataset.episodes=[10,11,12,13,14] in your training command")
