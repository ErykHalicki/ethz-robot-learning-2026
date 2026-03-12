python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys state_ee_xyz state_gripper "state_cube[:3]" \
        --action-keys action_ee_xyz action_gripper

