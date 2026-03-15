python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr         --state-keys state_ee_full state_gripper "state_cube[:3]" "state_joints[:5]" state_obstacle --action-keys action_ee_xyz action_gripper --epochs 200 --depth 4 --d_model 450 --chunk-size 10 --policy obstacle

