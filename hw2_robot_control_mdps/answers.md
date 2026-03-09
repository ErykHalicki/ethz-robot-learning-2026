# Theoretical Question Answers
## Ex1
### 1. If you increase the width of the Lemniscate (increasing a), what issue can happen with the robot performing IK?
If the lemniscate becomes too wide, there could be points on it that are unreachable. This would cause the ik solver to never converge.

### 2. What can happen if you change the dt parameter in IK?
If you reduce the dt parameter, the steps will be become smaller, requiring more iterations to converge, but allowing for stability. Conversely, if you increase dt steps will become big, potentially requiring less steps to converge, but also potentially being more unstable.

### 3. We implemented a simple numerical IK solver. What are the advantages and disadvantages compared to an analytical IK solver?
Some advantages of numerical IK solvers: 
They still work for underconstrained systems, like arms that are greater than 6dof
They are easier to develop since they dont require deriving complex trig equations

Disadvantages:
In over or well constrained systems (like 6dof or less arms) they take a variable amount of iterations to converge, whereas analytical solvers are practically instant
They can fail to converge even if a solution exists

### 4. What are the limits of our IK solver compared to state-of-the-art IK solvers?
Our IK solver doesnt take into account rotational error.
It also only provides once solution, and doesnt take into account any collision checking
Our IK solver also doesnt support any secondary objectives like minimizing joint differences or shortest path planning

## Ex2

### 1. If you keep increasing $K_P$, what issue arises when tracking the waypoints?
The pid controller becomes unstable, and we end up overshooting the waypoints.

### 2. How does $K_D$ mitigate the effect you saw above when increasing $K_P$?
When a high Kp causes a fast approach towards the setpoint,
the derivative of the error will also be very large, counteracting the proportional component of the signal and preventing overshoot

### 3. In what scenarios is a non-zero $K_I$ needed for the controller to perform well?
If there is a constant force applied to the arm, like gravity, there would be a constant need for a proportional correction. 
A KI term can solve this by accumulating the error caused by gravity over time.

## Ex3 (Bonus)
What difference can you observe when the robot is tracking the keypoints on the Lemniscate curve? To improve the performance of the RL policy, what changes can you make in the functions in ex3? Modify these functions (you can also change their arguments, and make corresponding changes in `env/so100_tracking_env.py`). Train another RL policy with your new environments and show the performance in the video, and explain how your changes impact the robot's performance. You can also make changes to the PPO hyperparameters (gamma, ent_coef, etc.).

The RL policy is noticably more random, it doesnt follow the shortest path to the next marker, often moving away on accident. 
I beleive that an easy improvement to the RL policy would be introducing a relative velocity penalty to the reward function (and increasing the magnitude of the sparse reward?)
To do this, I modified the reward function in exercise 3 to compute the difference between the current and last tracking error, reducing the reward if the end effector moves further away from the target
As we can see, the policy with the new reward function tracks points more smoothly than one that only takes current error into account. 
