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
The pid controller becomes unstable, and we end up overshooting and oscillating around the waypoints.

### 2. How does $K_D$ mitigate the effect you saw above when increasing $K_P$?
When a high Kp causes a fast approach towards the setpoint,
the derivative of the error will also be very large, counteracting the proportional 
component of the signal and preventing overshoot / dampening oscillation

### 3. In what scenarios is a non-zero $K_I$ needed for the controller to perform well?
If there is a constant force applied to the arm, like gravity, there would be a constant need for a proportional correction. 
A KI term can solve this by accumulating the error caused by gravity over time, taking the load off the proportional term.

## Ex3 (Bonus)
What difference can you observe when the robot is tracking the keypoints on the Lemniscate curve? To improve the performance of the RL policy, what changes can you make in the functions in ex3? Modify these functions (you can also change their arguments, and make corresponding changes in `env/so100_tracking_env.py`). Train another RL policy with your new environments and show the performance in the video, and explain how your changes impact the robot's performance. You can also make changes to the PPO hyperparameters (gamma, ent_coef, etc.).

The RL policy is noticably more random than the IK solver, it doesnt follow the shortest 
path to the next marker, often moving away on accident. 
I beleive that an easy improvement would be to punish velocity and 
acceleration of the end effector in the reward function, 
encouraging the policy to take more direct and smooth paths to its targets.

After training policies using both reward functions, the more complex reward 
function generates both lower tracking error, and smoother trajectories.


# Video Plan
1. robot tracking generated keypoints (inverse_kinematics.py)
2. answers to theoretical questions overlaying the video

3. robot tracking generated keypoints with pid (pid_control.py)
4. answers to theoretical questions overlaying the video

5. robot tracking random keypoints (evaluate_rand_targets.py) 30s
6. error printout of base model

7. robot tracking random keypoints with improved reward function
8. error printout

9. show pid_control version side by side as well
10. both policies doing trajectory tracking (side by side)
11. overlay bonus question answer
