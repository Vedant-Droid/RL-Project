#Launching Gazebo
source /opt/ros/jazzy/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

#Training
cd ~/Turtle
source ~/rl_venv/bin/activate
python3 train_ppo.py --world <name> --timesteps 200000

#Testing
python3 run_policy.py --model ppo_tb3_goal.zip --episodes 10 --world <name>
