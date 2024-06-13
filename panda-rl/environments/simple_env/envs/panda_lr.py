import os
import random
import math
import numpy as np

import gym
from gym import spaces

import pybullet as p
import pybullet_data
from stable_baselines3 import PPO

class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, observation_dim=8, action_dim=4, max_episode_len=2000, use_gui=True):
        # Connect the pybullet physics engine(server) with GUI mode or DIRECT mode
        if use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # Set the initial position and orientation of the view in GUI
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,     # distance from eye to camera target position
            cameraYaw=0,            # camera yaw angle (in degrees) left/right
            cameraPitch=-40,        # camera pitch angle (in degrees) up/down
            cameraTargetPosition=[0.65, 0, 0]  # the camera focus point
        )

        # Define the observation space in ranging -1 to 1
        # observation: Cartesian position of the gripper (3) + two fingers joint positions (2) + object position (3)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(observation_dim,), dtype=np.float32)

        # Define the action space in ranging -1 to 1
        # action: Cartesian position of the gripper (3) + a joint variable for both fingers (1). 
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # Initialize step length
        self.step_counter = 0
        self.max_episode_len = max_episode_len


    def reset(self):
        # Path configuration
        urdf_path = pybullet_data.getDataPath()
        plane_path = os.path.join(urdf_path, "plane.urdf")
        table_path = os.path.join(urdf_path, "table/table.urdf")
        tray_path = os.path.join(urdf_path, "tray/tray.urdf")
        object_path = os.path.join(urdf_path, "random_urdfs/000/000.urdf")
        robot_path = os.path.join(urdf_path, "franka_panda/panda.urdf")
        humanoid_path = os.path.join(urdf_path, "humanoid/humanoid.urdf")
        
        # Initial states of robot and objects
        self.step_counter = 0
        robot_joint_position = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        object_position = [0.5, 0, 0.05]  # Object outside the tray

        # Reset simulation and environment
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # Stop rendering
        p.setGravity(0, 0, -10)

        # Load objects into the simulation
        plane_id = p.loadURDF(plane_path, basePosition=[0, 0, -0.3])
        table_id = p.loadURDF(table_path, basePosition=[0.5, 0, -0.65])
        tray_id = p.loadURDF(tray_path, basePosition=[1.0, 0, 0])  # Move tray further to the right
        self.object_id = p.loadURDF(object_path, basePosition=object_position)
        self.panda_id = p.loadURDF(robot_path, useFixedBase=True)

        # Load and configure the humanoid
        humanoid_id = p.loadURDF(humanoid_path, basePosition=[0.3, -0.5, 0], globalScaling=0.25, useFixedBase=True)
        p.resetBasePositionAndOrientation(humanoid_id, [0.3, -0.5, 0.5], p.getQuaternionFromEuler([math.pi / 2, 0, 0]))

        # Reset robot joint states
        for i in range(7):
            p.resetJointState(self.panda_id, i, robot_joint_position[i])
        p.resetJointState(self.panda_id, 9, 0.08)
        p.resetJointState(self.panda_id, 10, 0.08)

        # Get observation info
        robot_gripper_state = p.getLinkState(self.panda_id, 11)[0]
        robot_fingers_state = (p.getJointState(self.panda_id, 9)[0], p.getJointState(self.panda_id, 10)[0])
        object_state = p.getBasePositionAndOrientation(self.object_id)[0]
        initial_state = robot_gripper_state + robot_fingers_state + object_state
        observation = np.array(initial_state, dtype=np.float32)

        # Start rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return observation


    def step(self, action):
        # Set single-step rendering for immediate feedback
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

        # Get the current position of the gripper
        current_pose = p.getLinkState(self.panda_id, 11)
        current_position = current_pose[0]

        # Set the goal position of the gripper
        # gradually move the gripper toward the desired Cartesian position using dv
        dv = 0.2
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        finger_position = (action[3] +1) /2
        goal_position = [current_position[0] + dx,
                         current_position[1] + dy,
                         current_position[2] + dz]
        # the gripper orientation is considered to be perpendicular to the ground.
        goal_orientation = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])

        # Calculate inverse kinematics for the goal position
        ik_joint_states = p.calculateInverseKinematics(
            bodyUniqueId=self.panda_id,
            endEffectorLinkIndex=11,
            targetPosition=goal_position,
            targetOrientation=goal_orientation)
        goal_arm_joint_states = ik_joint_states[:7]
        goal_states = list(goal_arm_joint_states) + [finger_position, finger_position]

        # Apply the calculated joint positions and finger position to the robot
        joints_to_be_controlled = list(range(7)) + [9, 10]
        p.setJointMotorControlArray(
            bodyUniqueId=self.panda_id,
            jointIndices=joints_to_be_controlled,
            controlMode=p.POSITION_CONTROL,
            targetPositions=goal_states
        )

        # Step the simulation
        p.stepSimulation()

        # Get the state of the object and the robot
        object_state = p.getBasePositionAndOrientation(self.object_id)[0]
        robot_gripper_state = p.getLinkState(self.panda_id, 11)[0]
        robot_fingers_state = (p.getJointState(self.panda_id, 9)[0], p.getJointState(self.panda_id, 10)[0])

        # Calculate the Euclidean distance between the gripper and the object
        distance = np.linalg.norm(np.array(robot_gripper_state) - np.array(object_state[:3]))

        # Determine 'reward' and whether the task is 'done'
        reward = -distance

        # 객체를 잡으면 더 큰 보상을 줌
        if distance < 0.05 and abs(robot_fingers_state[0] - robot_fingers_state[1]) < 0.01:
            reward += 8  # 보상 추가
            if object_state[2] > 0.1:  # 객체가 들어올려진 경우
                reward += 13  # 보상 추가
                if object_state[0] > 0.95 and object_state[0] < 1.05 and object_state[2] > 0.05:  # Object moved inside the tray
                    reward += 20  # 보상 추가
                    done = True
                else:
                    done = False
            else:
                done = False
        else:
            done = False

        self.step_counter += 1
        if self.step_counter > self.max_episode_len:
            reward = 0
            done = True

        # Set 'info'
        info = {'object_position': object_state}

        # Get the observation
        current_state = robot_gripper_state + robot_fingers_state + object_state
        observation = np.array(current_state).astype(np.float32)

        return observation, reward, done, info


    def render(self, mode='human'):
        # Configure the camera view using yaw, pitch, and roll angles
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.7, 0.0, 0.05],
            distance=.7,
            yaw=90,
            pitch=-70,
            roll=0,
            upAxisIndex=2
        )

        # Configure the projection matrix with field of view and aspect ratio
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(960) / 720,
            nearVal=0.1,
            farVal=100.0
        )

        # Capture an image from the current view
        width, height, rgb_pixels, depth_pixels, _ = p.getCameraImage(
            width=960,
            height=720,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Convert pixel data to a NumPy array for image manipulation
        rgb_array = np.array(rgb_pixels, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))
        rgb_array = rgb_array[:, :, :3]

        # Convert depth data to a NumPy array and normalize it for visualization
        depth_array = np.array(depth_pixels, dtype=np.float32)
        depth_array = np.reshape(depth_array, (height, width))
        depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())  # Normalize the depth values

        if mode == 'human':
            return None
        elif mode == 'rgb_array':
            return rgb_array
        elif mode == 'depth_array':
            # Normalize the depth data for visualization
            depth_norm = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
            depth_image = (depth_norm * 255).astype(np.uint8)
            return depth_image
        else:
            raise ValueError("Unsupported mode. Supported modes are 'human', 'rgb_array', and 'depth_array'.")


    def close(self):
        if p.isConnected():
            p.disconnect()

    def test(self, model_path, num_episodes=10):
        # Load the trained model
        model = PPO.load(model_path)

        for episode in range(num_episodes):
            observation = self.reset()
            done = False
            total_reward = 0

            while not done:
                # Use the model to predict the action
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, done, info = self.step(action)
                total_reward += reward

                # Render the environment
                self.render()

            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

        self.close()


if __name__ == "__main__":
    model_path = "/home/wilterovago/work/panda-rl/agents/models/PPO_PandaEnv/PPO_PandaEnv_13_JS.zip"
    env = PandaEnv(use_gui=True)
    env.test(model_path)
