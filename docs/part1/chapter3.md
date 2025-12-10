---
sidebar_position: 4
---

# Chapter 3: URDF and Robot Modeling

## Overview

Unified Robot Description Format (URDF) is the standard for representing robot models in ROS. This chapter covers how to model humanoid robots using URDF, including kinematics, dynamics, and visual properties.

## What is URDF?

URDF (Unified Robot Description Format) is an XML format for representing a robot model. It describes:

- Kinematic and dynamic structure (links and joints)
- Visual and collision properties
- Inertial properties
- Sensor and actuator locations
- Materials and colors

## URDF Structure for Humanoid Robots

A humanoid robot URDF typically includes:

- **Links**: Rigid bodies (torso, head, arms, legs, feet)
- **Joints**: Connections between links (revolute, prismatic, fixed)
- **Materials**: Visual appearance properties
- **Gazebo plugins**: Simulation-specific properties

## Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Links -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.2 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.2 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints -->
  <joint name="joint_name" type="revolute">
    <parent link="base_link"/>
    <child link="child_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>
</robot>
```

## Links in Humanoid Robots

### Link Components

Each link contains three main elements:

1. **Visual**: How the link appears in simulation and visualization
2. **Collision**: Shape used for collision detection
3. **Inertial**: Physical properties for dynamics simulation

### Visual Element

```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <!-- Options: box, cylinder, sphere, mesh -->
    <mesh filename="package://robot_description/meshes/link_name.dae"/>
  </geometry>
  <material name="red">
    <color rgba="1 0 0 1"/>
  </material>
</visual>
```

### Collision Element

```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <cylinder radius="0.05" length="0.3"/>
  </geometry>
</collision>
```

### Inertial Element

```xml
<inertial>
  <mass value="1.0"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
</inertial>
```

## Joints in Humanoid Robots

### Joint Types

- **Fixed**: No movement (e.g., sensor mounting)
- **Revolute**: Rotational movement around an axis
- **Continuous**: Like revolute but unlimited rotation
- **Prismatic**: Linear sliding movement
- **Floating**: 6 DOF movement (rarely used)

### Joint Limits

```xml
<limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
<safety_controller k_position="20.0" k_velocity="400.0" soft_lower_limit="-1.5" soft_upper_limit="1.5"/>
```

## Humanoid Robot Kinematic Structure

### Typical Humanoid Structure

```
base_link
├── torso
│   ├── head
│   ├── left_arm
│   │   ├── left_forearm
│   │   └── left_hand
│   ├── right_arm
│   │   ├── right_forearm
│   │   └── right_hand
│   ├── left_leg
│   │   ├── left_shin
│   │   └── left_foot
│   └── right_leg
│       ├── right_shin
│       └── right_foot
```

### Joint Configuration

Humanoid robots typically have 20-40+ degrees of freedom:

- **Head**: 3 DOF (yaw, pitch, roll)
- **Arms**: 6-7 DOF each (shoulder: 3 DOF, elbow: 1 DOF, wrist: 2-3 DOF)
- **Legs**: 6 DOF each (hip: 3 DOF, knee: 1 DOF, ankle: 2 DOF)

## Creating a Simple Humanoid URDF

Here's a minimal humanoid robot example:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="2"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="-0.05 0.15 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="3"/>
  </joint>
</robot>
```

## Advanced URDF Features

### Transmission Elements

Define how joints connect to actuators:

```xml
<transmission name="left_shoulder_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_shoulder_joint">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_shoulder_motor">
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Gazebo Integration

Add simulation-specific properties:

```xml
<gazebo reference="torso">
  <material>Gazebo/Grey</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
</gazebo>
```

## URDF Validation

### Using check_urdf

Validate your URDF file:

```bash
check_urdf /path/to/robot.urdf
```

### Using xacro

For complex robots, use Xacro (XML Macros) to avoid repetition:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot">
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <xacro:macro name="simple_arm" params="prefix">
    <link name="${prefix}_upper_arm">
      <visual>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
      </visual>
    </link>
  </xacro:macro>

  <xacro:simple_arm prefix="left"/>
  <xacro:simple_arm prefix="right"/>
</robot>
```

## Isaac Sim Integration

Isaac Sim can import URDF files with some considerations:

- Use supported geometry types (meshes should be in supported formats)
- Ensure proper scaling
- Verify joint limits and types
- Check material definitions

## Best Practices

1. **Start Simple**: Begin with a basic skeleton and add complexity gradually
2. **Validate Regularly**: Use `check_urdf` frequently
3. **Use Xacro**: For complex robots with repeated elements
4. **Proper Scaling**: Ensure units are consistent (typically meters for length)
5. **Realistic Inertias**: Use proper mass properties for simulation stability
6. **Joint Limits**: Define realistic limits based on physical constraints

## Chapter Summary

This chapter covered URDF fundamentals for modeling humanoid robots, including link and joint definitions, visual and collision properties, and best practices. We also touched on advanced features like transmissions and Gazebo integration. In the next part, we'll explore simulation environments starting with Isaac Sim fundamentals.