# Interactive Code Playgrounds

This section provides interactive code playgrounds for ROS 2 examples. These playgrounds allow you to experiment with ROS 2 code directly in your browser without needing to set up a local development environment.

## Publisher Example

The following playground demonstrates a simple ROS 2 publisher node that publishes "Hello World" messages to a topic:

import CodePlayground from '@site/src/components/code-playground/CodePlayground';

<CodePlayground
  initialCode={`import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()`}
  language="python"
  title="ROS 2 Publisher Example"
  description="A simple ROS 2 publisher that sends messages to a topic"
/>

## Subscriber Example

The following playground demonstrates a ROS 2 subscriber node that listens to messages from a topic:

import ROS2SubscriberExample from '@site/src/components/code-playground/ROS2SubscriberExample';

<ROS2SubscriberExample />

## Service Example

The following playground demonstrates a ROS 2 service that adds two integers:

import ROS2ServiceExample from '@site/src/components/code-playground/ROS2ServiceExample';

<ROS2ServiceExample />

## How to Use the Code Playgrounds

1. **Edit the Code**: Modify the code in the editor to experiment with different ROS 2 concepts
2. **Run the Code**: Click the "Run Code" button to see simulated output
3. **Reset**: Use the "Reset" button to restore the original example code
4. **Local Execution**: Follow the instructions in the "Info" section to run the code in your local ROS 2 environment

## Available Examples

The playgrounds include examples for:

- Basic publisher/subscriber patterns
- Service/client communication
- Parameter management
- Action servers and clients (coming soon)
- TF2 transformations (coming soon)
- Navigation and path planning (coming soon)

## Note on Simulation

These playgrounds simulate ROS 2 behavior in the browser. While the code shown is valid ROS 2 Python code, the actual execution is simulated. To run the code in a real ROS 2 environment, follow the instructions provided below each playground.

For complete functionality and real ROS 2 execution, you'll need to set up a local ROS 2 development environment as described in Chapter 2 of this book.