// src/components/code-playground/ROS2ServiceExample.jsx
// Interactive code playground for ROS 2 service example
import React from 'react';
import CodePlayground from './CodePlayground';

const ROS2ServiceExample = () => {
  const serviceCode = `import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\\na: %d b: %d' % (request.a, request.b))
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()`;

  return (
    <CodePlayground
      initialCode={serviceCode}
      language="python"
      title="ROS 2 Service Example"
      description="A simple ROS 2 service that adds two integers"
    />
  );
};

export default ROS2ServiceExample;