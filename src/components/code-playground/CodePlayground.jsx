// src/components/code-playground/CodePlayground.jsx
// Interactive code playground for ROS 2 examples
import React, { useState, useEffect } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import styles from './CodePlayground.module.css';

const CodePlayground = ({ initialCode, language = 'python', title = 'ROS 2 Code Playground', description = 'Run and interact with ROS 2 code examples' }) => {
  const [code, setCode] = useState(initialCode || `import rclpy
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
    main()`);

  const [output, setOutput] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [executionTime, setExecutionTime] = useState(0);

  const handleRunCode = () => {
    setIsRunning(true);
    setOutput([]);

    // Simulate code execution
    const startTime = Date.now();

    // Add simulated output to demonstrate the concept
    setTimeout(() => {
      const simulatedOutput = [
        'Initializing ROS 2 publisher node...',
        'Node initialized successfully',
        'Publishing: "Hello World: 0"',
        'Publishing: "Hello World: 1"',
        'Publishing: "Hello World: 2"',
        'Node spinning - waiting for messages...',
        `Execution completed in ${Date.now() - startTime}ms`
      ];

      setOutput(simulatedOutput);
      setExecutionTime(Date.now() - startTime);
      setIsRunning(false);
    }, 1500);
  };

  const handleResetCode = () => {
    setCode(initialCode || `import rclpy
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
    main()`);
    setOutput([]);
  };

  return (
    <div className={styles.codePlayground}>
      <div className={styles.codePlaygroundHeader}>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>

      <div className={styles.codeEditor}>
        <div className={styles.editorHeader}>
          <span className={styles.languageIndicator}>{language}</span>
          <div className={styles.editorControls}>
            <button
              className={styles.runButton}
              onClick={handleRunCode}
              disabled={isRunning}
            >
              {isRunning ? 'Running...' : '▶ Run Code'}
            </button>
            <button
              className={styles.resetButton}
              onClick={handleResetCode}
            >
              ⟳ Reset
            </button>
          </div>
        </div>
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          className={styles.codeTextarea}
          rows="15"
        />
      </div>

      {output.length > 0 && (
        <div className={styles.terminalOutput}>
          <div className={styles.terminalHeader}>
            <span>Output</span>
            <span>Execution time: {executionTime}ms</span>
          </div>
          <div className={styles.terminalContent}>
            {output.map((line, index) => (
              <div key={index} className={styles.terminalLine}>
                {line}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className={styles.playgroundInfo}>
        <p><strong>Note:</strong> This is a simulation of ROS 2 code execution. In a real environment, this code would run on your ROS 2 installation.</p>
        <p>To run this code in your local environment:</p>
        <ol>
          <li>Create a new ROS 2 package: <code className={styles.inlineCode}>ros2 pkg create --build-type ament_python my_publisher</code></li>
          <li>Copy this code to <code className={styles.inlineCode}>my_publisher/my_publisher/publisher_member_function.py</code></li>
          <li>Build and run: <code className={styles.inlineCode}>ros2 run my_publisher publisher_member_function</code></li>
        </ol>
      </div>
    </div>
  );
};

// Wrapper component to handle client-side rendering
const CodePlaygroundWrapper = (props) => {
  return (
    <BrowserOnly>
      {() => <CodePlayground {...props} />}
    </BrowserOnly>
  );
};

export default CodePlaygroundWrapper;