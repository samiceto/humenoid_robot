#!/usr/bin/env python3
"""
Performance Optimizer for Voice Command Pipeline

This script implements performance optimization techniques to achieve ≥15 Hz
real-time inference for the voice command pipeline in the Physical AI & Humanoid Robotics course.
"""

import os
import sys
import time
import threading
import queue
import statistics
import gc
from collections import deque
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import whisper
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print(f"Required packages not available: {e}")
    print("Install with: pip install torch whisper transformers")
    sys.exit(1)

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import Twist
except ImportError:
    print("ROS 2 not available. This script can run without ROS 2 for performance testing.")
    rclpy = None


class PerformanceOptimizer:
    """
    Implements performance optimization techniques for the voice command pipeline
    """

    def __init__(self):
        self.optimization_level = 2  # 0: None, 1: Light, 2: Heavy
        self.batch_size = 1
        self.use_fp16 = True
        self.max_queue_size = 10

        # Performance monitoring
        self.execution_times = deque(maxlen=100)
        self.throughput_history = deque(maxlen=50)

        # Memory management
        self.tensor_pools = {}
        self.cache = {}

        print("Performance optimizer initialized")

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply model optimization techniques"""
        if self.use_fp16:
            model = model.half()  # Convert to half precision

        model.eval()  # Set to evaluation mode

        # Apply optimizations based on level
        if self.optimization_level >= 1:
            model = torch.jit.optimize_for_inference(torch.jit.script(model))

        if self.optimization_level >= 2:
            # Additional heavy optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        return model

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply quantization to model for faster inference"""
        import torch.quantization as tq

        model.eval()

        # Define quantization configuration
        model.qconfig = tq.get_default_qconfig('qnnpack')
        model_prepared = tq.prepare(model, inplace=False)

        # For actual quantization, we'd need calibration data
        # For this example, we'll convert directly
        model_quantized = tq.convert(model_prepared, inplace=False)

        return model_quantized

    def create_tensor_pool(self, shape: tuple, dtype=torch.float32, pool_size: int = 10):
        """Create a pool of tensors to avoid allocation overhead"""
        pool = deque(maxlen=pool_size)

        for _ in range(pool_size):
            tensor = torch.empty(shape, dtype=dtype)
            pool.append(tensor)

        self.tensor_pools[shape] = pool
        return pool

    def get_tensor_from_pool(self, shape: tuple, dtype=torch.float32):
        """Get tensor from pool or create new one"""
        key = (shape, dtype)

        if key in self.tensor_pools and len(self.tensor_pools[key]) > 0:
            return self.tensor_pools[key].popleft()
        else:
            return torch.empty(shape, dtype=dtype)

    def return_tensor_to_pool(self, tensor, shape: tuple, dtype=torch.float32):
        """Return tensor to pool for reuse"""
        key = (shape, dtype)

        if key in self.tensor_pools:
            if len(self.tensor_pools[key]) < self.tensor_pools[key].maxlen:
                self.tensor_pools[key].append(tensor)

    def batch_process(self, inputs: List[Any], process_func: Callable) -> List[Any]:
        """Process inputs in batches for better throughput"""
        if self.batch_size <= 1:
            # No batching, process individually
            return [process_func(inp) for inp in inputs]

        results = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]

            # Pad batch if necessary
            if len(batch) < self.batch_size:
                batch.extend([None] * (self.batch_size - len(batch)))

            batch_results = process_func(batch)
            results.extend(batch_results[:len(inputs) - i])  # Remove padding

        return results

    def time_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Time execution of a function"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        self.execution_times.append(execution_time)

        return result, execution_time

    def get_current_throughput(self) -> float:
        """Calculate current throughput in Hz"""
        if len(self.execution_times) == 0:
            return 0.0

        avg_time = statistics.mean(list(self.execution_times)[-20:])  # Last 20 samples
        throughput = 1.0 / avg_time if avg_time > 0 else 0.0
        self.throughput_history.append(throughput)

        return throughput

    def is_meeting_requirements(self) -> bool:
        """Check if performance requirements are being met"""
        current_throughput = self.get_current_throughput()
        return current_throughput >= 15.0  # ≥15 Hz requirement

    def optimize_memory_usage(self):
        """Optimize memory usage"""
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        # Clear internal caches
        self.cache.clear()


class OptimizedWhisperWrapper:
    """
    Wrapper for Whisper model with performance optimizations
    """

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.model = None
        self.load_model()

    def load_model(self):
        """Load and optimize Whisper model"""
        print(f"Loading Whisper model: {self.model_size}")

        # Load model with optimizations
        self.model = whisper.load_model(
            self.model_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            download_root="./models"
        )

        # Apply optimizations
        if torch.cuda.is_available():
            self.model = self.model.half()  # Half precision

        self.model.eval()

        print("Whisper model loaded and optimized")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio with optimizations"""
        # Use optimized transcription settings
        result = self.model.transcribe(
            audio,
            fp16=True,
            without_timestamps=True,
            max_new_tokens=128,
            compression_ratio_threshold=2.0,
            logprob_threshold=-1.0
        )

        return result["text"]


class OptimizedLLMWrapper:
    """
    Wrapper for LLM with performance optimizations
    """

    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load and optimize LLM"""
        print(f"Loading LLM: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=True if torch.cuda.is_available() else False  # Use 8-bit if available
        )

        # Set to evaluation mode
        self.model.eval()

        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("LLM loaded and optimized")

    def generate(self, prompt: str, max_length: int = 64) -> str:
        """Generate text with optimizations"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        with torch.no_grad():  # Disable gradient computation
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                min_length=len(inputs[0]) + 10,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
                num_beams=1,  # Greedy decoding for speed
                no_repeat_ngram_size=2
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):]


class OptimizedVoicePipeline:
    """
    Optimized voice command pipeline with performance enhancements
    """

    def __init__(self):
        self.optimizer = PerformanceOptimizer()
        self.whisper_wrapper = OptimizedWhisperWrapper()
        self.llm_wrapper = OptimizedLLMWrapper()

        # ROS 2 components if available
        self.node = None
        self.publisher = None

        if rclpy:
            rclpy.init()
            self.node = Node('optimized_voice_pipeline')
            self.publisher = self.node.create_publisher(String, '/processed_command', 10)

        print("Optimized voice pipeline initialized")

    def process_audio_input(self, audio_data: np.ndarray) -> str:
        """Process audio input through optimized pipeline"""
        # Step 1: Speech recognition
        text, recognition_time = self.optimizer.time_function(
            self.whisper_wrapper.transcribe, audio_data
        )

        print(f"Speech recognition took: {recognition_time:.3f}s")

        # Step 2: Command interpretation with LLM
        command_prompt = f"Convert this natural language command to a structured format: {text}"
        interpreted_command, interpretation_time = self.optimizer.time_function(
            self.llm_wrapper.generate, command_prompt, max_length=32
        )

        print(f"Command interpretation took: {interpretation_time:.3f}s")

        # Calculate total processing time
        total_time = recognition_time + interpretation_time
        print(f"Total processing time: {total_time:.3f}s")

        # Check if we're meeting performance requirements
        current_throughput = self.optimizer.get_current_throughput()
        print(f"Current throughput: {current_throughput:.2f} Hz")

        if not self.optimizer.is_meeting_requirements():
            print("⚠️  Performance requirement not met! Consider adjusting optimization settings.")

        return interpreted_command

    def run_performance_test(self, duration: float = 60.0):
        """Run performance test for specified duration"""
        print(f"Running performance test for {duration} seconds...")

        start_time = time.time()
        command_count = 0
        successful_commands = 0

        # Simulate audio data (in real scenario, this would come from microphone)
        dummy_audio = np.random.rand(16000 * 5).astype(np.float32)  # 5 seconds of dummy audio

        while time.time() - start_time < duration:
            try:
                # Process dummy command
                result = self.process_audio_input(dummy_audio)

                command_count += 1
                successful_commands += 1

                # Brief pause to simulate realistic timing
                time.sleep(0.1)

            except Exception as e:
                print(f"Error processing command: {e}")
                command_count += 1
                continue

        # Calculate performance metrics
        actual_duration = time.time() - start_time
        avg_throughput = successful_commands / actual_duration

        print(f"\nPerformance Test Results:")
        print(f"Duration: {actual_duration:.2f}s")
        print(f"Commands processed: {successful_commands}/{command_count}")
        print(f"Success rate: {(successful_commands/command_count)*100:.1f}%")
        print(f"Average throughput: {avg_throughput:.2f} Hz")
        print(f"Required throughput: ≥15 Hz")

        # Check if requirements are met
        if avg_throughput >= 15.0:
            print("✅ Performance requirements met!")
            return True
        else:
            print("❌ Performance requirements not met!")
            return False

    def adaptive_optimization(self):
        """Adjust optimization settings based on performance"""
        current_throughput = self.optimizer.get_current_throughput()

        if current_throughput < 15.0:
            # Need to improve performance
            if self.optimizer.optimization_level < 2:
                print("Increasing optimization level to improve performance...")
                self.optimizer.optimization_level = min(2, self.optimizer.optimization_level + 1)

                # Reload models with higher optimization
                self.whisper_wrapper = OptimizedWhisperWrapper()
                self.llm_wrapper = OptimizedLLMWrapper()
            elif self.optimizer.batch_size < 4:
                # Increase batching
                self.optimizer.batch_size = min(4, self.optimizer.batch_size + 1)
                print(f"Increasing batch size to {self.optimizer.batch_size}")
            else:
                # Consider model quantization
                print("Consider quantizing models for better performance")
        elif current_throughput > 20.0:
            # Performance is exceeding requirements, could reduce optimization
            if self.optimizer.optimization_level > 1:
                print("Performance exceeding requirements, reducing optimization level...")
                self.optimizer.optimization_level = max(1, self.optimizer.optimization_level - 1)

    def run_continuous_optimization(self, test_duration: float = 10.0):
        """Run continuous optimization with periodic adjustments"""
        print("Starting continuous optimization...")

        while True:
            try:
                # Run short performance test
                start_time = time.time()
                processed_count = 0

                # Simulate some commands
                dummy_audio = np.random.rand(16000 * 3).astype(np.float32)  # 3 seconds of dummy audio

                while time.time() - start_time < test_duration:
                    try:
                        self.process_audio_input(dummy_audio)
                        processed_count += 1
                    except:
                        continue

                # Adaptive optimization based on results
                self.adaptive_optimization()

                # Print current status
                current_throughput = self.optimizer.get_current_throughput()
                print(f"Current throughput: {current_throughput:.2f} Hz, "
                      f"Processed: {processed_count} commands")

                # Brief pause
                time.sleep(2.0)

            except KeyboardInterrupt:
                print("Continuous optimization stopped by user")
                break
            except Exception as e:
                print(f"Error in continuous optimization: {e}")
                time.sleep(1.0)


def main():
    """Main function to run performance optimization"""
    print("Initializing optimized voice command pipeline...")

    # Create optimized pipeline
    pipeline = OptimizedVoicePipeline()

    try:
        # Run initial performance test
        print("\n" + "="*60)
        print("INITIAL PERFORMANCE TEST")
        print("="*60)

        success = pipeline.run_performance_test(duration=30.0)  # 30-second test

        if not success:
            print("\nInitial performance test failed. Starting continuous optimization...")
            pipeline.run_continuous_optimization()
        else:
            print("\nInitial performance test passed!")

            # Run continuous monitoring
            print("\nStarting continuous performance monitoring...")
            pipeline.run_continuous_optimization()

    except KeyboardInterrupt:
        print("\nPerformance optimization stopped by user")
    except Exception as e:
        print(f"Error during performance optimization: {e}")
    finally:
        if rclpy:
            rclpy.shutdown()


if __name__ == '__main__':
    main()