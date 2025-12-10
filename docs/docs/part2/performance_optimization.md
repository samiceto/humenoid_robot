# Performance Optimization for ≥15 Hz Real-Time Inference

## Overview

This document outlines the performance optimization strategies required to achieve ≥15 Hz real-time inference for the voice command pipeline in the Physical AI & Humanoid Robotics course. Meeting this performance requirement is critical for responsive human-robot interaction.

## Performance Requirements

### Minimum Specifications
- **Processing Frequency**: ≥15 Hz for all pipeline stages
- **Response Latency**: < 500ms from voice input to action execution
- **Concurrent Operations**: Support 3+ simultaneous processing tasks
- **Resource Constraints**:
  - CPU: < 80% average utilization
  - Memory: < 2GB RAM usage
  - GPU: < 80% utilization (if applicable)

### Performance Targets by Component
- **Speech Recognition**: ≤300ms processing time
- **Natural Language Understanding**: ≤100ms processing time
- **Action Planning**: ≤100ms processing time
- **Robot Control**: ≤50ms command cycle time

## Hardware Optimization

### Jetson Orin Nano Configuration
```bash
# Performance mode setup
sudo nvpmodel -m 0  # Maximum performance mode
sudo jetson_clocks  # Lock all clocks to maximum frequency

# Verify performance mode
sudo nvpmodel -q
jetson_clocks --show
```

### System-Level Optimizations
```bash
# CPU governor configuration
sudo cpufreq-set -g performance

# Memory management
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

# Real-time scheduling
sudo usermod -a -G realtime $USER
echo 'ulimit -r 99' | sudo tee -a /etc/security/limits.conf
```

## Software Optimization Strategies

### 1. Model Optimization

#### Quantization
```python
import torch
import torch.quantization as tq

def quantize_model(model, calib_loader):
    """Quantize model for faster inference"""
    # Set model to evaluation mode
    model.eval()

    # Specify quantization configuration
    model.qconfig = tq.get_default_qat_qconfig('fbgemm')

    # Prepare model for quantization
    model_prepared = tq.prepare_qat(model, inplace=False)

    # Calibrate with sample data
    with torch.no_grad():
        for data, target in calib_loader:
            model_prepared(data)

    # Convert to quantized model
    model_quantized = tq.convert(model_prepared, inplace=False)

    return model_quantized

# Example usage
quantized_model = quantize_model(whisper_model, calibration_data)
```

#### Model Distillation
```python
class DistilledModel(torch.nn.Module):
    """Distilled model for faster inference"""
    def __init__(self, teacher_model, student_config):
        super().__init__()
        self.teacher = teacher_model
        self.student = self.create_student_model(student_config)

    def create_student_model(self, config):
        """Create lightweight student model"""
        # Implement distilled architecture
        pass

    def forward(self, x):
        """Forward pass with knowledge transfer"""
        with torch.no_grad():
            teacher_output = self.teacher(x)

        student_output = self.student(x)
        return student_output
```

### 2. Pipeline Optimization

#### Asynchronous Processing
```python
import asyncio
import concurrent.futures
from queue import Queue
import threading

class AsyncPipelineOptimizer:
    def __init__(self):
        # Thread pool for CPU-bound tasks
        self.cpu_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # Process pool for CPU-intensive tasks
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=2)

        # Queues for pipeline stages
        self.audio_queue = Queue(maxsize=10)
        self.recognition_queue = Queue(maxsize=10)
        self.planning_queue = Queue(maxsize=10)

    async def process_audio_stream(self, audio_stream):
        """Process audio stream asynchronously"""
        loop = asyncio.get_event_loop()

        # Process in parallel stages
        tasks = [
            loop.run_in_executor(self.cpu_pool, self.speech_recognition, audio_chunk)
            for audio_chunk in audio_stream
        ]

        results = await asyncio.gather(*tasks)
        return results

    def speech_recognition(self, audio_chunk):
        """Perform speech recognition on audio chunk"""
        # Use optimized Whisper model
        result = self.whisper_model.transcribe(
            audio_chunk,
            fp16=True,  # Use half precision
            without_timestamps=True,
            max_new_tokens=128
        )
        return result
```

#### Batch Processing
```python
class BatchProcessor:
    def __init__(self, batch_size=4, max_wait_time=0.05):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.input_queue = asyncio.Queue()
        self.pending_requests = []
        self.processing_lock = threading.Lock()

    async def add_request(self, input_data):
        """Add request to batch processor"""
        future = asyncio.Future()
        request = {
            'input': input_data,
            'future': future,
            'timestamp': time.time()
        }

        self.pending_requests.append(request)

        # Process if batch is full or timeout reached
        if len(self.pending_requests) >= self.batch_size:
            await self.process_batch()
        else:
            # Schedule processing after timeout
            asyncio.create_task(self.maybe_process_batch())

        return await future

    async def maybe_process_batch(self):
        """Process batch if enough time has passed"""
        await asyncio.sleep(self.max_wait_time)
        if len(self.pending_requests) > 0:
            await self.process_batch()

    async def process_batch(self):
        """Process accumulated batch"""
        with self.processing_lock:
            requests_to_process = self.pending_requests[:self.batch_size]
            self.pending_requests = self.pending_requests[self.batch_size:]

            # Extract inputs
            inputs = [req['input'] for req in requests_to_process]

            # Process batch
            outputs = self.process_batched_inputs(inputs)

            # Complete futures
            for req, output in zip(requests_to_process, outputs):
                req['future'].set_result(output)
```

### 3. Memory Optimization

#### Efficient Memory Management
```python
import gc
import weakref
from collections import deque

class MemoryEfficientPipeline:
    def __init__(self, max_cache_size=100):
        self.max_cache_size = max_cache_size
        self.intermediate_cache = deque(maxlen=max_cache_size)
        self.tensor_pools = {}

    def process_with_memory_management(self, input_data):
        """Process data with efficient memory management"""
        # Clear cache periodically
        if len(self.intermediate_cache) > self.max_cache_size * 0.8:
            self.clear_cache()

        # Process data
        result = self.process_step(input_data)

        # Cache intermediate results
        self.intermediate_cache.append(result)

        return result

    def clear_cache(self):
        """Clear cached intermediate results"""
        self.intermediate_cache.clear()
        gc.collect()  # Force garbage collection

    def reuse_tensors(self, shape, dtype=torch.float32):
        """Reuse tensors to avoid allocation overhead"""
        key = (tuple(shape), dtype)

        if key in self.tensor_pools:
            tensor = self.tensor_pools[key].pop()
            if len(self.tensor_pools[key]) == 0:
                del self.tensor_pools[key]
        else:
            tensor = torch.empty(shape, dtype=dtype)

        return tensor
```

## Profiling and Monitoring

### Performance Profiling
```python
import cProfile
import pstats
import time
from functools import wraps

def profile_function(func):
    """Decorator to profile function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        result = func(*args, **kwargs)

        pr.disable()

        # Save profile data
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions

        return result
    return wrapper

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'execution_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'throughput': []
        }
        self.start_times = {}

    def start_timer(self, operation_name):
        """Start timer for operation"""
        self.start_times[operation_name] = time.time()

    def end_timer(self, operation_name):
        """End timer and record metrics"""
        if operation_name in self.start_times:
            elapsed = time.time() - self.start_times[operation_name]
            self.metrics['execution_times'].append(elapsed)
            return elapsed
        return None

    def get_average_frequency(self, window_size=100):
        """Calculate average processing frequency"""
        if len(self.metrics['execution_times']) == 0:
            return 0.0

        recent_times = self.metrics['execution_times'][-window_size:]
        if len(recent_times) == 0:
            return 0.0

        avg_time = sum(recent_times) / len(recent_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
```

### Real-time Monitoring
```python
import psutil
import threading
import time

class RealTimeMonitor:
    def __init__(self, update_interval=0.1):
        self.update_interval = update_interval
        self.monitoring = False
        self.stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'gpu_percent': [],
            'temperature': []
        }
        self.monitor_thread = None

    def start_monitoring(self):
        """Start real-time monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self):
        """Monitoring loop"""
        while self.monitoring:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.stats['cpu_percent'].append(cpu_percent)

            # Memory usage
            memory_percent = psutil.virtual_memory().percent
            self.stats['memory_percent'].append(memory_percent)

            # GPU usage (if available)
            gpu_percent = self._get_gpu_usage()
            self.stats['gpu_percent'].append(gpu_percent)

            # Temperature
            temp = self._get_temperature()
            self.stats['temperature'].append(temp)

            time.sleep(self.update_interval)

    def _get_gpu_usage(self):
        """Get GPU usage (NVIDIA specific)"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0.0

    def _get_temperature(self):
        """Get system temperature"""
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
            elif 'cpu_thermal' in temps:
                return temps['cpu_thermal'][0].current
        except:
            pass
        return 0.0
```

## Optimization Techniques for Specific Components

### Speech Recognition Optimization
```python
class OptimizedWhisper:
    def __init__(self, model_size="small"):
        # Load model with optimizations
        self.model = whisper.load_model(model_size, device="cuda",
                                       download_root="./models")

        # Enable half precision
        self.model = self.model.half()

        # Set to evaluation mode
        self.model.eval()

    def transcribe_optimized(self, audio, language="en"):
        """Optimized transcription with performance settings"""
        # Use half precision and limit tokens
        result = self.model.transcribe(
            audio,
            language=language,
            fp16=True,  # Half precision
            without_timestamps=True,
            max_new_tokens=128,  # Limit output length
            compression_ratio_threshold=2.0,  # Early stopping
            logprob_threshold=-1.0  # Early stopping
        )
        return result
```

### LLM Optimization
```python
class OptimizedLLM:
    def __init__(self, model_name="distilgpt2"):
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            GPT2LMHeadModel
        )

        # Load quantized model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Half precision
            device_map="auto"  # Automatic device placement
        )

        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set to evaluation mode
        self.model.eval()

    def generate_optimized(self, prompt, max_length=64):
        """Optimized text generation"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        # Move to appropriate device
        inputs = inputs.to(self.model.device)

        # Generate with optimizations
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
```

## Benchmarking Framework

### Performance Benchmarking
```python
import time
import statistics
from typing import Callable, List, Dict

class PerformanceBenchmark:
    def __init__(self):
        self.results = {}

    def benchmark_function(self, func: Callable, test_data: List, iterations: int = 100) -> Dict:
        """Benchmark a function with test data"""
        execution_times = []

        for i in range(iterations):
            start_time = time.perf_counter()

            # Run function with test data
            for data in test_data:
                func(data)

            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)

        # Calculate statistics
        stats = {
            'mean_time': statistics.mean(execution_times),
            'median_time': statistics.median(execution_times),
            'std_dev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'min_time': min(execution_times),
            'max_time': max(execution_times),
            'throughput_hz': len(test_data) / statistics.mean(execution_times),
            'iterations': iterations
        }

        return stats

    def run_pipeline_benchmark(self, pipeline_func, test_scenarios):
        """Benchmark complete pipeline"""
        scenario_results = {}

        for scenario_name, scenario_data in test_scenarios.items():
            print(f"Benchmarking {scenario_name}...")

            # Warm up
            for _ in range(10):
                pipeline_func(scenario_data['input'])

            # Actual benchmark
            results = self.benchmark_function(
                pipeline_func,
                [scenario_data['input']] * 50,  # 50 iterations
                iterations=50
            )

            scenario_results[scenario_name] = results

            # Check if requirements are met
            if results['throughput_hz'] >= 15.0:
                print(f"✅ {scenario_name}: {results['throughput_hz']:.2f} Hz (PASS)")
            else:
                print(f"❌ {scenario_name}: {results['throughput_hz']:.2f} Hz (FAIL)")

        return scenario_results
```

## Optimization Validation

### Performance Validation Script
```python
def validate_performance_requirements():
    """Validate that performance requirements are met"""

    # Initialize benchmarking framework
    benchmark = PerformanceBenchmark()

    # Define test scenarios
    test_scenarios = {
        'speech_recognition': {
            'input': 'audio_sample.wav',
            'expected_min_freq': 15.0
        },
        'command_processing': {
            'input': 'move forward 1 meter',
            'expected_min_freq': 25.0
        },
        'action_planning': {
            'input': {'intent': 'navigate', 'target': (1.0, 2.0)},
            'expected_min_freq': 20.0
        },
        'end_to_end': {
            'input': 'Please move forward and turn left',
            'expected_min_freq': 15.0
        }
    }

    # Run benchmarks
    results = benchmark.run_pipeline_benchmark(process_voice_command, test_scenarios)

    # Validate requirements
    all_passed = True
    for scenario, metrics in results.items():
        expected = test_scenarios[scenario]['expected_min_freq']
        actual = metrics['throughput_hz']

        if actual < expected:
            print(f"❌ {scenario} failed: {actual:.2f} Hz < {expected:.2f} Hz required")
            all_passed = False
        else:
            print(f"✅ {scenario} passed: {actual:.2f} Hz >= {expected:.2f} Hz required")

    # Overall result
    if all_passed:
        print("\n🎉 All performance requirements satisfied!")
        return True
    else:
        print("\n❌ Some performance requirements not met!")
        return False
```

## Continuous Performance Monitoring

### Performance Dashboard
```python
# Example dashboard setup using Dash
import dash
from dash import dcc, html
import plotly.graph_objs as go
from threading import Thread
import time

class PerformanceDashboard:
    def __init__(self, monitor: RealTimeMonitor):
        self.monitor = monitor
        self.app = dash.Dash(__name__)

        self.app.layout = html.Div([
            html.H1("Real-time Performance Dashboard"),

            dcc.Graph(id='cpu-usage-graph'),
            dcc.Graph(id='memory-usage-graph'),
            dcc.Graph(id='gpu-usage-graph'),
            dcc.Graph(id='temperature-graph'),

            dcc.Interval(
                id='interval-component',
                interval=1*1000,  # Update every second
                n_intervals=0
            )
        ])

        # Set up callbacks
        self.setup_callbacks()

    def setup_callbacks(self):
        @self.app.callback(
            [dash.dependencies.Output('cpu-usage-graph', 'figure'),
             dash.dependencies.Output('memory-usage-graph', 'figure'),
             dash.dependencies.Output('gpu-usage-graph', 'figure'),
             dash.dependencies.Output('temperature-graph', 'figure')],
            [dash.dependencies.Input('interval-component', 'n_intervals')]
        )
        def update_graphs(n):
            # CPU usage graph
            cpu_fig = go.Figure(data=go.Scatter(y=self.monitor.stats['cpu_percent'][-100:]))
            cpu_fig.update_layout(title='CPU Usage (%)', yaxis=dict(range=[0, 100]))

            # Memory usage graph
            mem_fig = go.Figure(data=go.Scatter(y=self.monitor.stats['memory_percent'][-100:]))
            mem_fig.update_layout(title='Memory Usage (%)', yaxis=dict(range=[0, 100]))

            # GPU usage graph
            gpu_fig = go.Figure(data=go.Scatter(y=self.monitor.stats['gpu_percent'][-100:]))
            gpu_fig.update_layout(title='GPU Usage (%)', yaxis=dict(range=[0, 100]))

            # Temperature graph
            temp_fig = go.Figure(data=go.Scatter(y=self.monitor.stats['temperature'][-100:]))
            temp_fig.update_layout(title='Temperature (°C)')

            return cpu_fig, mem_fig, gpu_fig, temp_fig

    def run(self, debug=False):
        self.app.run_server(debug=debug)
```

## Troubleshooting Performance Issues

### Common Performance Bottlenecks
1. **Model Loading**: Pre-load models at startup
2. **Memory Allocation**: Use tensor pools and reuse memory
3. **CPU-GPU Transfer**: Minimize data transfers between devices
4. **Sequential Processing**: Use asynchronous and parallel processing
5. **I/O Operations**: Optimize disk and network I/O

### Performance Tuning Checklist
- [ ] Enable GPU acceleration where applicable
- [ ] Use model quantization for faster inference
- [ ] Implement batch processing for throughput
- [ ] Optimize memory management to reduce allocation overhead
- [ ] Use asynchronous processing for non-blocking operations
- [ ] Profile code to identify bottlenecks
- [ ] Monitor resource usage during operation
- [ ] Test with realistic workloads
- [ ] Validate performance meets ≥15 Hz requirement

## Conclusion

Achieving ≥15 Hz real-time inference requires a combination of hardware optimization, software optimization techniques, and careful system design. The optimization strategies outlined in this document should be implemented systematically, with continuous monitoring and validation to ensure the performance requirements are consistently met across different operating conditions and workloads.

The key to success is balancing performance with accuracy, ensuring that optimization techniques do not significantly degrade the quality of the voice command pipeline while achieving the required processing frequency.