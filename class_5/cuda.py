import torch
import time

# It returns True if CUDA is available, False otherwise
cuda_available = torch.cuda.is_available()

print(f"\nIs CUDA available? {cuda_available}")

if cuda_available:
    # Get the name of the GPU (e.g., "NVIDIA GeForce RTX 3080")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Device: {gpu_name}")
    
    # Check how many CUDA-capable GPUs are detected
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")
    
    # Get the CUDA version that PyTorch was compiled with
    # This might differ from the CUDA version shown by nvidia-smi, and that's okay
    # as long as the driver supports this version or newer
    cuda_version = torch.version.cuda
    print(f"CUDA Version (PyTorch): {cuda_version}")
    
    # Check total GPU memory available
    # This is the constraint we'll be working within - everything must fit here
    total_memory = torch.cuda.get_device_properties(0).total_memory
    
    # Convert from bytes to gigabytes for readability
    total_memory_gb = total_memory / (1024**3)
    print(f"Total GPU Memory: {total_memory_gb:.2f} GB")
    
else:
    print("\nCUDA is not available on this system.")

# Set the device we'll use for computations
device = torch.device("cuda" if cuda_available else "cpu")
print(f"\nUsing device: {device}")

# Create a tensor on the CPU (the default)
# When you create a tensor normally, it lives in CPU memory
cpu_tensor = torch.randn(3, 3)
print(f"\nTensor created on CPU:")
print(f"Device: {cpu_tensor.device}")
print(f"Shape: {cpu_tensor.shape}")

if cuda_available:
    # Move the tensor to GPU
    # This copies the data from CPU memory to GPU memory
    # The original CPU tensor still exists - we're creating a copy on the GPU
    gpu_tensor = cpu_tensor.to(device)
    print(f"\nSame tensor moved to GPU:")
    print(f"Device: {gpu_tensor.device}")
    print(f"Shape: {gpu_tensor.shape}")
    
    # Alternative ways to create tensors directly on GPU
    # Instead of creating on CPU then moving, we can create directly on GPU
    gpu_tensor_direct = torch.randn(3, 3, device=device)
    print(f"\nTensor created directly on GPU:")
    print(f"Device: {gpu_tensor_direct.device}")
    
    # Important note about operations: tensors must be on the same device
    # You cannot add a CPU tensor to a GPU tensor - you'll get an error
    # This is a common mistake for beginners
    print("\nImportant: All tensors in an operation must be on the same device!")
    
    # Moving data back to CPU
    # You need to do this when you want to print values, save to disk, or use
    # the data with libraries that don't understand GPU tensors
    result_on_cpu = gpu_tensor.to('cpu')
    print(f"Tensor moved back to CPU: {result_on_cpu.device}")

# We'll use matrix multiplication because it's:
# 1. Fundamental to neural networks (every layer involves matrix multiplication)
# 2. Highly parallelizable (perfect for GPU acceleration)
# 3. Computationally intensive enough to show clear differences

# Size of the matrices we'll multiply
# Try changing this value to see how it affects the speedup
# Larger matrices show more dramatic GPU advantage
matrix_size = 10000

print(f"\nMultiplying two {matrix_size}x{matrix_size} matrices...")

# Create random matrices for testing
# We create them on CPU first
matrix_a_cpu = torch.randn(matrix_size, matrix_size)
matrix_b_cpu = torch.randn(matrix_size, matrix_size)

# ===== CPU TIMING =====
print("\nTiming CPU computation...")
# Use time.time() to measure elapsed time
cpu_start = time.time()

# Perform matrix multiplication on CPU
result_cpu = torch.matmul(matrix_a_cpu, matrix_b_cpu)

cpu_end = time.time()
cpu_time = cpu_end - cpu_start

print(f"CPU time: {cpu_time:.4f} seconds")

# ===== GPU TIMING =====
if cuda_available:
    print("\nTiming GPU computation...")
    
    # Move matrices to GPU
    matrix_a_gpu = matrix_a_cpu.to(device)
    matrix_b_gpu = matrix_b_cpu.to(device)
    
    # Important: CUDA operations are asynchronous by default
    # This means Python might continue before the GPU finishes
    # We use torch.cuda.synchronize() to wait for GPU to complete
    # Without this, our timing would be inaccurate
    
    gpu_start = time.time()
    
    # Perform matrix multiplication on GPU
    result_gpu = torch.matmul(matrix_a_gpu, matrix_b_gpu)
    
    # Wait for GPU to finish
    torch.cuda.synchronize()
    
    gpu_end = time.time()
    gpu_time = gpu_end - gpu_start
    
    print(f"GPU time: {gpu_time:.4f} seconds")
    
    # Calculate and display the speedup
    speedup = cpu_time / gpu_time
    print(f"\nSpeedup: {speedup:.2f}x faster on GPU")
    
    # Verify the results are the same (within floating point precision)
    # We move GPU result to CPU to compare
    result_gpu_cpu = result_gpu.to('cpu')
    # Use allclose instead of equal because floating point math isn't exact
    results_match = torch.allclose(result_cpu, result_gpu_cpu, rtol=1e-4)
    print(f"Results match: {results_match}")
    
    if not results_match:
        print("Warning: CPU and GPU results differ slightly due to floating point precision")
        print("This is normal and expected - GPUs may use slightly different math operations")

# Understanding memory management is crucial for working with GPUs effectively.
# GPU memory is limited and must be managed carefully.

if cuda_available:
    # Function to display current GPU memory usage
    def print_gpu_memory():
        """Helper function to show GPU memory statistics"""
        # Memory currently allocated by tensors
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        # Memory reserved by PyTorch's caching allocator
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
    
    print("\nInitial GPU memory state:")
    print_gpu_memory()
    
    # Create a large tensor on GPU
    print("\nCreating a large tensor (2GB)...")
    # Calculate size for approximately 2GB of float32 data
    # float32 = 4 bytes, so 2GB = 500 million floats
    large_tensor = torch.randn(25000, 20000, device=device)
    print("After creating large tensor:")
    print_gpu_memory()
    
    # When we delete the tensor, memory isn't immediately freed
    # PyTorch keeps it cached for potential reuse (faster than requesting from GPU)
    print("\nDeleting the tensor...")
    del large_tensor
    print("After deletion (memory still cached):")
    print_gpu_memory()
    
    # Explicitly clear the cache to free memory
    # This is useful between training runs or when you need maximum memory available
    print("\nClearing GPU cache...")
    torch.cuda.empty_cache()
    print("After clearing cache:")
    print_gpu_memory()

# Here's a minimal working example with a tiny dummy network
# This shows the pattern working end-to-end
if cuda_available:
    print("\nRunning minimal example with dummy network...")
    
    # Create a tiny neural network
    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 2)
    )
    
    # Move model to GPU
    dummy_model = dummy_model.to(device)
    print(f"Model is on: {next(dummy_model.parameters()).device}")
    
    # Create dummy data
    dummy_data = torch.randn(32, 10, device=device)  # batch of 32 samples
    dummy_labels = torch.randint(0, 2, (32,), device=device)  # binary labels
    
    # Set up training components
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)
    
    # Train for a few steps
    print("\nTraining for 5 steps...")
    for step in range(5):
        # Forward pass
        outputs = dummy_model(dummy_data)
        loss = criterion(outputs, dummy_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step+1}, Loss: {loss.item():.4f}")
    
    print("\nYour model will follow this same pattern!")
    print("Just replace the dummy model with your prepared neural network.")
