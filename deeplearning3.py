import numpy as np 
# McCulloch-Pitts Neuron 

def mcculloch_pitts_neuron(inputs, weights, threshold1): 
  # Calculate the weighted sum 
  weighted_sum = np.dot(inputs, weights)
  #print(f"weighted_sum:\t{weighted_sum}")  
  # Apply threshold logic 
  output = 1 if weighted_sum > threshold1 else 0 
  #print(f"Output:\t{output}")
  return output 
#Training the McCulloch-Pitts Neuron for the OR problem 

def train_mcculloch_pitts_neuron(inputs, target_output, weights, threshold1, learning_rate=0.2, epochs=50): 
  for epoch in range(epochs): 
    print(f"Epoch {epoch + 1}")
    for input_values, target in zip(inputs, target_output): 
      # Get the neuron output 
      output = mcculloch_pitts_neuron(input_values, weights, threshold1) 
      # Update the weights based on the error and learning rate 
      error = target - output 
      weights =(weights + learning_rate * error * input_values).astype(float)
      # Display the updated weights and output 
      print("Input:", input_values, "Target Output:", target, "Predicted Output:", output, "Updated Weights:", weights) 
      # print("-----------------------")
    print("\n")

def step_function(x):
    return 1 / (1 + np.exp(-x))


# Inputs for the OR problem 
inputs = np.array([[1, 1, 1, 1], [0, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 1]]) 

# Target output for OR problem 
target_output = np.array([1, 1, 0, 0])  

# Initial weights 
initial_weights = np.array([0, 0, 0, 0]) 

#sigmoid value
x = 2.0

# Training the McCulloch-Pitts Neuron for 4 epochs with a threshold of 0.5 and learning rate of 0.2 
train_mcculloch_pitts_neuron(inputs, target_output, initial_weights, threshold1 = sigmoid_step_function(x) , learning_rate=0.2, epochs=8)
