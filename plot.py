import math
import matplotlib.pyplot as plt

pivot = 3.879  # single process performance in seconds

two_processes_per_node_time = [2.007, 3.446, 4.824, 6.524]
two_processes_per_node_np = [2, 4, 8, 16]

single_process_per_node_time = [3.879, 4.092, 4.978, 6.194]
single_process_per_node_np = [1, 2, 4, 8]

four_processes_per_node_time = [1.130, 5.826, 8.318, 10.206]
four_processes_per_node_np = [4, 8, 16, 32]

def plot(time, np, process_number_per_node, y_low_limit, y_high_limit):
  print time
  speedup = [(pivot / t) for t in time]
  print speedup
  number_of_processes_in_power_of_two = [int(math.log(n, 2)) for n in np]
  x_axis = [-1]
  x_axis.extend(number_of_processes_in_power_of_two)
  x_axis.append(number_of_processes_in_power_of_two[len(number_of_processes_in_power_of_two) - 1] + 1)
  fig, ax = plt.subplots()

  ax.set_xticks(x_axis)
  ax.autoscale(False)

  plt.ylim(y_low_limit, y_high_limit)
  plt.xlabel('Number of Processes in power of 2')
  plt.ylabel('Speedup')
  ax.plot(number_of_processes_in_power_of_two, speedup, 'o', number_of_processes_in_power_of_two, speedup, 'k')
  plt.title("MPI Speedup Analysis: " + process_number_per_node +" per Node")
  plt.show()
  fig.savefig(process_number_per_node)


plot(single_process_per_node_time, single_process_per_node_np, "One Process", 0.5, 1.05)
plot(two_processes_per_node_time, two_processes_per_node_np, "Two Processes", 0.5, 2)
plot(four_processes_per_node_time, four_processes_per_node_np, "Four Processes", 0.3, 4)