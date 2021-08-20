import sys
import matplotlib.pyplot as plt 

if __name__ == "__main__":

    # Get the lists for both CPU & Memory Usage
    cpu = []
    mem = []

    for line in open(f'resources-{sys.argv[1]}.log'):
        lineList = line.lstrip().rstrip().split(' ')
        cpu.append(float(lineList[0]))
        mem.append(float(lineList[len(lineList)-1]))

    # Make a list for representing x-axis by checking length of the list 
    # (e.g. get amount of recordings to graph by looking at length of lists)
    x = [i for i in range(0, len(mem))]

    # Define the CPU usage subplot
    maxValue = max(10, round(max(cpu)))
    plt.subplot(2, 1, 1)
    plt.plot(x, cpu, color='#2ca02c', linestyle='dashed', linewidth = 1,
            marker='o', markerfacecolor='black', markersize=3) 

    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

    plt.yticks([a for a in range(0, maxValue, round(maxValue/10))],
               [b for b in range(0, maxValue, round(maxValue/10))])

    plt.ylabel('CPU Usage (%)')

    # Define the Memory usage subplot
    maxValue = max(10, round(max(mem)))
    plt.subplot(2, 1, 2)
    plt.plot(x, mem, color='#1f77b4', linestyle='dashed', linewidth = 1,
            marker='o', markerfacecolor='black', markersize=3) 

    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)

    plt.yticks([a for a in range(0, maxValue, round(maxValue/10))],
               [b for b in range(0, maxValue, round(maxValue/10))])

    plt.ylabel('Memory Usage (%)') 
    plt.xlabel('Time (seconds)') 

    # Write out the final plot including both subplots
    plt.suptitle(f'Resource Consumption')
    plt.savefig(f'resource_usage_{sys.argv[1]}.png')