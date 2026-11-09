# Goal of the project

1. measures the performance of different ASR models running on device
2. We will measure 3 main metrics: accuracy, speed, memory
3. 3 main platforms: Jetson orin NX 16Gb, Jetson orin Nano 8gb, Mac M2


# Rules

- I need to code myself , you don't help me to first write the code
- When I tell the agent that I finished , the agent needs to ask me questions to check that I well understood and it checks for bugs in my code
- everything needs to pass a Pull Request (PR) and the continuous integration (CI)
- If I ask you to code something, don't tell me directly the answer first tell me some indices or build me a plan

# Explanation of the metrics:

## Accuracy

- WER for european languages
- CER for hindi dialects
- LID (language identification) routing accuracy

## Speed

- p50, p99
- time to load the model and to process the first chunk of data
- RTF

## Memory

- Peak of memory 
- freeing of memory working correctly


# Repo structure

- /bench
- /kernels
- /tests