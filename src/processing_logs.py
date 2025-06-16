import re
import sys
import os
import matplotlib.pyplot as plt

def main(log_filename):
    # Construct full path to logs directory in parent folder
    log_file_path = os.path.join("..", "logs", log_filename)

    # Value lists
    balls_picked = []
    balls_deposited = []
    q_values_step = []
    sum_q_values = []

    # Cumulative counters
    annealed_count = []
    random_action_count = []
    model_action_count = []

    # Temporary counters
    annealed = 0
    random_action = 0
    model_action = 0

    # Regex patterns
    patterns = {
        "Balls Picked:": (balls_picked, r"Balls Picked:\s*(\d+(\.\d+)?)"),
        "Balls Disposed:": (balls_deposited, r"Balls Deposited:\s*(\d+(\.\d+)?)"),
        "Q values this step:": (q_values_step, r"Q values this step:\s*(\d+(\.\d+)?)"),
        "Sum Q Values:": (sum_q_values, r"Sum Q Values:\s*(\d+(\.\d+)?)")
    }

    # Read and process the log file
    try:
        with open(log_file_path, "r") as file:
            for line in file:
                line_tracked = False

                # Extract numeric data
                for key, (lst, pattern) in patterns.items():
                    if key in line:
                        match = re.search(pattern, line)
                        if match:
                            lst.append(float(match.group(1)))
                            line_tracked = True

                # Count events
                if "Annealed" in line:
                    annealed += 1
                if "Random action" in line:
                    random_action += 1
                if "Model action" in line:
                    model_action += 1

                # Record counters per step
                if line_tracked:
                    annealed_count.append(annealed)
                    random_action_count.append(random_action)
                    model_action_count.append(model_action)

    except FileNotFoundError:
        print(f"Error: File '../logs/{log_filename}' not found.")
        sys.exit(1)


    # ---- Plot 1: Value Metrics ----
    plt.figure(figsize=(12, 6))
    plt.title("Log Value Metrics")

    if balls_picked:
        plt.plot(balls_picked, label="Balls Picked")
    if balls_deposited:
        plt.plot(balls_deposited, label="Balls Deposited")
    if q_values_step:
        plt.plot(q_values_step, label="Q values this step")
    if sum_q_values:
        plt.plot(sum_q_values, label="Sum Q Values")

    plt.xlabel("Steps / Entries")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # ---- Plot 2: Cumulative Event Counts ----
    plt.figure(figsize=(12, 6))
    plt.title("Cumulative Count of Events")

    plt.plot(annealed_count, label="Annealed", color="orange")
    plt.plot(random_action_count, label="Random action", color="red")
    plt.plot(model_action_count, label="Model action", color="green")

    plt.xlabel("Steps / Entries")
    plt.ylabel("Cumulative Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python log_plotter.py <log_filename>")
        sys.exit(1)

    log_filename = sys.argv[1]
    main(log_filename)
