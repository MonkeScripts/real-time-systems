#!/usr/bin/python3

# POTENTIAL TESTCASES FROM GROK
from VSS import *

def run_test_cases():
    print("Test Case 1: Single Task")
    tasks1 = [Task(1, 10, 2, 4)]
    simulator1 = VSS(tasks1, 20)
    simulator1.simulate()
    simulator1.print_wcrt()

    print("\nTest Case 2: Two Tasks with Different Periods")
    tasks2 = [
        Task(1, 10, 2, 3),  # Higher priority (shorter period)
        Task(2, 20, 4, 6),  # Lower priority (longer period)
    ]
    simulator2 = VSS(tasks2, 40)
    simulator2.simulate()
    simulator2.print_wcrt()

    print("\nTest Case 3: Three Tasks with Potential Preemption")
    tasks3 = [
        Task(1, 5, 1, 2),  # Highest priority
        Task(2, 10, 2, 3),  # Medium priority
        Task(3, 20, 3, 5),  # Lowest priority
    ]
    simulator3 = VSS(tasks3, 30)
    simulator3.simulate()
    simulator3.print_wcrt()

    print("\nTest Case 4: Heavy Load with Tight Periods")
    tasks4 = [Task(1, 4, 1, 2), Task(2, 6, 2, 3), Task(3, 8, 2, 4)]
    simulator4 = VSS(tasks4, 20)
    simulator4.simulate()
    simulator4.print_wcrt()


if __name__ == "__main__":
    run_test_cases()
