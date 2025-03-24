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
        Task(1, 6, 0, 1), 
              Task(2, 60, 3, 4), 
              Task(3, 10, 1, 1),
              Task(4, 12, 1, 2),
              Task(5, 15, 1, 2),
              Task(6, 20, 1, 3),
              Task(7, 30, 1, 4),
    ]
    simulator2 = VSS(tasks2, 40)
    simulator2.simulate()
    simulator2.print_wcrt()

    print("\nTest Case 3: Three Tasks with Potential Preemption")
    tasks3 = [
        Task(1, 15, 0, 1), 
              Task(2, 20, 1, 2), 
              Task(3, 25, 2, 3),
              Task(4, 30, 2, 4),
              Task(5, 50, 3, 5),
              Task(6, 60, 4, 6),
              Task(7, 75, 5, 9),
              Task(8, 100, 3, 12),
              Task(9, 120, 5, 11),
              Task(10, 15, 5, 11),
              Task(11, 300, 5, 15)
    ]
    simulator3 = VSS(tasks3, 30)
    simulator3.simulate()
    simulator3.print_wcrt()

    print("\nTest Case 4: Heavy Load with Tight Periods")
    tasks4 = [Task(1, 40, 1, 3), 
              Task(2, 80, 2, 7), 
              Task(3, 100, 1, 13),
              Task(4, 160, 3, 18),
              Task(5, 200, 1, 22),
              Task(6, 300, 5, 27),
              Task(7, 320, 8, 29),
              Task(8, 400, 10, 34),
              Task(9, 480, 22, 35)]
    simulator4 = VSS(tasks4, 20)
    simulator4.simulate()
    simulator4.print_wcrt()


if __name__ == "__main__":
    run_test_cases()
