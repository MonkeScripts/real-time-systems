#!/usr/bin/python3
# Fixed-priority preemptive scheduling is presented in the lecture slides and in Chapter 4 of [1]. When
# the priorities are proportional (monotonic) to the rate of the tasks (the inverse of the period, i.e.,
# 1/T ), this scheduling technique is also called Rate Monotonic scheduling (RM).
# The simulator takes as input the application consisting of a set of tasks stored in a file and the
# simulation time. The output is a list of worst-case response times (WCRT) observed during the simu-
# lation for each task. A suggestion for implementation is available in Algorithm 1. You may implement
# it differently if you want, especially the part related to advancing the time.

# You want to simulate every single step, AdvanceTime can simply return 1 time unit each time.
# If you prefer to skip idle intervals, AdvanceTime could jump to the next release. You can decide
# on its exact implementation.
# • Consider how you generate the random values for Ci. Are they uniformly distributed in the
# interval [BCET, WCET], or did you use another distribution?
# • A ready job at the time moment currentTime is a job that has already been released, i.e., it
# has τi.jobj .release ≤ currentTime. The job executing at currentTime is the highest-priority
# job out of those that are ready.
# • There is no need to explicitly “stop” or “preempt” a job already on the processor, since at every
# time instant currentTime (or after each AdvanceTime call) the simulator checks which job has
# the highest priority and decrements that job’s execution time.
# • remember worst-case response time: For each task, you have to print out its WCRT. This
# means that for a task τi you have to find the maximum response time over all the simulated jobs
# τi.jobj of the task.
import random
from queue import PriorityQueue
import matplotlib.pyplot as plt

class Task():
    """
    A class representing a task.

    Attributes:
        task_id (int): ID of the task
        period (float): Fixed amount of time between when one instance of the task starts (or becomes ready to run) and when the next instance begins.
        bcet (float): Best case execution time
        wcet (float): Worst case execution time
        priority (float): Priority of the task compared to other tasks
        jobs (list(Job)): List of jobs executing the task. A task running several times on different input data generates a sequence of instances / jobs
        max_wcrt (float): maximum worst case response time out of all jobs executing the task
    """
    def __init__(self, task_id: int, period: float, bcet: float, wcet: float):
        self.task_id = task_id
        self.period = period
        self.bcet = bcet
        self.wcet = wcet
        # Rate monotonic: Fixed priority proportional to its rate
        self.priority = 1.0 / period
        self.jobs = []
        self.max_wcrt = 0

class Job():
    """
    A class representing a job.

    Attributes:
        job_id (int): ID of job
        release_time (float): Time when the job becomes available to run
        response_time (float): Time from when a job is released to when it completes
        exec_time (float): Total amount of time remaining to complete the job
        original_exec_time (float): Original amount of time given to complete the job
        start_time (float): Starting time of the job
        end_time (float): Ending time of the job
    Note:
        exec_time is also a measure of how much the task is completed. 
        It is in time units because we expect the task to be completed every period, so the period is the allowable time for execution for each task
    """
    def __init__(self, job_id: int, release_time: float, exec_time: float):
        self.job_id = job_id
        self.release_time = release_time
        self.response_time = 0.0
        self.exec_time = exec_time
        self.original_exec_time = exec_time
        self.start_time = 0.0
        self.end_time = 0.0


class VSS():
    """
    A class representing a Very Simple Simulator(VSS).

    Attributes:
        tasks (List(Task)): List of tasks
        simulation_time (float): Time available for simulation
        current_time (float): Current time of the simulator
        history (List((task_id, start, end)): History of the simulation
    """

    def __init__(self, tasks: list[Task], simultation_time: int):
        self.tasks = tasks
        self.simulation_time = simultation_time
        self.current_time = 0.0
        self.history = []

    def initialize_jobs(self):
        for task in self.tasks:
            release_time = 0
            job_counter = 1
            # Add jobs every period
            while release_time <= self.simulation_time:
                exec_time = random.uniform(task.bcet, task.wcet)
                job = Job(job_id=job_counter, release_time=release_time, exec_time=exec_time)
                task.jobs.append(job)
                release_time += task.period
                job_counter += 1

    def get_ready_queue(self):
        ready_queue = PriorityQueue()
        for task in self.tasks:
            for job in task.jobs:
                if job.release_time <= self.current_time and job.exec_time > 0:
                    # TODO WHAT SHOULD WE DO IF JOBS HAVE THE SAME PRIORITY?
                    # CURRENTLY TIEBREAKER IS THE TASKID

                    ready_queue.put((-task.priority, task.task_id, job.job_id, task, job))

                    # print(f"Error putting job into queue. Suspect jobs have the same priority")
        return ready_queue

    def advance_time(self):
        return 1

    def print_wcrt(self):
        for task in self.tasks:
            print(f"wcrt for task{task.task_id} is {task.max_wcrt}\n")

    def peek_highest_priority(self, queue):
        if queue.empty():
            return None
        return queue.queue[0]  # Peek at highest priority item

    def simulate(self):
        self.initialize_jobs()
        while self.current_time <= self.simulation_time:
            ready_queue = self.get_ready_queue()
            if ready_queue.empty():
                # print("No jobs in queue!")
                delta = self.advance_time()
                self.current_time += delta
                continue
            _, _, _, task, current_job = self.peek_highest_priority(ready_queue)
            if not(current_job):
                print(f"No current jobs exist")
                delta = self.advance_time()
                self.current_time += delta
                continue
            new_job = current_job.exec_time == current_job.original_exec_time
            if new_job:
                current_job.start_time = self.current_time
            delta = self.advance_time()
            self.current_time += delta
            current_job.exec_time -= delta
            self.history.append(
                (task.task_id, current_job.job_id, self.current_time - delta, self.current_time)
            )
            if current_job.exec_time <= 0:
                current_job.response_time = self.current_time - current_job.release_time
                task.max_wcrt = max(task.max_wcrt, current_job.response_time)
                current_job.end_time = self.current_time





