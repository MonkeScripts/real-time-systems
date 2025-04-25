class Core:
    def __init__(self, core_id: str, speed_factor: float, scheduler: str):
        """
        A class representing a core.

        Attributes:
            core_id (str): id of core
            speed_factor(float): A numerical value representing the core's speed relative to a nominal speed. 1.0 represents the nominal speed. A value of 0.5 indicates a core that is 50% slower, and a value of 1.2 indicates a core that is 20% faster. The WCET of tasks assigned to a core must be adjusted by dividing the nominal WCET by the speed_factor
            scheduler (str): either EDF or RM
            componets (List(component)): list of components
        """
        self.core_id = core_id
        self.speed_factor = speed_factor
        self.scheduler = scheduler
        self.components = []

        def __str__(self):
            return (f"Core(core_id={self.core_id}, speed_factor={self.speed_factor}, "
                    f"scheduler={self.scheduler}, components={self.components})")