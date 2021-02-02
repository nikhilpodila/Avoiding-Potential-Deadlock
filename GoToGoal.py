from numpy.linalg import norm

# Class for Go to Goal algorithm (Attractive Potential)
class AttractivePotential(object):
    """
    Class to perform the Go to Goal task for the assignment
    Uses Attractive Potential method

    Attributes
    ----------
    x_start : list or list-like
        End-effector start position in task space
    x_target : list or list-like
        End-effector target position in task space
    K_att : float
        Attractive Potential Coefficient
    toGoal : list or list-like
        Vector from Current position to target position
    distanceGoal : float
        Euclidean distance from current position to target position

    """

    def __init__(self, x_start, x_target):
        """AttractivePotential contructor. Initializes the parameters
        for Attractive Potential algorithm.

        Parameters
        ----------
        x_start : list or list-like
            End-effector start position in task space
        x_target : list or list-like
            End-effector target position in task space

        Returns
        -------
        None

        """

        # Set target
        self.x_target = x_target

        # Attractive Potential Coefficient
        self.K_att = 0.2

        # Initialize Goal distance
        self.updateState(x_start)

    def updateState(self, current):
        """Update state value of end-effector in the algorithm
        given the measurements.
        Calculates required parameters from the state directly.

        Parameters
        ----------
        current : list or list-like
            Current end-effector position in task space

        Returns
        -------
        None

        """

        # Goal distance vector
        self.toGoal = self.x_target - current

        # Goal distance
        self.distanceGoal = norm(self.toGoal)

    def plan(self):
        """Performs planning to generate reference velocity,
        with the current known state of robot's end-effector

        Returns
        -------
        list or list-like
            Reference velocity in task space.

        """

        # Attractive potential
        return self.K_att * self.toGoal / self.distanceGoal
