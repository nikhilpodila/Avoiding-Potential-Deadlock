import numpy as np
import pybullet as p
from numpy.linalg import norm
from GoToGoal import AttractivePotential

# Class for obstacle algorithm
class ObstacleAlgorithm(object):
    """
    This is an Abstract Class for providing common attributes and methods
    for all Obstacle Avoidance algorithms

    Attributes
    ----------
    x_obs : list or list-like
        Position of obstacle in task space
    x_target : list or list-like
        Target position of end-effector in task space
    d_min : float
        Minimum allowed distance between obstacle and end-effector
    _t : float
        (Private) time value of the simulation
    toObstacle : list or list-like
        Vector from end-effector position to obstacle position
    distanceObstacle : float
        Euclidean distance between end-effector and obstacle
    toGoal : list or list-like
        Vector from end-effector position to target position
    distanceGoal : float
        Euclidean distance between end-effector and target
    """


    def __init__(self, x_start, x_target):
        """ObstacleAlgorithm constructor. Sets the basic parameters for
        the Obstacle Avoidance algorithms

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

        # Location of obstacle
        self.x_obs = np.array(x_start) + 0.5 * (np.array(x_target) - np.array(x_start))

        # Location of target
        self.x_target = x_target

        # Place text at obstacle location
        p.addUserDebugText("O <- obstacle", list(self.x_obs),[0,0,0])

        # Minimum distance to obstacle for avoidance
        self.d_min = .2

        # Initialize experiment time
        self._t = None

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

        # Vector from Obstacle to current state
        self.toObstacle = self.x_obs - current

        # Distance value
        self.distanceObstacle = norm(self.toObstacle)

        # Vector from Current state to Goal
        self.toGoal = self.x_target - current

        # Distance value
        self.distanceGoal = norm(self.toGoal)

class TotalPotentialAlgorithm(object):
    """
    Class to implement the net/total potential field algorithm as
    a combination of Attractive Potential field and Repulsive Potential field.

    Attributes
    ----------
    attractivePotential : AttractivePotential
        Class instance to execute attractive potential algorithm on robot
    repulsivePotential : RepulsivePotentialAlgorithm
        Class instance to execute repulsive potential algorithm on robot.


    """

    def __init__(self, x_start, x_target):
        """TotalPotentialAlgorithm constructor. Initializes both
        RepulsivePotentialAlgorithm and AttractivePotential algorithm

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

        self.repulsivePotential = RepulsivePotentialAlgorithm(x_start, x_target)

        self.attractivePotential = AttractivePotential(x_start, x_target)

    def getStatus(self):
        """Returns the status of Obstacle avoidance

        Returns
        -------
        bool
            Returns True is obstacle is close to the robot

        """
        return self.repulsivePotential.getStatus()

    def updateState(self, current):
        """Updates the state of both RepulsivePotentialAlgorithm and
        AttractivePotential based on the measurements provided

        Parameters
        ----------
        current : list or list-like
            Current end-effector position in task space

        Returns
        -------
        None

        """

        self.repulsivePotential.updateState(current)
        self.attractivePotential.updateState(current)

    def plan(self, t):
        """Performs planning to generate reference velocity,
        with the current known state of robot's end-effector.
        Here, attractive potential field and repulsive potential field
        results are combined to give a net potential field.

        Parameters
        ----------
        t : float
            Time value is unused. Implemented only for Boundary Following
        Returns
        -------
        list or list-like
            Reference velocity in task space.

        """

        x_dot_ref = self.attractivePotential.plan()
        x_dot_ref -= self.repulsivePotential.plan()

        return x_dot_ref

# Class for Repulsive potential field algorithm
class RepulsivePotentialAlgorithm(ObstacleAlgorithm):
    """
    Class to perform Repulsive Potential Field algorithm for Obstacle Avoidance
    on the end-effector

    Attributes
    ----------
    K_rep : float
        Repulsive Potential Field Coefficient

    """

    def __init__(self, x_start, x_target):
        """RepulsivePotentialAlgorithm constructor. Initialized parameters
        that are required to perform Repulsive Potential field.

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

        # Set obstacle avoidance
        super().__init__(x_start, x_target)

        # Repulsive Potential Coefficient
        self.K_rep = .3

        # Set algorithm based parameters
        self.updateState(x_start)

    def getStatus(self):
        """Returns the status of Obstacle avoidance

        Returns
        -------
        bool
            Returns True is obstacle is close to the robot

        """

        return self.distanceObstacle < self.d_min

    def plan(self):
        """Performs planning to generate reference velocity,
        with the current known state of robot's end-effector.
        Here, the reference velocity is calculated according to
        the repulsive potential field algorithm.

        Returns
        -------
        list or list-like
            Reference velocity in task space.

        """
        # Dxxo = self.distanceObstacle

        tangent = self.K_rep/(self.distanceObstacle**2)
        tangent *= ((1/self.distanceObstacle) - (1/self.d_min))
        tangent *= self.toObstacle / self.distanceObstacle

        return tangent


class BoundaryFollowingAlgorithm(ObstacleAlgorithm):
    """
    Class to perform Boundary Following as a method of Obstacle Avoidance
    for the end-effector

    Attributes
    ----------
    K_boundaryStart : float
        Coefficient of Boundary Following at the first iteration of the algorithm
    K_boundaryContinue : float
        Coefficient of Boundary Following after first iteration of the algorithm
    boundaryDetectedGoalDistance : float
        Distance to goal at the first iteration of this algorithm
    longitudinalDirection : list or list-like
        Direction along the longitude of the sphere. See documentation.
    _dt : float
        (Private) Time step or time difference parameter for the simulation


    """
    
    def __init__(self, x_start, x_target, dt):
        """BoundaryFollowingAlgorithm constructor. Initializes all parameters
        required to perform Boundary Following.

        Parameters
        ----------
        x_start : list or list-like
            End-effector start position in task space
        x_target : list or list-like
            End-effector target position in task space
        dt : float
            Timestep of simulation.

        Returns
        -------
        None

        """

        # Set obstacle avoidance
        super().__init__(x_start, x_target)

        # Boundary Following start 3D coefficient
        self.K_boundaryStart = .6

        # Boundary Following continuation coefficient
        self.K_boundaryContinue = .3

        # Distance to goal when Obstacle avoidance starts
        self.boundaryDetectedGoalDistance = 0

        # Direction that ensures longitudinal movement
        self.longitudinalDirection = None

        # Set algorithm based parameters
        self.updateState(x_start)

        # Set dt from Robot
        self._dt = dt

    def getStatus(self):
        """Mentions the status whether all conditions are satisfied for the
        robot to ignore the obstacle and aim for the target only.

        Returns
        -------
        bool
            True if Boundary Following switch-out conditions are satisfied.
            See documentation.

        """

        # Check if Goal is closer now
        goalIsCloser = (self.boundaryDetectedGoalDistance > self.distanceGoal)

        # Check if Obstacle is behind the trajectory to goal
        pathIsObstacleFree = np.dot(self.toObstacle, self.toGoal) < 0

        # Check if Obstacle is really close
        obstacleIsClose = self.distanceObstacle < self.d_min

        # True only if obstacle is close by and
        # the robot has not crossed the obstacle yet
        return obstacleIsClose and not (goalIsCloser and pathIsObstacleFree)

    def plan(self, expTime):
        """Performs planning to generate reference velocity,
        with the current known state of robot's end-effector.
        Here, the plan is generated according to the boundary following
        algorithm.

        Parameters
        ----------
        expTime : float
            Simulation time in the experiment.

        Returns
        -------
        list or list-like
            Reference velocity in task space.

        """

        # Check if timestep is nearby
        if self._t is not None:
            timeSinceLastCall = expTime - self._t
        else:
            # Make this
            timeSinceLastCall = None

        # Update Last time
        self._t = expTime

        # If previous call was done just last time,
        # It should be considered as a continued boundary following
        # 1.5*dt ensures any calculation precision errors are correct
        if timeSinceLastCall is None or timeSinceLastCall > 1.5*self._dt:

            # Update Distance to goal when Obstacle avoidance starts
            self.boundaryDetectedGoalDistance = self.distanceGoal

            # Execute the algorithm which establishes the start
            # of boundary following
            x_dot_ref = self.boundaryFollowingStart()

            # Set longitudinal direction
            self.longitudinalDirection = x_dot_ref / norm(x_dot_ref)

        else:

            # Update Distance to goal continuing with Obstacle avoidance
            x_dot_ref = self.boundaryFollowingContinue()

        return x_dot_ref




    def boundaryFollowingStart(self):
        """When boundary following starts, the reference velocity is
        generated according to an arbitrary perpendicular direction.
        This function implements the concept.
        See documentation.

        Returns
        -------
        list or list-like
            Reference velocity in task space.

        """

        # Direction of avoid obstacle
        obs_dir = self.toObstacle / self.distanceObstacle

        # Orthogonal direction (randomly chosen - hardcoded)
        R = np.array([[0,1,0],
                      [-1,0,0],
                      [0,0,0]])

        # Distance perpendicular to avoid obstacle
        tangent = self.K_boundaryStart * R @ obs_dir

        # Add perpendicular line
        #p.addUserDebugLine(x, list(x + 4 * tangent), [0,0,1],4,2)

        return tangent

    def boundaryFollowingContinue(self):
        """Once boundary following has started and is on-going, a
        cross product of the first perpendicular direction and
        the direction to obstacle are taken. This generates a
        "longitudinal" direction for the boundary following.
        This methods implements the concept.
        See documentation

        Returns
        -------
        list or list-like
            Reference velocity in task space.

        """

        # Direction of obstacle avoidance
        obs_dir = self.toObstacle / self.distanceObstacle

        # Direction of improvement
        tangent = self.K_boundaryContinue * np.cross(obs_dir, self.longitudinalDirection)

        # Add trajectory tangent
        #p.addUserDebugLine(x, list(x + 4 * tangent), [0,1,1],5,2)

        return tangent
