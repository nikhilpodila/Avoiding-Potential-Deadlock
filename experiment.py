import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data
import numpy as np
import sys
import copy
from numpy.linalg import norm, pinv
from GoToGoal import AttractivePotential
from ObstacleAvoidance import ObstacleAlgorithm, \
                                RepulsivePotentialAlgorithm, \
                                BoundaryFollowingAlgorithm, \
                                TotalPotentialAlgorithm


# Class for the Kuka IIWA Experiment
class KukaRobotExperiment(object):
    """
    This class runs the KUKA IIWA Obstacle Avoidance experiment for Assignment 1
    and attempts to avoid deadlock.

    Attributes
    ----------

    ALGORITHM_REPULSIVE_POTENTIAL : int
        Input to constructor to run the Repulsive Potential algorithm for
        Obstacle avoidance.
    ALGORITHM_BOUNDARY_FOLLOWING : int
        Input to constructor to run the Boundary Following algorithm for
        Obstacle avoidance
    dt : float
        Time step for the simulation
    clid : int
        PyBullet parameter for connecting to simulator.
    prevPose : list or list-like
        Previous position of end-effector in task space.
    hasPrevPose : bool
        If True, the simulation has been executed for previous timestep, and
        position has been updated
    attractivePotential : AttractivePotential
        Class instance to execute attractive potential algorithm on robot
    obstacleAlgorithm : TotalPotentialAlgorithm or BoundaryFollowingAlgorithm
        Class instance to execute obstacle avoidance algorithm on robot.
        If TotalPotentialAlgorithm, Attractive and Repulsive Potential are performed
    robot : int
        PyBullet parameter to track bodyUniqueId
    robotNumJoints : int
        PyBullet parameter on number of joints on the robot
    robotEndEffectorIndex : int
        PyBullet parameter on the End-effector link's index value
    ll : list of int
        Lower limits of robot's joints in null space (size = robotNumJoints)
    ul : list of int
        Upper limits of robot's joints in null space (size = robotNumJoints)
    jr : list of int
        Joint ranges of robot's joints in null space (size = robotNumJoints)
    rp : list of int
        Rest poses of robot's joints in null space (size = robotNumJoints)
    jd : list of int
        Joint damping coefficients of robot's joints in null space
        (size = robotNumJoints)
    x : list or list-like
        Current robot's state in task space (End-effector position) in 3D space
    q : list or list-like
        Current robot's state in config space
        (Joint positions, size = robotNumJoints)
    J : numpy.ndarray
        Jacobian matrix based on the current joint positions
    trailDuration : int
        PyBullet parameter on number of steps to trail the debug lines
    x_start : list or list-like
        End-effector start position in task space
    x_target : list or list-like
        End-effector target position in task space
    t : float
        Current time in simulation since simulation started

    """

    # Class variables to choose between:
    # Potential field and Boundary following
    ALGORITHM_REPULSIVE_POTENTIAL = 0
    ALGORITHM_BOUNDARY_FOLLOWING = 1

    # Set timestep in simulation
    dt = 1/240.

    def __init__(self, obstacleAlgorithm, debug = False):
        """KukaRobotExperiment constructor. Initializes and runs the experiment

        Parameters
        ----------
        obstacleAlgorithm : int
            Algorithm used for Obstacle Avoidance. The possible values are
            class variables in this class.
        debug : bool
            If True, Debug lines are drawn in the simulation. (Default: False)

        Returns
        -------
        None

        """

        # Connect to PyBullet simulator
        self.clid = p.connect(p.SHARED_MEMORY)
        if self.clid < 0:
            p.connect(p.GUI)

        # Set PyBullet installed Data path for URDFs
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load objects
        self.loadObjects()

        # Obtain hardcoded limits, ranges and coefficents
        self.setRobotLimitsRanges()

        # Initialize Robot to rest position
        self.setJointStates(self.rp)

        # Hardcoded value for rest position of Kuka
        self.prevPose = [0, 0, 0]
        self.hasPrevPose = False

        # Initialize states
        self.initializeParamsAndState()

        # Set Go to goal algorithm
        self.attractivePotential = AttractivePotential(self.x_start, self.x_target)

        # Set algorithm for Obstacle avoidance
        if obstacleAlgorithm == self.ALGORITHM_REPULSIVE_POTENTIAL:
            self.obstacleAlgorithm = TotalPotentialAlgorithm(self.x_start, self.x_target)
        elif obstacleAlgorithm == self.ALGORITHM_BOUNDARY_FOLLOWING:
            self.obstacleAlgorithm = BoundaryFollowingAlgorithm(self.x_start, self.x_target, self.dt)
        else:
            raise Exception("Algorithm type not implemented")

        # Conduct experiment
        self.experiment(debug)

        # Show experiment results
        self.experimentResults()

    def loadObjects(self):
        """ Loads the required models on the simulator and
        sets simulator paremeters

        Returns
        -------
        None

        """

        # Load floor plane at -2
        p.loadURDF("plane.urdf",[0,0,-2])

        # Load Robot
        self.robot = p.loadURDF("kuka_iiwa/model.urdf",[0,0,0])
        p.resetBasePositionAndOrientation(
            self.robot,
            [0, 0, 0],
            [0, 0, 0, 1]
        )

        # Joints and End effector Index
        self.robotNumJoints = p.getNumJoints(self.robot)
        self.robotEndEffectorIndex = 6
        assert self.robotNumJoints == 7, "Model incorrect"

        # Camera adjustment
        p.resetDebugVisualizerCamera(
            cameraDistance = 3,
            cameraYaw = 230,
            cameraPitch = -22,
            cameraTargetPosition = [0,0,0]
        )

        # Gravity setting
        p.setGravity(0, 0, 0)

        # Is Simulation Real Time?
        p.setRealTimeSimulation(0)

    def setRobotLimitsRanges(self):
        """Sets the Lower limits, upper limits, joint ranges and
        rest poses for the null space of the robot. Hardcoded values here.

        Returns
        -------
        None

        """

        # lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]

        # upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]

        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]

        # restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]

        # joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    def updateState(self, updateJ = False):
        """Retrieves/Measures the end-effector position
        and derives the configuration position and Jacobian matrix from it.
        (Ideally, configuration position can also be measured directly)


        Parameters
        ----------
        updateJ : bool
            If True, Jacobian matrix is updated based on measurements
            (Default: False)

        Returns
        -------
        None

        """

        # Get link state
        linkState = p.getLinkState(
                        self.robot,
                        self.robotEndEffectorIndex,
                        computeLinkVelocity = 1,
                        computeForwardKinematics = 1
        )

        # Save x value and find q
        self.x = linkState[4]
        self.q = self.ik(self.x)

        # Calculate Jacobian
        if updateJ:
            J, _ = p.calculateJacobian(
                        bodyUniqueId = self.robot,
                        linkIndex = self.robotEndEffectorIndex,
                        localPosition = list(linkState[2]),
                        objPositions = list(self.q),
                        objVelocities = [0.] * len(list(self.q)),
                        objAccelerations = [0.] * len(list(self.q))
            )

            self.J = np.array(J)


    def setJointStates(self, q):
        """Hard set joint values. This is used to reset simulations or
        perform manipulations outside an experiment.

        Parameters
        ----------
        q : list or list-like of floats (size = number of joints of robot)
            Configuration states to reset the joints to.

        Returns
        -------
        None

        """

        # Set each joint's states
        for jointNumber in range(self.robotNumJoints):
            p.resetJointState(self.robot, jointNumber, float(q[jointNumber]))

    def ik(self, x):
        """Performs Inverse Kinematics to obtain the joint positions
        from the end-effector position (Config space from Task space)

        Parameters
        ----------
        x : list or list-like of 3 floats
            Task space state (End-effector position) in 3D space

        Returns
        -------
        numpy.array
            Configuration space state (Joint positions)
            Array of size = number of joints on robot

        """

        q = p.calculateInverseKinematics(
                        bodyUniqueId = self.robot,
                        endEffectorLinkIndex = self.robotEndEffectorIndex,
                        targetPosition = list(x),
                        lowerLimits = self.ll,
                        upperLimits = self.ul,
                        jointRanges = self.jr,
                        restPoses = self.rp
                        )

        return np.array(q)


    def initializeParamsAndState(self):
        """Initializes parameters such as start and target states.
        Also initilizes simulation parameters

        Returns
        -------
        None

        """

        # Trail debug line delay
        self.trailDuration = 15

        # Start state
        self.x_start = np.random.uniform(
                        low = [-.4, -.3, .7],
                        high = [-.4, -.2, .8],
                        size = 3
        )

        # Set current state to start state
        self.x = copy.deepcopy(self.x_start)
        self.q = self.ik(self.x_start)

        # Update states on robot
        self.setJointStates(self.q)

        # Target state
        self.x_target = np.random.uniform(
                        low = [-.4, .2, .2],
                        high = [-.4, .3, .3],
                        size = 3
        )

        # Initialize time
        self.t = 0


    def setRobotTaskReference(self, x_dot_ref):
        """Specifies motor command given the reference velocity in
        Task space (end effector reference velocity).
        It convert task space reference to joint space reference using
        the pseudo-inverse of Jacobian matrix.

        Parameters
        ----------
        x_dot_ref : list or list-like
            Reference velocity in task space. Must be of size 3.

        Returns
        -------
        None

        """

        # Task to Joint Reference
        q_dot_ref = pinv(self.J) @ x_dot_ref

        # Set joint reference for each joint
        for robotJoint in range(self.robotNumJoints):
            p.setJointMotorControl2(
                bodyIndex = self.robot,
                jointIndex = robotJoint,
                controlMode = p.VELOCITY_CONTROL,
                targetVelocity = float(q_dot_ref[robotJoint]),
                force = 500,
                positionGain = 0.03,
                velocityGain = 0.01
            )


    def experiment(self, debug):
        """Performs the experiment after all the initializations.

        Parameters
        ----------
        debug : bool
            If True, Debug lines are drawn in the simulation.
            (Default: False)

        Returns
        -------
        None

        """

        # Continue motion until target is reached OR if time exceeds 90s
        while norm(self.x - self.x_target) > 1e-3 and self.t < 90 / self.dt:

            # Update timestep
            self.t += self.dt

            # Step simulation seems to not pause the simulation for dt time
            # when debug is false. Corrects that term here.
            if not debug:
                time.sleep(self.dt)

            # Perform simulation in this step
            p.stepSimulation()

            # Obtain robot state by updating the parameters from measurements.
            self.updateState(updateJ = True)
            self.obstacleAlgorithm.updateState(self.x)
            self.attractivePotential.updateState(self.x)


            # HYBRID ALGORITHM:
            # If Obstacle is not reached, Attractive Potential is used
            # If the Obstacle is reached, an Obstacle Algorithm is used
            if self.obstacleAlgorithm.getStatus():

                # Perform planning and obtain the reference velocity
                x_dot_ref = self.obstacleAlgorithm.plan(self.t)

                # Draw line for tangents to reference velocity while debugging
                if debug:
                    p.addUserDebugLine(self.x, list(self.x + 4 * x_dot_ref), [0, 1, 1], 5, 2)

            else:

                # Perform planning and obtain the reference velocity
                x_dot_ref = self.attractivePotential.plan()

            # Move the robot joints based on given reference velocity.
            self.setRobotTaskReference(x_dot_ref)

            # Draw trail line of the end-effector path while debugging
            if debug and self.hasPrevPose:
                p.addUserDebugLine(self.prevPose, self.x, [1, 0, 0], 1, self.trailDuration)

            # Keep track of previous iteration's position.
            self.prevPose = self.x
            self.hasPrevPose = True

    def experimentResults(self):
        """Generates and displays any experiment results that are shown
        after the simulation.

        Returns
        -------
        None

        """

        print("Reached target in: ",self.t," seconds")
        time.sleep(10)
        p.disconnect()

# Runs the following code when run from command line
if __name__ == "__main__":

    # Run experiment

    algorithm = None

    if "repulsive" in sys.argv:
        algorithm = KukaRobotExperiment.ALGORITHM_REPULSIVE_POTENTIAL
    else:
        algorithm = KukaRobotExperiment.ALGORITHM_BOUNDARY_FOLLOWING

    debug = "debug" in sys.argv

    # This is the line to be modified to get different results
    KukaRobotExperiment(algorithm, debug = debug)
