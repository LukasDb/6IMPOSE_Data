from scipy.spatial.transform import Rotation as R
from ..placeable import Placeable


def randomize_rotation(subject: Placeable):
    subject.set_rotation(R.random())
