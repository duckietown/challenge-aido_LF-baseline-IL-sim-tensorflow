import math

REF_VELOCITY = 0.35

K_P = 10
K_D = 8.5


class PurePursuitExpert:
    def __init__(self, env,
                 ref_velocity=REF_VELOCITY):
        self.env = env
        self.ref_velocity = ref_velocity

    def predict(self, observation):
        lane_pose = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
        distance_to_road_center = lane_pose.dist
        angle_from_straight_in_rads = lane_pose.angle_rad

        steering = K_P * distance_to_road_center + K_D * angle_from_straight_in_rads

        action = [self.ref_velocity, steering]

        return action
