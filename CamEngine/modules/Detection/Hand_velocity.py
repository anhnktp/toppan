import  numpy as np

class HandVelocity:

    def __init__(self, handTracker, setHandId):
        self.handTracker = handTracker
        self.setHandId = setHandId

    def calculate_velocity(self, hands):
        for hand in hands:
            hand_center = hand[-1]
            hand_id = hand[-3]
            hand_time = hand[-2]
            self.setHandId.add(hand_id)
            for id in self.setHandId:
                if id not in self.handTracker.keys():
                    self.handTracker[id] = []
                if id == hand_id:
                    if len(self.handTracker[id]) > 1:
                        self.handTracker[id].pop(0)
                    self.handTracker[id].append([hand_center, hand_time])

                    if len(self.handTracker[id]) == 2:
                        c1 = self.handTracker[hand_id][0][0]
                        c2 = self.handTracker[hand_id][1][0]
                        deltaT = self.handTracker[hand_id][1][1] - self.handTracker[hand_id][0][1]
                        velo = (np.linalg.norm(np.array(c1) - np.array(c2))) / deltaT
                        vx = (c2[0] - c1[0]) / deltaT
                        vy = (c2[1] - c1[1]) / deltaT

                        hand.insert(6, vx)
                        hand.insert(7, vy)
                        hand.insert(8, velo)
                        width = hand[2] - hand[0]
                        height = hand[3] - hand[1]
                        if vx != 0:
                            xc = int(hand[-1][0] + width * 0.25 * (vx / abs(vx)))
                        else:
                            xc = hand[-1][0]

                        yc = int(hand[-1][1] - height * 0.25)
                        hand[-1] = (xc, yc)
                        # hand: [xmin, ymin, xmax, ymax, id, time, vx, vy, velo, (xc,yc)]

        return hands