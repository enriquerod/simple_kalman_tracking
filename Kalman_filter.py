from numpy import * 
from numpy.linalg import inv

class KalmanTrack(object):
    def __init__(self):
        self.dt = 1
        self.A = array([[1, self.dt, 0 , 0], [0, 1, 0, 0], [0, 0, 1, self.dt], [0, 0, 0,1]])
        self.H = array([[1, 0, 0, 0],[0, 0 , 1, 0]])
        # self.Q = eye(4)
        self.Q = 1.0*eye(4)
        # self.R = array([[50, 0],[0, 50]])
        self.R = array([[5, 0],[0, 5]])

        self.x = array([[0, 0, 0, 0]]).T
        self.P = 100*eye(4)
  
    
    def __call__(self, xm , ym):
        #Prediction Step
        #Project the state ahead
        xp = dot(self.A,self.x)
        #Project the error covariance ahead
        Pp = dot(self.A, dot(self.P, self.A.T)) + self.Q

        #Update Step
        #Compute the Kalman gain
        K = dot(Pp, dot(self.H.T, inv(dot(self.H, dot(Pp, self.H.T)) + self.R)))
        z = array([[xm, ym]]).T
        #Update estimate with measurement
        self.x = xp + dot(K, (z - dot(self.H, xp)))
        #Update the error covariance
        self.P = Pp - dot(K, dot(self.H,Pp))

        return self.x[0], self.x[2]




