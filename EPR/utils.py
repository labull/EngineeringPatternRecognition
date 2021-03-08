import numpy as np

class ellipse:
    # points around 2d covariance ellipse
    def __init__(self, mu, cov, points=500, MSD_th=None):
        theta = np.linspace(0, np.pi * 2, points)
        val, vec = np.linalg.eig(cov)
        a = np.arctan(vec[1, 0] / vec[0, 0]) # angle of fist eigenvector
        # std_dev ellipses
        self.cov_3 = []
        for j in range(3):
            cx = (j+1) * np.sqrt(val[0]) * np.cos(theta)
            cy = (j+1) * np.sqrt(val[1]) * np.sin(theta)
            cr = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.vstack((cx, cy))
            self.cov_3.append(cr.T + mu)
        # MSD ellipses - 95%       
        cx = np.sqrt(5.991 * val[0]) * np.cos(theta)
        cy = np.sqrt(5.991 * val[1]) * np.sin(theta)
        cr = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.vstack((cx, cy))
        self.MSDellipse_95 = cr.T + mu
        # MSD custom threshold
        if MSD_th is not None:
            cx = np.sqrt(MSD_th * val[0]) * np.cos(theta)
            cy = np.sqrt(MSD_th * val[1]) * np.sin(theta)
            cr = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.vstack((cx, cy))
            self.MSD_th = cr.T + mu

class struct(dict):
    """
    Credit to: epool on stackoverflow
    Example:
    m = struct({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(struct, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(struct, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(struct, self).__delitem__(key)
        del self.__dict__[key]

